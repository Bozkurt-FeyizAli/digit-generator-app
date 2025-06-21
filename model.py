import torch
import torch.nn as nn
from torch.nn import functional as F

class CVAE(nn.Module):
    def __init__(self, img_shape, num_classes, latent_dim):
        super(CVAE, self).__init__()
        self.img_shape = img_shape
        self.num_classes = num_classes
        self.latent_dim = latent_dim

        # Encoder
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(img_shape[0], 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(32 * 7 * 7, latent_dim)
        self.fc_log_var = nn.Linear(32 * 7 * 7, latent_dim)

        # Decoder
        self.decoder_fc = nn.Linear(latent_dim + num_classes, 32 * 7 * 7)
        self.decoder_deconv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, img_shape[0], kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, y):
        # One-hot encode the label y
        y_one_hot = F.one_hot(torch.tensor(y), self.num_classes).float().view(1, -1)
        z_cond = torch.cat([z, y_one_hot], dim=1)
        
        decoder_hidden = self.decoder_fc(z_cond)
        decoder_hidden_reshaped = decoder_hidden.view(-1, 32, 7, 7)
        recon_x = self.decoder_deconv(decoder_hidden_reshaped)
        return recon_x

    def forward(self, x, y):
        # Encoder
        conv_out = self.encoder_conv(x)
        flat_out = conv_out.view(conv_out.size(0), -1)
        mu = self.fc_mu(flat_out)
        log_var = self.fc_log_var(flat_out)
        
        # Reparameterization
        z = self.reparameterize(mu, log_var)
        
        # Decoder
        y_one_hot = F.one_hot(y, self.num_classes).float()
        z_cond = torch.cat([z, y_one_hot], dim=1)
        
        decoder_hidden = self.decoder_fc(z_cond)
        decoder_hidden_reshaped = decoder_hidden.view(-1, 32, 7, 7)
        recon_x = self.decoder_deconv(decoder_hidden_reshaped)
        
        return recon_x, mu, log_var