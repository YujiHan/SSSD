import torch
from torch import nn
from torch.nn import functional as F


class VanillaVAE(nn.Module):

    def __init__(self, input_features, latent_dim, hidden_dims=None, **kwargs):
        super(VanillaVAE, self).__init__()

        self.latent_dim = latent_dim
        if hidden_dims is None:
            hidden_dims = [1024, 512, 256]

        # Build Encoder
        modules = []
        layer_features = input_features

        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(layer_features, out_features=h_dim),
                    nn.BatchNorm1d(h_dim),
                    nn.ReLU(),
                )
            )
            layer_features = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)

        # Build Decoder
        modules = []
        hidden_dims.reverse()
        layer_features = latent_dim

        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(layer_features, out_features=h_dim),
                    nn.BatchNorm1d(h_dim),
                    nn.ReLU(),
                )
            )
            layer_features = h_dim

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Linear(hidden_dims[-1], input_features)

    def encode(self, input):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x F]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z):
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [N x F]
        """
        result = self.decoder(z)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input):
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        result = self.decode(z)
        return [result, input, mu, log_var]

    def loss_function(self, recons, input, mu, log_var, kld_weight=0):
        recons_loss = F.mse_loss(recons, input)

        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0
        )

        loss = recons_loss + kld_weight * kld_loss
        return {
            'loss': loss,
            'Reconstruction_Loss': recons_loss.detach(),
            'KLD': -kld_loss.detach(),
        }

    def sample(self, num_samples, current_device):
        z = torch.randn(num_samples, self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x):
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]
