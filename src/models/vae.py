"""
VAE: Variational Autoencoder com espaço latente probabilístico
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    """
    Variational Autoencoder (VAE) para geração e reconstrução.
    
    Diferente do Autoencoder tradicional, o VAE aprende uma distribuição 
    probabilística no espaço latente, permitindo geração de novas amostras.
    
    Args:
        input_dim (int): Dimensão da entrada (padrão: 784 para MNIST)
        latent_dim (int): Dimensão do espaço latente
        hidden_dims (list): Dimensões das camadas ocultas
    
    Exemplo:
        >>> vae = VAE(input_dim=784, latent_dim=2)
        >>> x = torch.randn(32, 784)
        >>> x_recon, mu, logvar, z = vae(x)
        >>> # Gerar novas amostras
        >>> z_sample = torch.randn(10, 2)
        >>> samples = vae.decode(z_sample)
    """
    
    def __init__(self, input_dim=784, latent_dim=2, hidden_dims=None):
        super(VAE, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [512, 256]
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        
        # Encoder base (comum para μ e σ)
        encoder_layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
            ])
            prev_dim = hidden_dim
        
        self.encoder_base = nn.Sequential(*encoder_layers)
        
        # Duas "cabeças" para μ (média) e log(σ²) (log-variância)
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)
        
        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
            ])
            prev_dim = hidden_dim
        
        decoder_layers.extend([
            nn.Linear(prev_dim, input_dim),
            nn.Sigmoid()
        ])
        
        self.decoder = nn.Sequential(*decoder_layers)
    
    def encode(self, x):
        """
        Codifica entrada para parâmetros da distribuição latente.
        
        Args:
            x (torch.Tensor): Entrada (batch_size, input_dim)
        
        Returns:
            tuple: (mu, logvar) - parâmetros da distribuição N(μ, σ²)
        """
        h = self.encoder_base(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """
        Truque de reparametrização: z = μ + ε·σ
        
        Permite backpropagation através da operação de amostragem.
        
        Args:
            mu (torch.Tensor): Média da distribuição
            logvar (torch.Tensor): Log-variância da distribuição
        
        Returns:
            torch.Tensor: Amostra do espaço latente z
        """
        std = torch.exp(0.5 * logvar)  # σ = exp(0.5 * log(σ²))
        eps = torch.randn_like(std)    # ε ~ N(0, 1)
        z = mu + eps * std             # z = μ + ε·σ
        return z
    
    def decode(self, z):
        """
        Decodifica do espaço latente.
        
        Args:
            z (torch.Tensor): Códigos latentes (batch_size, latent_dim)
        
        Returns:
            torch.Tensor: Reconstrução (batch_size, input_dim)
        """
        return self.decoder(z)
    
    def forward(self, x):
        """
        Forward pass completo do VAE.
        
        Args:
            x (torch.Tensor): Entrada (batch_size, input_dim)
        
        Returns:
            tuple: (reconstrução, mu, logvar, z)
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z)
        return reconstructed, mu, logvar, z
    
    def sample(self, num_samples=1, device='cpu'):
        """
        Gera novas amostras do prior p(z) = N(0, I).
        
        Args:
            num_samples (int): Número de amostras a gerar
            device: Dispositivo para gerar as amostras
        
        Returns:
            torch.Tensor: Amostras geradas (num_samples, input_dim)
        """
        self.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim).to(device)
            samples = self.decode(z)
        return samples
    
    def interpolate(self, x1, x2, num_steps=10):
        """
        Interpola entre duas entradas no espaço latente.
        
        Args:
            x1, x2 (torch.Tensor): Duas entradas
            num_steps (int): Número de passos na interpolação
        
        Returns:
            torch.Tensor: Imagens interpoladas
        """
        self.eval()
        with torch.no_grad():
            mu1, _ = self.encode(x1.unsqueeze(0) if x1.dim() == 1 else x1)
            mu2, _ = self.encode(x2.unsqueeze(0) if x2.dim() == 1 else x2)
            
            # Interpolação linear entre z1 e z2
            alphas = torch.linspace(0, 1, num_steps)
            interpolated = []
            
            for alpha in alphas:
                z = (1 - alpha) * mu1 + alpha * mu2
                img = self.decode(z)
                interpolated.append(img)
            
            return torch.cat(interpolated, dim=0)
    
    def get_latent_representation(self, x):
        """
        Obtém a representação latente (μ) sem amostragem.
        
        Args:
            x (torch.Tensor): Entrada
        
        Returns:
            torch.Tensor: Média da distribuição latente
        """
        self.eval()
        with torch.no_grad():
            mu, _ = self.encode(x)
        return mu
    
    def __repr__(self):
        return (f"VAE(input_dim={self.input_dim}, "
                f"latent_dim={self.latent_dim})")


def vae_loss(x_recon, x_orig, mu, logvar, beta=1.0):
    """
    Função de perda do VAE: Reconstruction Loss + β * KL Divergence
    
    Args:
        x_recon: Reconstrução
        x_orig: Entrada original
        mu: Média da distribuição latente
        logvar: Log-variância da distribuição latente
        beta: Peso da KL divergence (Beta-VAE)
    
    Returns:
        dict: Dicionário com 'total', 'reconstruction', 'kl'
    """
    # Perda de reconstrução (BCE)
    recon_loss = F.binary_cross_entropy(x_recon, x_orig, reduction='sum')
    
    # KL Divergence: KL(q(z|x) || p(z))
    # onde q(z|x) = N(μ, σ²) e p(z) = N(0, I)
    # KL = -0.5 * Σ(1 + log(σ²) - μ² - σ²)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Perda total
    total_loss = recon_loss + beta * kl_loss
    
    return {
        'total': total_loss,
        'reconstruction': recon_loss,
        'kl': kl_loss
    }
