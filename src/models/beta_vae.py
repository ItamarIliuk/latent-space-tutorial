"""
Beta-VAE: VAE com peso ajustável para KL divergence
"""

import torch
from .vae import VAE, vae_loss


class BetaVAE(VAE):
    """
    Beta-VAE: VAE com hiperparâmetro β para controlar o trade-off
    entre reconstrução e regularização.
    
    β > 1: Mais regularização → Melhor disentanglement, pior reconstrução
    β < 1: Menos regularização → Melhor reconstrução, pior separação
    β = 1: VAE padrão
    
    Args:
        input_dim (int): Dimensão da entrada
        latent_dim (int): Dimensão do espaço latente
        beta (float): Peso da KL divergence (padrão: 1.0)
        hidden_dims (list): Dimensões das camadas ocultas
    
    Exemplo:
        >>> # Beta alto para melhor disentanglement
        >>> beta_vae = BetaVAE(input_dim=784, latent_dim=10, beta=4.0)
        >>> 
        >>> # Beta baixo para melhor reconstrução
        >>> beta_vae_recon = BetaVAE(input_dim=784, latent_dim=10, beta=0.5)
    
    Referência:
        Higgins et al. "beta-VAE: Learning Basic Visual Concepts with a 
        Constrained Variational Framework" (ICLR 2017)
    """
    
    def __init__(self, input_dim=784, latent_dim=2, beta=1.0, hidden_dims=None):
        super(BetaVAE, self).__init__(input_dim, latent_dim, hidden_dims)
        self.beta = beta
    
    def loss_function(self, x_recon, x_orig, mu, logvar):
        """
        Calcula a perda do Beta-VAE.
        
        Args:
            x_recon: Reconstrução
            x_orig: Entrada original
            mu: Média da distribuição latente
            logvar: Log-variância da distribuição latente
        
        Returns:
            dict: Dicionário com componentes da perda
        """
        return vae_loss(x_recon, x_orig, mu, logvar, beta=self.beta)
    
    def set_beta(self, beta):
        """
        Ajusta o valor de β durante o treinamento.
        
        Args:
            beta (float): Novo valor de β
        """
        self.beta = beta
    
    def __repr__(self):
        return (f"BetaVAE(input_dim={self.input_dim}, "
                f"latent_dim={self.latent_dim}, "
                f"beta={self.beta})")


class AnnealedBetaVAE(BetaVAE):
    """
    Beta-VAE com annealing (aumento gradual de β durante o treinamento).
    
    Útil para treinar VAEs mais estáveis, começando com β baixo e
    aumentando gradualmente.
    
    Args:
        input_dim (int): Dimensão da entrada
        latent_dim (int): Dimensão do espaço latente
        beta_start (float): Valor inicial de β
        beta_end (float): Valor final de β
        anneal_epochs (int): Número de épocas para atingir beta_end
        hidden_dims (list): Dimensões das camadas ocultas
    
    Exemplo:
        >>> vae = AnnealedBetaVAE(
        ...     input_dim=784,
        ...     latent_dim=10,
        ...     beta_start=0.0,
        ...     beta_end=4.0,
        ...     anneal_epochs=50
        ... )
        >>> 
        >>> # Durante o treinamento
        >>> for epoch in range(100):
        ...     vae.update_beta(epoch)
        ...     # ... treinar ...
    """
    
    def __init__(self, input_dim=784, latent_dim=2, 
                 beta_start=0.0, beta_end=1.0, anneal_epochs=50,
                 hidden_dims=None):
        super(AnnealedBetaVAE, self).__init__(
            input_dim, latent_dim, beta_start, hidden_dims
        )
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.anneal_epochs = anneal_epochs
        self.current_epoch = 0
    
    def update_beta(self, epoch):
        """
        Atualiza β baseado na época atual.
        
        Args:
            epoch (int): Época atual do treinamento
        """
        self.current_epoch = epoch
        
        if epoch >= self.anneal_epochs:
            self.beta = self.beta_end
        else:
            # Annealing linear
            progress = epoch / self.anneal_epochs
            self.beta = self.beta_start + (self.beta_end - self.beta_start) * progress
    
    def __repr__(self):
        return (f"AnnealedBetaVAE(input_dim={self.input_dim}, "
                f"latent_dim={self.latent_dim}, "
                f"beta={self.beta:.2f} (epoch {self.current_epoch}))")
