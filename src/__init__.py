"""
Latent Space Tutorial
Tutorial completo sobre Espaço Latente, Autoencoders e VAEs
"""

__version__ = "1.0.0"
__author__ = "Profa. Itamar"
__email__ = "itamar@utfpr.edu.br"

from .models import Autoencoder, VAE, BetaVAE
from .utils import load_mnist, train_model, train_vae
from .experiments import LatentExplorer, BetaVAEComparison

__all__ = [
    "Autoencoder",
    "VAE",
    "BetaVAE",
    "load_mnist",
    "train_model",
    "train_vae",
    "LatentExplorer",
    "BetaVAEComparison",
]
