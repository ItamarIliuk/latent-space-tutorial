"""
Utilitários: data loading, training, visualization
"""

from .data_loader import load_mnist, MNISTDataModule
from .training import train_model, train_vae, EarlyStopping
from .visualization import (
    visualize_latent_space,
    plot_reconstructions,
    plot_vae_results,
    plot_latent_grid,
    plot_interpolation
)

__all__ = [
    "load_mnist",
    "MNISTDataModule",
    "train_model",
    "train_vae",
    "EarlyStopping",
    "visualize_latent_space",
    "plot_reconstructions",
    "plot_vae_results",
    "plot_latent_grid",
    "plot_interpolation",
]
