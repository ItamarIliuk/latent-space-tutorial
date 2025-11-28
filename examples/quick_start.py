"""
Quick Start: Exemplo rápido de uso do tutorial de espaço latente

Este script demonstra o uso básico da biblioteca em menos de 50 linhas.
"""

import torch
from src.models.vae import VAE
from src.utils.data_loader import load_mnist
from src.utils.training import train_vae
from src.utils.visualization import (
    visualize_latent_space,
    plot_reconstructions,
    plot_vae_results
)


def main():
    """Exemplo rápido de treinamento e visualização de um VAE."""

    print("=" * 60)
    print("LATENT SPACE TUTORIAL - QUICK START")
    print("=" * 60)

    # Configurações
    LATENT_DIM = 2
    NUM_EPOCHS = 10
    BATCH_SIZE = 128
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"\nConfigurations:")
    print(f"  Device: {DEVICE}")
    print(f"  Latent Dimension: {LATENT_DIM}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Batch Size: {BATCH_SIZE}")

    # 1. Carrega dados
    print("\n[1/4] Loading MNIST dataset...")
    train_loader, val_loader, test_loader = load_mnist(
        batch_size=BATCH_SIZE,
        num_workers=0
    )
    print(f"  Training samples: {len(train_loader.dataset)}")
    print(f"  Validation samples: {len(val_loader.dataset)}")
    print(f"  Test samples: {len(test_loader.dataset)}")

    # 2. Cria modelo
    print("\n[2/4] Creating VAE model...")
    vae = VAE(input_dim=784, latent_dim=LATENT_DIM)
    print(f"  Model: {vae}")

    # 3. Treina modelo
    print("\n[3/4] Training model...")
    history = train_vae(
        model=vae,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=NUM_EPOCHS,
        learning_rate=1e-3,
        beta=1.0,
        device=DEVICE,
        verbose=True
    )

    print("\n  Training completed!")
    print(f"  Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"  Final val loss: {history['val_loss'][-1]:.4f}")

    # 4. Visualiza resultados
    print("\n[4/4] Generating visualizations...")

    # Visualização completa
    print("  - Plotting VAE results overview...")
    plot_vae_results(vae, test_loader, device=DEVICE, save_path='vae_results.png')

    # Espaço latente
    print("  - Plotting latent space...")
    visualize_latent_space(vae, test_loader, device=DEVICE,
                          save_path='latent_space.png', show=False)

    # Reconstruções
    print("  - Plotting reconstructions...")
    plot_reconstructions(vae, test_loader, n_samples=10, device=DEVICE,
                        save_path='reconstructions.png', show=False)

    print("\n" + "=" * 60)
    print("QUICK START COMPLETED!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - vae_results.png")
    print("  - latent_space.png")
    print("  - reconstructions.png")
    print("\nNext steps:")
    print("  1. Check the generated images")
    print("  2. Try different latent dimensions (LATENT_DIM)")
    print("  3. Experiment with more epochs")
    print("  4. Explore the notebooks in notebooks/ directory")
    print("=" * 60)


if __name__ == "__main__":
    main()
