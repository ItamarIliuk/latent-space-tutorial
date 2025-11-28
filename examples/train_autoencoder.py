"""
Train Autoencoder: Script completo para treinar um Autoencoder

Este script treina um Autoencoder com opções configuráveis via linha de comando.
"""

import torch
import argparse
import os
from src.models.autoencoder import Autoencoder
from src.utils.data_loader import load_mnist
from src.utils.training import train_model, evaluate_model
from src.utils.visualization import (
    visualize_latent_space,
    plot_reconstructions,
    plot_training_history
)


def parse_args():
    """Parse argumentos de linha de comando."""
    parser = argparse.ArgumentParser(
        description='Train an Autoencoder on MNIST dataset'
    )

    # Hiperparâmetros do modelo
    parser.add_argument('--latent-dim', type=int, default=2,
                       help='Latent space dimension (default: 2)')
    parser.add_argument('--hidden-dims', type=int, nargs='+',
                       default=[512, 256, 128],
                       help='Hidden layer dimensions (default: 512 256 128)')

    # Hiperparâmetros de treinamento
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs (default: 50)')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Batch size (default: 128)')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--early-stopping', type=int, default=None,
                       help='Early stopping patience (default: None)')

    # Caminhos e dispositivo
    parser.add_argument('--data-dir', type=str, default='./data',
                       help='Directory for MNIST data (default: ./data)')
    parser.add_argument('--output-dir', type=str, default='./output',
                       help='Directory for outputs (default: ./output)')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use (default: auto)')

    # Flags
    parser.add_argument('--no-visualize', action='store_true',
                       help='Skip visualization generation')
    parser.add_argument('--save-model', action='store_true',
                       help='Save trained model')

    return parser.parse_args()


def main():
    """Função principal."""
    args = parse_args()

    # Configura device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    print("=" * 70)
    print("TRAINING AUTOENCODER")
    print("=" * 70)

    print("\nConfiguration:")
    print(f"  Latent Dimension: {args.latent_dim}")
    print(f"  Hidden Dimensions: {args.hidden_dims}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Learning Rate: {args.lr}")
    print(f"  Early Stopping: {args.early_stopping}")
    print(f"  Device: {device}")
    print(f"  Output Directory: {args.output_dir}")

    # Cria diretório de output
    os.makedirs(args.output_dir, exist_ok=True)

    # Carrega dados
    print("\n[1/5] Loading MNIST dataset...")
    train_loader, val_loader, test_loader = load_mnist(
        batch_size=args.batch_size,
        data_dir=args.data_dir,
        num_workers=0
    )
    print(f"  Train: {len(train_loader.dataset)} samples")
    print(f"  Val: {len(val_loader.dataset)} samples")
    print(f"  Test: {len(test_loader.dataset)} samples")

    # Cria modelo
    print("\n[2/5] Creating Autoencoder...")
    model = Autoencoder(
        input_dim=784,
        latent_dim=args.latent_dim,
        hidden_dims=args.hidden_dims
    )
    print(f"  {model}")
    print(f"  Compression ratio: {model.compression_ratio():.1f}x")

    # Conta parâmetros
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_params:,}")

    # Treina
    print("\n[3/5] Training model...")
    save_path = os.path.join(args.output_dir, 'best_autoencoder.pt') if args.save_model else None

    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        device=device,
        early_stopping_patience=args.early_stopping,
        save_path=save_path,
        verbose=True
    )

    print("\n  Training completed!")
    print(f"  Final train loss: {history['train_loss'][-1]:.6f}")
    if history['val_loss']:
        print(f"  Final val loss: {history['val_loss'][-1]:.6f}")
        print(f"  Best val loss: {min(history['val_loss']):.6f}")

    # Avalia no teste
    print("\n[4/5] Evaluating on test set...")
    test_metrics = evaluate_model(model, test_loader, device=device, is_vae=False)
    print(f"  Test loss: {test_metrics['loss']:.6f}")

    # Visualizações
    if not args.no_visualize:
        print("\n[5/5] Generating visualizations...")

        # História de treinamento
        print("  - Plotting training history...")
        plot_training_history(
            history,
            save_path=os.path.join(args.output_dir, 'training_history.png'),
            show=False
        )

        # Espaço latente (apenas se 2D)
        if args.latent_dim == 2:
            print("  - Plotting latent space...")
            visualize_latent_space(
                model, test_loader, device=device,
                save_path=os.path.join(args.output_dir, 'latent_space.png'),
                show=False
            )

        # Reconstruções
        print("  - Plotting reconstructions...")
        plot_reconstructions(
            model, test_loader, n_samples=10, device=device,
            save_path=os.path.join(args.output_dir, 'reconstructions.png'),
            show=False
        )

        print(f"\n  Visualizations saved to {args.output_dir}/")
    else:
        print("\n[5/5] Skipping visualizations (--no-visualize)")

    # Salva modelo final
    if args.save_model:
        final_path = os.path.join(args.output_dir, 'final_autoencoder.pt')
        torch.save(model.state_dict(), final_path)
        print(f"\n  Model saved to {final_path}")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETED!")
    print("=" * 70)

    print("\nResults:")
    print(f"  Test Loss: {test_metrics['loss']:.6f}")
    print(f"  Compression: {args.latent_dim} dimensions ({model.compression_ratio():.1f}x)")

    if not args.no_visualize:
        print(f"\nGenerated files in {args.output_dir}/:")
        print("  - training_history.png")
        if args.latent_dim == 2:
            print("  - latent_space.png")
        print("  - reconstructions.png")
        if args.save_model:
            print("  - best_autoencoder.pt")
            print("  - final_autoencoder.pt")

    print("\nNext steps:")
    print("  1. Check the generated visualizations")
    print("  2. Try different latent dimensions: --latent-dim 10")
    print("  3. Experiment with architecture: --hidden-dims 256 128 64")
    print("  4. Compare with VAE: python examples/train_vae.py")

    print("=" * 70)


if __name__ == "__main__":
    main()
