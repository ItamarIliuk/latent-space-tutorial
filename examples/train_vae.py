"""
Train VAE: Script completo para treinar um VAE ou Beta-VAE

Este script treina um Variational Autoencoder com opções configuráveis.
"""

import torch
import argparse
import os
from src.models.vae import VAE
from src.models.beta_vae import BetaVAE, AnnealedBetaVAE
from src.utils.data_loader import load_mnist
from src.utils.training import train_vae, evaluate_model
from src.utils.visualization import (
    visualize_latent_space,
    plot_vae_results,
    plot_latent_grid,
    plot_training_history
)


def parse_args():
    """Parse argumentos de linha de comando."""
    parser = argparse.ArgumentParser(
        description='Train a VAE or Beta-VAE on MNIST dataset'
    )

    # Tipo de modelo
    parser.add_argument('--model-type', type=str, default='vae',
                       choices=['vae', 'beta-vae', 'annealed-beta-vae'],
                       help='Type of model to train (default: vae)')

    # Hiperparâmetros do modelo
    parser.add_argument('--latent-dim', type=int, default=10,
                       help='Latent space dimension (default: 10)')
    parser.add_argument('--hidden-dims', type=int, nargs='+',
                       default=[512, 256],
                       help='Hidden layer dimensions (default: 512 256)')
    parser.add_argument('--beta', type=float, default=1.0,
                       help='Beta value for Beta-VAE (default: 1.0)')

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
    parser.add_argument('--generate-samples', type=int, default=16,
                       help='Number of samples to generate (default: 16)')

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
    print(f"TRAINING {args.model_type.upper()}")
    print("=" * 70)

    print("\nConfiguration:")
    print(f"  Model Type: {args.model_type}")
    print(f"  Latent Dimension: {args.latent_dim}")
    print(f"  Hidden Dimensions: {args.hidden_dims}")
    print(f"  Beta: {args.beta}")
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
    print(f"\n[2/5] Creating {args.model_type}...")

    model_kwargs = {
        'input_dim': 784,
        'latent_dim': args.latent_dim,
        'hidden_dims': args.hidden_dims
    }

    if args.model_type == 'vae':
        model = VAE(**model_kwargs)
    elif args.model_type == 'beta-vae':
        model = BetaVAE(**model_kwargs, beta=args.beta)
    elif args.model_type == 'annealed-beta-vae':
        model = AnnealedBetaVAE(**model_kwargs, max_beta=args.beta)

    print(f"  {model}")

    # Conta parâmetros
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_params:,}")

    # Treina
    print("\n[3/5] Training model...")
    save_path = os.path.join(args.output_dir, f'best_{args.model_type}.pt') if args.save_model else None

    history = train_vae(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        beta=args.beta,
        device=device,
        early_stopping_patience=args.early_stopping,
        save_path=save_path,
        verbose=True
    )

    print("\n  Training completed!")
    print(f"  Final train loss: {history['train_loss'][-1]:.6f}")
    print(f"  Final train recon: {history['train_recon'][-1]:.6f}")
    print(f"  Final train KL: {history['train_kl'][-1]:.6f}")

    if history['val_loss']:
        print(f"  Final val loss: {history['val_loss'][-1]:.6f}")
        print(f"  Best val loss: {min(history['val_loss']):.6f}")

    # Avalia no teste
    print("\n[4/5] Evaluating on test set...")
    test_metrics = evaluate_model(model, test_loader, device=device,
                                  is_vae=True, beta=args.beta)
    print(f"  Test loss: {test_metrics['loss']:.6f}")
    print(f"  Test reconstruction: {test_metrics['reconstruction']:.6f}")
    print(f"  Test KL divergence: {test_metrics['kl']:.6f}")

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

        # Visão geral do VAE
        print("  - Plotting VAE results overview...")
        plot_vae_results(
            model, test_loader, device=device,
            save_path=os.path.join(args.output_dir, 'vae_overview.png'),
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

            print("  - Plotting latent grid manifold...")
            plot_latent_grid(
                model, n_samples=20, device=device,
                save_path=os.path.join(args.output_dir, 'latent_grid.png'),
                show=False
            )

        # Amostras geradas
        print(f"  - Generating {args.generate_samples} samples...")
        model.eval()
        with torch.no_grad():
            samples = model.sample(num_samples=args.generate_samples, device=device)
            samples = samples.view(-1, 28, 28).cpu().numpy()

            import matplotlib.pyplot as plt
            import numpy as np

            n_cols = 8
            n_rows = (args.generate_samples + n_cols - 1) // n_cols

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, n_rows * 1.5))
            axes = axes.flatten() if args.generate_samples > 1 else [axes]

            for i in range(args.generate_samples):
                axes[i].imshow(samples[i], cmap='gray')
                axes[i].axis('off')

            # Remove axes vazios
            for i in range(args.generate_samples, len(axes)):
                axes[i].axis('off')

            plt.suptitle('Generated Samples', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(args.output_dir, 'generated_samples.png'),
                       dpi=300, bbox_inches='tight')
            plt.close()

        print(f"\n  Visualizations saved to {args.output_dir}/")
    else:
        print("\n[5/5] Skipping visualizations (--no-visualize)")

    # Salva modelo final
    if args.save_model:
        final_path = os.path.join(args.output_dir, f'final_{args.model_type}.pt')
        torch.save(model.state_dict(), final_path)
        print(f"\n  Model saved to {final_path}")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETED!")
    print("=" * 70)

    print("\nResults:")
    print(f"  Test Loss: {test_metrics['loss']:.6f}")
    print(f"  Reconstruction: {test_metrics['reconstruction']:.6f}")
    print(f"  KL Divergence: {test_metrics['kl']:.6f}")

    if not args.no_visualize:
        print(f"\nGenerated files in {args.output_dir}/:")
        print("  - training_history.png")
        print("  - vae_overview.png")
        if args.latent_dim == 2:
            print("  - latent_space.png")
            print("  - latent_grid.png")
        print("  - generated_samples.png")
        if args.save_model:
            print(f"  - best_{args.model_type}.pt")
            print(f"  - final_{args.model_type}.pt")

    print("\nNext steps:")
    print("  1. Check the generated visualizations")
    print("  2. Try different beta values: --beta 4.0")
    print("  3. Use Beta-VAE: --model-type beta-vae")
    print("  4. Explore latent space: python examples/explore_latent_space.py")
    print("  5. Compare different betas in notebooks/04_beta_vae_experimento.ipynb")

    print("=" * 70)


if __name__ == "__main__":
    main()
