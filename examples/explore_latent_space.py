"""
Explore Latent Space: Script para exploração interativa do espaço latente

Este script permite explorar o espaço latente de modelos treinados.
"""

import torch
import argparse
import os
import numpy as np
from src.models.vae import VAE
from src.models.autoencoder import Autoencoder
from src.utils.data_loader import load_mnist
from src.experiments.latent_explorer import LatentExplorer


def parse_args():
    """Parse argumentos de linha de comando."""
    parser = argparse.ArgumentParser(
        description='Explore the latent space of a trained model'
    )

    # Modelo
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--model-type', type=str, default='vae',
                       choices=['vae', 'autoencoder'],
                       help='Type of model (default: vae)')
    parser.add_argument('--latent-dim', type=int, default=10,
                       help='Latent space dimension (default: 10)')
    parser.add_argument('--hidden-dims', type=int, nargs='+',
                       default=[512, 256],
                       help='Hidden layer dimensions (default: 512 256)')

    # Exploração
    parser.add_argument('--mode', type=str, default='grid-walk',
                       choices=['grid-walk', 'interpolate', 'random-samples', 'nearest-neighbors'],
                       help='Exploration mode (default: grid-walk)')
    parser.add_argument('--n-samples', type=int, default=10,
                       help='Number of samples (default: 10)')
    parser.add_argument('--dim1', type=int, default=0,
                       help='First dimension to explore (default: 0)')
    parser.add_argument('--dim2', type=int, default=1,
                       help='Second dimension to explore (default: 1)')

    # Dados e dispositivo
    parser.add_argument('--data-dir', type=str, default='./data',
                       help='Directory for MNIST data (default: ./data)')
    parser.add_argument('--output-dir', type=str, default='./output',
                       help='Directory for outputs (default: ./output)')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use (default: auto)')

    return parser.parse_args()


def load_model(args, device):
    """Carrega o modelo treinado."""

    model_kwargs = {
        'input_dim': 784,
        'latent_dim': args.latent_dim,
        'hidden_dims': args.hidden_dims
    }

    if args.model_type == 'vae':
        model = VAE(**model_kwargs)
    else:
        model = Autoencoder(**model_kwargs)

    # Carrega pesos
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    model.eval()

    print(f"  Model loaded from {args.model_path}")
    print(f"  {model}")

    return model


def main():
    """Função principal."""
    args = parse_args()

    # Configura device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    print("=" * 70)
    print("LATENT SPACE EXPLORATION")
    print("=" * 70)

    print("\nConfiguration:")
    print(f"  Model Type: {args.model_type}")
    print(f"  Model Path: {args.model_path}")
    print(f"  Latent Dimension: {args.latent_dim}")
    print(f"  Exploration Mode: {args.mode}")
    print(f"  Device: {device}")

    # Verifica se arquivo existe
    if not os.path.exists(args.model_path):
        print(f"\nERROR: Model file not found: {args.model_path}")
        print("\nTrain a model first using:")
        print("  python examples/train_vae.py --save-model")
        print("  python examples/train_autoencoder.py --save-model")
        return

    # Cria diretório de output
    os.makedirs(args.output_dir, exist_ok=True)

    # Carrega modelo
    print("\n[1/3] Loading model...")
    model = load_model(args, device)

    # Cria explorer
    print("\n[2/3] Creating latent explorer...")
    explorer = LatentExplorer(
        model=model,
        latent_dim=args.latent_dim,
        device=device,
        latent_range=(-3, 3)
    )
    print(f"  {explorer}")

    # Carrega dados (se necessário)
    test_loader = None
    if args.mode in ['nearest-neighbors']:
        print("\n  Loading MNIST dataset for nearest neighbors...")
        _, _, test_loader = load_mnist(
            batch_size=128,
            data_dir=args.data_dir,
            num_workers=0
        )

    # Exploração
    print(f"\n[3/3] Exploring latent space ({args.mode})...")

    if args.mode == 'grid-walk':
        if args.latent_dim < 2:
            print("  ERROR: grid-walk requires latent_dim >= 2")
            return

        print(f"  Walking through dimensions {args.dim1} and {args.dim2}")
        explorer.generate_grid_walk(
            dim1=args.dim1,
            dim2=args.dim2,
            n_steps=args.n_samples,
            range_=(-3, 3),
            save_path=os.path.join(args.output_dir, f'grid_walk_dim{args.dim1}_dim{args.dim2}.png')
        )

    elif args.mode == 'interpolate':
        print(f"  Generating {args.n_samples} random interpolations...")
        explorer.interpolate_between_samples(
            n_samples=args.n_samples,
            steps=10
        )

        import matplotlib.pyplot as plt
        plt.savefig(os.path.join(args.output_dir, 'interpolations.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved to {args.output_dir}/interpolations.png")

    elif args.mode == 'random-samples':
        print(f"  Generating {args.n_samples} random samples...")

        import matplotlib.pyplot as plt

        # Gera amostras
        samples = []
        for _ in range(args.n_samples):
            z = explorer.sample_random()
            img = explorer.decode_latent(z)
            samples.append(img)

        # Plota
        n_cols = min(10, args.n_samples)
        n_rows = (args.n_samples + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.5, n_rows * 1.5))

        if args.n_samples == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for i in range(args.n_samples):
            axes[i].imshow(samples[i], cmap='gray')
            axes[i].axis('off')

        for i in range(args.n_samples, len(axes)):
            axes[i].axis('off')

        plt.suptitle('Random Samples from Latent Space', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 'random_samples.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  Saved to {args.output_dir}/random_samples.png")

    elif args.mode == 'nearest-neighbors':
        print(f"  Finding nearest neighbors for {args.n_samples} random points...")

        if test_loader is None:
            print("  ERROR: Test loader not available")
            return

        for i in range(args.n_samples):
            z = explorer.sample_random()
            print(f"\n  Sample {i+1}/{args.n_samples}:")

            imgs, labels, dists = explorer.find_nearest_sample(
                z, test_loader, n_nearest=5
            )

            import matplotlib.pyplot as plt
            plt.savefig(os.path.join(args.output_dir, f'nearest_neighbors_{i+1}.png'),
                       dpi=300, bbox_inches='tight')
            plt.close()

        print(f"\n  Saved {args.n_samples} files to {args.output_dir}/")

    print("\n" + "=" * 70)
    print("EXPLORATION COMPLETED!")
    print("=" * 70)

    print(f"\nGenerated files in {args.output_dir}/:")

    if args.mode == 'grid-walk':
        print(f"  - grid_walk_dim{args.dim1}_dim{args.dim2}.png")
    elif args.mode == 'interpolate':
        print("  - interpolations.png")
    elif args.mode == 'random-samples':
        print("  - random_samples.png")
    elif args.mode == 'nearest-neighbors':
        for i in range(args.n_samples):
            print(f"  - nearest_neighbors_{i+1}.png")

    print("\nNext steps:")
    print("  1. Check the generated visualizations")
    print("  2. Try different modes:")
    print("     --mode grid-walk")
    print("     --mode interpolate")
    print("     --mode random-samples")
    print("     --mode nearest-neighbors")
    print("  3. Explore different dimensions: --dim1 2 --dim2 3")
    print("  4. Use interactive exploration in Jupyter notebooks")

    print("=" * 70)


if __name__ == "__main__":
    main()
