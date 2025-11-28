"""
Visualization: Funções para plotar resultados de Autoencoders e VAEs
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from typing import Optional, Tuple


def visualize_latent_space(model, dataloader, device='cpu', labels=None,
                           figsize=(10, 8), title='Latent Space Visualization',
                           save_path=None, show=True):
    """
    Visualiza o espaço latente 2D colorindo por classe.

    Args:
        model: Modelo Autoencoder ou VAE
        dataloader: DataLoader com os dados
        device (str): 'cpu' ou 'cuda'
        labels (list): Rótulos customizados (opcional)
        figsize (tuple): Tamanho da figura
        title (str): Título do gráfico
        save_path (str): Caminho para salvar a figura
        show (bool): Se deve mostrar o gráfico

    Returns:
        tuple: (latent_codes, labels) - arrays numpy

    Exemplo:
        >>> from src.models.autoencoder import Autoencoder
        >>> model = Autoencoder(latent_dim=2)
        >>> visualize_latent_space(model, test_loader, device='cuda')
    """
    model = model.to(device)
    model.eval()

    latent_codes = []
    all_labels = []

    with torch.no_grad():
        for data, label in dataloader:
            data = data.view(-1, model.input_dim).to(device)

            # Para VAE, pega a média (mu)
            if hasattr(model, 'encode') and len(model.encode(data)) == 2:
                # É um VAE
                mu, _ = model.encode(data)
                latent = mu
            else:
                # É um Autoencoder
                _, latent = model(data)

            latent_codes.append(latent.cpu().numpy())
            all_labels.append(label.numpy())

    latent_codes = np.concatenate(latent_codes, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Verifica se é 2D
    if latent_codes.shape[1] != 2:
        print(f"Warning: Latent space has {latent_codes.shape[1]} dimensions, "
              f"only showing first 2 dimensions")
        latent_codes = latent_codes[:, :2]

    # Plotagem
    fig, ax = plt.subplots(figsize=figsize)

    scatter = ax.scatter(latent_codes[:, 0], latent_codes[:, 1],
                        c=all_labels, cmap='tab10', alpha=0.6, s=5)

    ax.set_xlabel('Latent Dimension 1', fontsize=12)
    ax.set_ylabel('Latent Dimension 2', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Colorbar com labels
    cbar = plt.colorbar(scatter, ax=ax, ticks=range(10))
    cbar.set_label('Digit', fontsize=12)

    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return latent_codes, all_labels


def plot_reconstructions(model, dataloader, n_samples=10, device='cpu',
                         figsize=(20, 4), save_path=None, show=True):
    """
    Plota imagens originais e suas reconstruções lado a lado.

    Args:
        model: Modelo Autoencoder ou VAE
        dataloader: DataLoader com os dados
        n_samples (int): Número de amostras a plotar
        device (str): 'cpu' ou 'cuda'
        figsize (tuple): Tamanho da figura
        save_path (str): Caminho para salvar a figura
        show (bool): Se deve mostrar o gráfico

    Exemplo:
        >>> plot_reconstructions(model, test_loader, n_samples=10, device='cuda')
    """
    model = model.to(device)
    model.eval()

    # Pega um batch
    data, labels = next(iter(dataloader))
    data = data[:n_samples].to(device)
    labels = labels[:n_samples]

    # Reconstrução
    with torch.no_grad():
        data_flat = data.view(-1, model.input_dim)

        if hasattr(model, 'encode') and len(model.encode(data_flat)) == 2:
            # É um VAE
            reconstructed, _, _, _ = model(data_flat)
        else:
            # É um Autoencoder
            reconstructed, _ = model(data_flat)

        reconstructed = reconstructed.view(-1, 1, 28, 28)

    # Plotagem
    fig, axes = plt.subplots(2, n_samples, figsize=figsize)

    for i in range(n_samples):
        # Original
        axes[0, i].imshow(data[i].cpu().squeeze(), cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Original', fontsize=12, fontweight='bold')
        axes[0, i].text(0.5, -0.1, f'{labels[i]}', transform=axes[0, i].transAxes,
                       ha='center', fontsize=10)

        # Reconstrução
        axes[1, i].imshow(reconstructed[i].cpu().squeeze(), cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Reconstructed', fontsize=12, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_vae_results(model, dataloader, device='cpu', figsize=(16, 10),
                    save_path=None, show=True):
    """
    Plota uma visualização completa dos resultados do VAE:
    - Espaço latente
    - Reconstruções
    - Amostras geradas
    - Interpolações

    Args:
        model: Modelo VAE
        dataloader: DataLoader com os dados
        device (str): 'cpu' ou 'cuda'
        figsize (tuple): Tamanho da figura
        save_path (str): Caminho para salvar a figura
        show (bool): Se deve mostrar o gráfico

    Exemplo:
        >>> from src.models.vae import VAE
        >>> vae = VAE(latent_dim=2)
        >>> plot_vae_results(vae, test_loader, device='cuda')
    """
    model = model.to(device)
    model.eval()

    # Cria grid de subplots
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    # 1. Espaço latente
    ax1 = fig.add_subplot(gs[0, 0])
    latent_codes = []
    labels = []

    with torch.no_grad():
        for data, label in dataloader:
            data = data.view(-1, model.input_dim).to(device)
            mu, _ = model.encode(data)
            latent_codes.append(mu.cpu().numpy())
            labels.append(label.numpy())

    latent_codes = np.concatenate(latent_codes, axis=0)[:1000]  # Limita para performance
    labels = np.concatenate(labels, axis=0)[:1000]

    if latent_codes.shape[1] >= 2:
        scatter = ax1.scatter(latent_codes[:, 0], latent_codes[:, 1],
                             c=labels, cmap='tab10', alpha=0.5, s=5)
        ax1.set_xlabel('z[0]')
        ax1.set_ylabel('z[1]')
        ax1.set_title('Latent Space', fontweight='bold')
        plt.colorbar(scatter, ax=ax1)

    # 2. Reconstruções
    ax2 = fig.add_subplot(gs[0, 1])
    data, _ = next(iter(dataloader))
    data = data[:8].to(device)

    with torch.no_grad():
        data_flat = data.view(-1, model.input_dim)
        reconstructed, _, _, _ = model(data_flat)
        reconstructed = reconstructed.view(-1, 1, 28, 28)

    # Concatena original e reconstrução
    comparison = torch.cat([data, reconstructed], dim=0)
    grid = comparison.cpu().numpy()

    # Cria grid 2x8
    img_grid = np.zeros((2 * 28, 8 * 28))
    for i in range(8):
        img_grid[0:28, i*28:(i+1)*28] = grid[i, 0]
        img_grid[28:56, i*28:(i+1)*28] = grid[i+8, 0]

    ax2.imshow(img_grid, cmap='gray')
    ax2.axis('off')
    ax2.set_title('Reconstructions (top: original, bottom: recon)', fontweight='bold')

    # 3. Amostras geradas
    ax3 = fig.add_subplot(gs[1, 0])

    with torch.no_grad():
        samples = model.sample(num_samples=16, device=device)
        samples = samples.view(-1, 1, 28, 28).cpu().numpy()

    # Grid 4x4
    sample_grid = np.zeros((4 * 28, 4 * 28))
    for i in range(4):
        for j in range(4):
            idx = i * 4 + j
            sample_grid[i*28:(i+1)*28, j*28:(j+1)*28] = samples[idx, 0]

    ax3.imshow(sample_grid, cmap='gray')
    ax3.axis('off')
    ax3.set_title('Generated Samples', fontweight='bold')

    # 4. Interpolação
    ax4 = fig.add_subplot(gs[1, 1])

    data, _ = next(iter(dataloader))
    x1, x2 = data[0].to(device), data[1].to(device)

    with torch.no_grad():
        interpolated = model.interpolate(x1.view(-1, 784), x2.view(-1, 784), num_steps=10)
        interpolated = interpolated.view(-1, 28, 28).cpu().numpy()

    # Grid 1x10
    interp_grid = np.concatenate(interpolated, axis=1)

    ax4.imshow(interp_grid, cmap='gray')
    ax4.axis('off')
    ax4.set_title('Latent Space Interpolation', fontweight='bold')

    plt.suptitle('VAE Results Overview', fontsize=16, fontweight='bold', y=0.98)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_latent_grid(model, n_samples=20, latent_range=(-3, 3),
                     device='cpu', figsize=(15, 15),
                     save_path=None, show=True):
    """
    Plota um grid 2D explorando o espaço latente (apenas para latent_dim=2).

    Args:
        model: Modelo VAE com latent_dim=2
        n_samples (int): Número de pontos em cada dimensão
        latent_range (tuple): Range do espaço latente a explorar
        device (str): 'cpu' ou 'cuda'
        figsize (tuple): Tamanho da figura
        save_path (str): Caminho para salvar a figura
        show (bool): Se deve mostrar o gráfico

    Exemplo:
        >>> plot_latent_grid(vae, n_samples=20, device='cuda')
    """
    if model.latent_dim != 2:
        raise ValueError("plot_latent_grid only works with latent_dim=2")

    model = model.to(device)
    model.eval()

    # Cria grid no espaço latente
    x = np.linspace(latent_range[0], latent_range[1], n_samples)
    y = np.linspace(latent_range[0], latent_range[1], n_samples)

    fig, ax = plt.subplots(figsize=figsize)

    # Grid para a imagem final
    img_grid = np.zeros((n_samples * 28, n_samples * 28))

    with torch.no_grad():
        for i, yi in enumerate(reversed(y)):
            for j, xi in enumerate(x):
                # Ponto no espaço latente
                z = torch.tensor([[xi, yi]], dtype=torch.float32).to(device)

                # Decodifica
                sample = model.decode(z)
                sample = sample.view(28, 28).cpu().numpy()

                # Coloca no grid
                img_grid[i*28:(i+1)*28, j*28:(j+1)*28] = sample

    ax.imshow(img_grid, cmap='gray', extent=[latent_range[0], latent_range[1],
                                             latent_range[0], latent_range[1]])
    ax.set_xlabel('Latent Dimension 1', fontsize=12)
    ax.set_ylabel('Latent Dimension 2', fontsize=12)
    ax.set_title('Latent Space Manifold', fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_interpolation(model, x1, x2, num_steps=10, device='cpu',
                       figsize=(15, 3), save_path=None, show=True):
    """
    Plota interpolação entre duas imagens no espaço latente.

    Args:
        model: Modelo VAE ou Autoencoder
        x1, x2 (torch.Tensor): Duas imagens de entrada
        num_steps (int): Número de passos na interpolação
        device (str): 'cpu' ou 'cuda'
        figsize (tuple): Tamanho da figura
        save_path (str): Caminho para salvar a figura
        show (bool): Se deve mostrar o gráfico

    Exemplo:
        >>> data, _ = next(iter(test_loader))
        >>> plot_interpolation(vae, data[0], data[1], num_steps=10)
    """
    model = model.to(device)
    model.eval()

    x1 = x1.to(device)
    x2 = x2.to(device)

    with torch.no_grad():
        if hasattr(model, 'interpolate'):
            # VAE tem método interpolate
            interpolated = model.interpolate(x1.view(-1, 784), x2.view(-1, 784),
                                            num_steps=num_steps)
        else:
            # Autoencoder - interpolação manual
            _, z1 = model(x1.view(-1, 784))
            _, z2 = model(x2.view(-1, 784))

            alphas = torch.linspace(0, 1, num_steps).to(device)
            interpolated = []

            for alpha in alphas:
                z = (1 - alpha) * z1 + alpha * z2
                img = model.decode(z)
                interpolated.append(img)

            interpolated = torch.cat(interpolated, dim=0)

        interpolated = interpolated.view(-1, 28, 28).cpu().numpy()

    # Plotagem
    fig, axes = plt.subplots(1, num_steps, figsize=figsize)

    for i in range(num_steps):
        axes[i].imshow(interpolated[i], cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(f'{i/(num_steps-1):.1f}', fontsize=9)

    plt.suptitle('Latent Space Interpolation', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_training_history(history, figsize=(12, 5), save_path=None, show=True):
    """
    Plota histórico de treinamento (loss curves).

    Args:
        history (dict): Dicionário retornado por train_model() ou train_vae()
        figsize (tuple): Tamanho da figura
        save_path (str): Caminho para salvar a figura
        show (bool): Se deve mostrar o gráfico

    Exemplo:
        >>> history = train_vae(vae, train_loader, val_loader, num_epochs=30)
        >>> plot_training_history(history)
    """
    is_vae = 'train_recon' in history

    if is_vae:
        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # Total Loss
        axes[0].plot(history['train_loss'], label='Train', linewidth=2)
        if history['val_loss']:
            axes[0].plot(history['val_loss'], label='Validation', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Total Loss')
        axes[0].set_title('Total Loss', fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Reconstruction Loss
        axes[1].plot(history['train_recon'], label='Train', linewidth=2)
        if history['val_recon']:
            axes[1].plot(history['val_recon'], label='Validation', linewidth=2)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Reconstruction Loss')
        axes[1].set_title('Reconstruction Loss', fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # KL Divergence
        axes[2].plot(history['train_kl'], label='Train', linewidth=2)
        if history['val_kl']:
            axes[2].plot(history['val_kl'], label='Validation', linewidth=2)
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('KL Divergence')
        axes[2].set_title('KL Divergence', fontweight='bold')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
    else:
        fig, ax = plt.subplots(figsize=(8, 5))

        ax.plot(history['train_loss'], label='Train', linewidth=2)
        if history['val_loss']:
            ax.plot(history['val_loss'], label='Validation', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Training History', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()
