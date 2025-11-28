"""
Latent Explorer: Ferramenta para exploração interativa do espaço latente
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from typing import Optional, Callable
import ipywidgets as widgets
from IPython.display import display


class LatentExplorer:
    """
    Classe para exploração interativa do espaço latente de VAEs e Autoencoders.

    Permite navegar pelo espaço latente e ver as imagens geradas em tempo real.

    Args:
        model: Modelo VAE ou Autoencoder (deve ter método decode())
        latent_dim (int): Dimensão do espaço latente
        device (str): 'cpu' ou 'cuda'
        latent_range (tuple): Range inicial para os sliders

    Exemplo:
        >>> from src.models.vae import VAE
        >>> vae = VAE(latent_dim=10)
        >>> explorer = LatentExplorer(vae, latent_dim=10)
        >>> explorer.launch_interactive()  # Em Jupyter Notebook
    """

    def __init__(self, model, latent_dim=None, device='cpu', latent_range=(-3, 3)):
        self.model = model.to(device)
        self.model.eval()
        self.device = device

        if latent_dim is None:
            self.latent_dim = model.latent_dim
        else:
            self.latent_dim = latent_dim

        self.latent_range = latent_range
        self.current_z = torch.zeros(1, self.latent_dim).to(device)

    def decode_latent(self, z):
        """
        Decodifica um vetor latente para imagem.

        Args:
            z (torch.Tensor): Vetor latente

        Returns:
            np.ndarray: Imagem gerada (28, 28)
        """
        with torch.no_grad():
            if isinstance(z, np.ndarray):
                z = torch.tensor(z, dtype=torch.float32).to(self.device)

            if z.dim() == 1:
                z = z.unsqueeze(0)

            img = self.model.decode(z)
            img = img.view(28, 28).cpu().numpy()

        return img

    def sample_random(self):
        """
        Gera uma amostra aleatória do espaço latente.

        Returns:
            torch.Tensor: Vetor latente aleatório
        """
        return torch.randn(1, self.latent_dim).to(self.device)

    def launch_interactive(self, backend='ipywidgets'):
        """
        Lança interface interativa para exploração.

        Args:
            backend (str): 'ipywidgets' para Jupyter ou 'matplotlib' para standalone

        Exemplo:
            >>> explorer.launch_interactive(backend='ipywidgets')
        """
        if backend == 'ipywidgets':
            self._launch_ipywidgets()
        elif backend == 'matplotlib':
            self._launch_matplotlib()
        else:
            raise ValueError(f"Backend '{backend}' não suportado. Use 'ipywidgets' ou 'matplotlib'")

    def _launch_ipywidgets(self):
        """Lança interface usando ipywidgets (para Jupyter)."""
        # Sliders para cada dimensão latente
        sliders = []
        for i in range(self.latent_dim):
            slider = widgets.FloatSlider(
                value=0.0,
                min=self.latent_range[0],
                max=self.latent_range[1],
                step=0.1,
                description=f'z[{i}]',
                continuous_update=True
            )
            sliders.append(slider)

        # Botões
        random_button = widgets.Button(description='Random Sample')
        reset_button = widgets.Button(description='Reset to Zero')

        # Output para a imagem
        output = widgets.Output()

        def update_image(*args):
            """Atualiza a imagem quando sliders mudam."""
            z = torch.tensor([[s.value for s in sliders]], dtype=torch.float32).to(self.device)
            img = self.decode_latent(z)

            with output:
                output.clear_output(wait=True)
                fig, ax = plt.subplots(figsize=(5, 5))
                ax.imshow(img, cmap='gray')
                ax.axis('off')
                ax.set_title('Generated Image', fontsize=14, fontweight='bold')
                plt.tight_layout()
                plt.show()

        def on_random_click(b):
            """Gera amostra aleatória."""
            z = self.sample_random().cpu().numpy()[0]
            for i, slider in enumerate(sliders):
                slider.value = float(z[i])

        def on_reset_click(b):
            """Reseta todos os sliders para zero."""
            for slider in sliders:
                slider.value = 0.0

        # Conecta callbacks
        for slider in sliders:
            slider.observe(update_image, names='value')

        random_button.on_click(on_random_click)
        reset_button.on_click(on_reset_click)

        # Layout
        slider_box = widgets.VBox(sliders[:min(10, len(sliders))])  # Limita a 10 sliders
        if self.latent_dim > 10:
            slider_box2 = widgets.VBox(sliders[10:])
            controls = widgets.HBox([slider_box, slider_box2])
        else:
            controls = slider_box

        buttons = widgets.HBox([random_button, reset_button])

        ui = widgets.VBox([buttons, controls, output])

        # Mostra imagem inicial
        update_image()

        display(ui)

    def _launch_matplotlib(self):
        """Lança interface usando matplotlib sliders (standalone)."""
        if self.latent_dim > 10:
            print("Warning: matplotlib interface supports up to 10 dimensions. Showing first 10.")
            n_dims = 10
        else:
            n_dims = self.latent_dim

        # Cria figura
        fig = plt.figure(figsize=(12, 8))

        # Axes para a imagem
        ax_img = plt.subplot2grid((4, 4), (0, 0), colspan=4, rowspan=3)

        # Axes para os sliders
        slider_axes = []
        for i in range(n_dims):
            ax = plt.subplot2grid((4, 4), (3, i % 4))
            slider_axes.append(ax)

        # Cria sliders
        sliders = []
        for i, ax in enumerate(slider_axes):
            slider = Slider(
                ax, f'z[{i}]',
                self.latent_range[0],
                self.latent_range[1],
                valinit=0.0,
                valstep=0.1,
                orientation='vertical'
            )
            sliders.append(slider)

        # Imagem inicial
        z = torch.zeros(1, self.latent_dim).to(self.device)
        img = self.decode_latent(z)
        im = ax_img.imshow(img, cmap='gray')
        ax_img.axis('off')
        ax_img.set_title('Generated Image', fontsize=14, fontweight='bold')

        def update(val):
            """Atualiza quando slider muda."""
            z = torch.zeros(1, self.latent_dim).to(self.device)
            for i, slider in enumerate(sliders):
                z[0, i] = slider.val

            img = self.decode_latent(z)
            im.set_data(img)
            fig.canvas.draw_idle()

        # Conecta sliders
        for slider in sliders:
            slider.on_changed(update)

        plt.tight_layout()
        plt.show()

    def generate_grid_walk(self, dim1=0, dim2=1, n_steps=10, range_=(-3, 3),
                          save_path=None):
        """
        Gera um grid explorando duas dimensões específicas.

        Args:
            dim1, dim2 (int): Índices das dimensões a explorar
            n_steps (int): Número de passos em cada dimensão
            range_ (tuple): Range dos valores
            save_path (str): Caminho para salvar a figura

        Returns:
            np.ndarray: Grid de imagens (n_steps*28, n_steps*28)

        Exemplo:
            >>> grid = explorer.generate_grid_walk(dim1=0, dim2=1, n_steps=15)
        """
        values = np.linspace(range_[0], range_[1], n_steps)
        grid_img = np.zeros((n_steps * 28, n_steps * 28))

        z_base = torch.zeros(1, self.latent_dim).to(self.device)

        for i, val1 in enumerate(reversed(values)):
            for j, val2 in enumerate(values):
                z = z_base.clone()
                z[0, dim1] = val1
                z[0, dim2] = val2

                img = self.decode_latent(z)
                grid_img[i*28:(i+1)*28, j*28:(j+1)*28] = img

        # Plotagem
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(grid_img, cmap='gray', extent=[range_[0], range_[1],
                                                 range_[0], range_[1]])
        ax.set_xlabel(f'Dimension {dim2}', fontsize=12)
        ax.set_ylabel(f'Dimension {dim1}', fontsize=12)
        ax.set_title(f'Latent Walk: Dim {dim1} vs Dim {dim2}',
                    fontsize=14, fontweight='bold')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Grid saved to {save_path}")

        plt.show()

        return grid_img

    def interpolate_between_samples(self, n_samples=5, steps=10):
        """
        Gera interpolações entre amostras aleatórias.

        Args:
            n_samples (int): Número de pares de amostras
            steps (int): Passos de interpolação

        Returns:
            np.ndarray: Grid de interpolações

        Exemplo:
            >>> explorer.interpolate_between_samples(n_samples=3, steps=10)
        """
        fig, axes = plt.subplots(n_samples, steps, figsize=(15, 3*n_samples))

        if n_samples == 1:
            axes = axes.reshape(1, -1)

        for i in range(n_samples):
            # Duas amostras aleatórias
            z1 = self.sample_random()
            z2 = self.sample_random()

            # Interpolação
            alphas = torch.linspace(0, 1, steps)
            for j, alpha in enumerate(alphas):
                z = (1 - alpha) * z1 + alpha * z2
                img = self.decode_latent(z)

                axes[i, j].imshow(img, cmap='gray')
                axes[i, j].axis('off')

                if i == 0:
                    axes[i, j].set_title(f'{alpha:.1f}', fontsize=10)

        plt.suptitle('Random Latent Interpolations', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()

    def find_nearest_sample(self, target_z, dataloader, n_nearest=5):
        """
        Encontra as amostras do dataset mais próximas de um ponto latente.

        Args:
            target_z (torch.Tensor): Ponto alvo no espaço latente
            dataloader: DataLoader com o dataset
            n_nearest (int): Número de vizinhos mais próximos

        Returns:
            tuple: (imagens, labels, distâncias)

        Exemplo:
            >>> z = explorer.sample_random()
            >>> imgs, labels, dists = explorer.find_nearest_sample(z, test_loader)
        """
        self.model.eval()

        all_latents = []
        all_images = []
        all_labels = []

        # Extrai todos os códigos latentes
        with torch.no_grad():
            for data, labels in dataloader:
                data = data.view(-1, self.model.input_dim).to(self.device)

                # Encode
                if hasattr(self.model, 'encode'):
                    mu, _ = self.model.encode(data)
                    latent = mu
                else:
                    _, latent = self.model(data)

                all_latents.append(latent.cpu())
                all_images.append(data.cpu())
                all_labels.append(labels)

        all_latents = torch.cat(all_latents, dim=0)
        all_images = torch.cat(all_images, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        # Calcula distâncias
        target_z = target_z.cpu()
        distances = torch.norm(all_latents - target_z, dim=1)

        # Encontra os k mais próximos
        _, indices = torch.topk(distances, n_nearest, largest=False)

        nearest_images = all_images[indices].view(-1, 28, 28).numpy()
        nearest_labels = all_labels[indices].numpy()
        nearest_distances = distances[indices].numpy()

        # Plotagem
        fig, axes = plt.subplots(1, n_nearest, figsize=(15, 3))

        for i in range(n_nearest):
            axes[i].imshow(nearest_images[i], cmap='gray')
            axes[i].axis('off')
            axes[i].set_title(f'Label: {nearest_labels[i]}\n'
                            f'Dist: {nearest_distances[i]:.2f}', fontsize=10)

        plt.suptitle('Nearest Neighbors in Latent Space', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()

        return nearest_images, nearest_labels, nearest_distances

    def __repr__(self):
        return (f"LatentExplorer(latent_dim={self.latent_dim}, "
                f"device={self.device}, range={self.latent_range})")
