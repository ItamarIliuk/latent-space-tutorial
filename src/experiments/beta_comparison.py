"""
Beta Comparison: Experimentos comparando diferentes valores de Beta no Beta-VAE
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional
from tqdm import tqdm
import pandas as pd


class BetaVAEComparison:
    """
    Classe para comparar diferentes valores de Beta no Beta-VAE.

    O parâmetro Beta controla o trade-off entre qualidade de reconstrução
    e disentanglement (separação) das dimensões latentes.

    Args:
        model_class: Classe do modelo (VAE ou BetaVAE)
        model_kwargs (dict): Argumentos para inicializar o modelo
        device (str): 'cpu' ou 'cuda'

    Exemplo:
        >>> from src.models.vae import VAE
        >>> from src.utils.data_loader import load_mnist
        >>>
        >>> train_loader, val_loader, test_loader = load_mnist(batch_size=128)
        >>> comparison = BetaVAEComparison(VAE, {'latent_dim': 10})
        >>> results = comparison.compare_betas(
        ...     betas=[0.5, 1.0, 2.0, 4.0],
        ...     train_loader=train_loader,
        ...     val_loader=val_loader,
        ...     num_epochs=20
        ... )
    """

    def __init__(self, model_class, model_kwargs=None, device='cpu'):
        self.model_class = model_class
        self.model_kwargs = model_kwargs or {}
        self.device = device
        self.results = {}

    def train_single_beta(self, beta, train_loader, val_loader=None,
                         num_epochs=20, learning_rate=1e-3, verbose=True):
        """
        Treina um modelo com um valor específico de beta.

        Args:
            beta (float): Valor de beta a usar
            train_loader: DataLoader de treino
            val_loader: DataLoader de validação
            num_epochs (int): Número de épocas
            learning_rate (float): Taxa de aprendizado
            verbose (bool): Se deve mostrar progresso

        Returns:
            dict: Dicionário com modelo treinado e histórico
        """
        from src.utils.training import train_vae

        # Cria novo modelo
        model = self.model_class(**self.model_kwargs)

        if verbose:
            print(f"\n{'='*60}")
            print(f"Training with Beta = {beta}")
            print(f"{'='*60}")

        # Treina
        history = train_vae(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            beta=beta,
            device=self.device,
            verbose=verbose
        )

        return {
            'model': model,
            'history': history,
            'beta': beta
        }

    def compare_betas(self, betas, train_loader, val_loader=None,
                     num_epochs=20, learning_rate=1e-3, verbose=True):
        """
        Compara múltiplos valores de beta.

        Args:
            betas (list): Lista de valores de beta para testar
            train_loader: DataLoader de treino
            val_loader: DataLoader de validação
            num_epochs (int): Número de épocas por modelo
            learning_rate (float): Taxa de aprendizado
            verbose (bool): Se deve mostrar progresso

        Returns:
            dict: Resultados para cada beta

        Exemplo:
            >>> results = comparison.compare_betas(
            ...     betas=[0.5, 1.0, 2.0, 4.0, 8.0],
            ...     train_loader=train_loader,
            ...     val_loader=val_loader,
            ...     num_epochs=25
            ... )
        """
        self.results = {}

        for beta in betas:
            result = self.train_single_beta(
                beta=beta,
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=num_epochs,
                learning_rate=learning_rate,
                verbose=verbose
            )
            self.results[beta] = result

        if verbose:
            print(f"\n{'='*60}")
            print("Training completed for all beta values!")
            print(f"{'='*60}\n")

        return self.results

    def plot_comparison(self, figsize=(16, 10), save_path=None, show=True):
        """
        Plota comparação visual entre diferentes betas.

        Args:
            figsize (tuple): Tamanho da figura
            save_path (str): Caminho para salvar
            show (bool): Se deve mostrar o gráfico

        Exemplo:
            >>> comparison.plot_comparison(figsize=(18, 12))
        """
        if not self.results:
            raise ValueError("No results to plot. Run compare_betas() first.")

        n_betas = len(self.results)
        betas = sorted(self.results.keys())

        fig = plt.figure(figsize=figsize)

        # 1. Loss curves
        ax1 = plt.subplot(3, 3, 1)
        for beta in betas:
            history = self.results[beta]['history']
            ax1.plot(history['train_loss'], label=f'β={beta}', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Total Loss')
        ax1.set_title('Training Loss (Total)', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Reconstruction loss
        ax2 = plt.subplot(3, 3, 2)
        for beta in betas:
            history = self.results[beta]['history']
            ax2.plot(history['train_recon'], label=f'β={beta}', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Reconstruction Loss')
        ax2.set_title('Reconstruction Loss', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. KL divergence
        ax3 = plt.subplot(3, 3, 3)
        for beta in betas:
            history = self.results[beta]['history']
            ax3.plot(history['train_kl'], label=f'β={beta}', linewidth=2)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('KL Divergence')
        ax3.set_title('KL Divergence', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4-6. Final metrics comparison
        ax4 = plt.subplot(3, 3, 4)
        final_losses = [self.results[b]['history']['train_loss'][-1] for b in betas]
        ax4.bar(range(len(betas)), final_losses, color='steelblue')
        ax4.set_xticks(range(len(betas)))
        ax4.set_xticklabels([f'β={b}' for b in betas])
        ax4.set_ylabel('Final Loss')
        ax4.set_title('Final Total Loss', fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')

        ax5 = plt.subplot(3, 3, 5)
        final_recon = [self.results[b]['history']['train_recon'][-1] for b in betas]
        ax5.bar(range(len(betas)), final_recon, color='coral')
        ax5.set_xticks(range(len(betas)))
        ax5.set_xticklabels([f'β={b}' for b in betas])
        ax5.set_ylabel('Final Reconstruction Loss')
        ax5.set_title('Final Reconstruction Loss', fontweight='bold')
        ax5.grid(True, alpha=0.3, axis='y')

        ax6 = plt.subplot(3, 3, 6)
        final_kl = [self.results[b]['history']['train_kl'][-1] for b in betas]
        ax6.bar(range(len(betas)), final_kl, color='mediumseagreen')
        ax6.set_xticks(range(len(betas)))
        ax6.set_xticklabels([f'β={b}' for b in betas])
        ax6.set_ylabel('Final KL Divergence')
        ax6.set_title('Final KL Divergence', fontweight='bold')
        ax6.grid(True, alpha=0.3, axis='y')

        # 7-9. Sample generations for each beta
        for idx, beta in enumerate(betas[:3]):  # Primeiros 3 betas
            ax = plt.subplot(3, 3, 7 + idx)
            model = self.results[beta]['model'].to(self.device)
            model.eval()

            with torch.no_grad():
                samples = model.sample(num_samples=16, device=self.device)
                samples = samples.view(-1, 28, 28).cpu().numpy()

            # Grid 4x4
            sample_grid = np.zeros((4 * 28, 4 * 28))
            for i in range(4):
                for j in range(4):
                    sample_idx = i * 4 + j
                    sample_grid[i*28:(i+1)*28, j*28:(j+1)*28] = samples[sample_idx]

            ax.imshow(sample_grid, cmap='gray')
            ax.axis('off')
            ax.set_title(f'Samples (β={beta})', fontweight='bold')

        plt.suptitle('Beta-VAE Comparison', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_latent_spaces(self, dataloader, figsize=(15, 4),
                          save_path=None, show=True):
        """
        Plota o espaço latente para cada beta lado a lado.

        Args:
            dataloader: DataLoader para extrair códigos latentes
            figsize (tuple): Tamanho da figura
            save_path (str): Caminho para salvar
            show (bool): Se deve mostrar

        Exemplo:
            >>> comparison.plot_latent_spaces(test_loader)
        """
        if not self.results:
            raise ValueError("No results to plot. Run compare_betas() first.")

        betas = sorted(self.results.keys())
        n_betas = len(betas)

        fig, axes = plt.subplots(1, n_betas, figsize=figsize)

        if n_betas == 1:
            axes = [axes]

        for idx, beta in enumerate(betas):
            model = self.results[beta]['model'].to(self.device)
            model.eval()

            latent_codes = []
            labels = []

            with torch.no_grad():
                for data, label in dataloader:
                    data = data.view(-1, model.input_dim).to(self.device)
                    mu, _ = model.encode(data)
                    latent_codes.append(mu.cpu().numpy())
                    labels.append(label.numpy())

            latent_codes = np.concatenate(latent_codes, axis=0)[:1000]
            labels = np.concatenate(labels, axis=0)[:1000]

            # Plota primeiras 2 dimensões
            if latent_codes.shape[1] >= 2:
                scatter = axes[idx].scatter(latent_codes[:, 0], latent_codes[:, 1],
                                           c=labels, cmap='tab10', alpha=0.5, s=5)
                axes[idx].set_xlabel('z[0]')
                axes[idx].set_ylabel('z[1]')
                axes[idx].set_title(f'β = {beta}', fontweight='bold')
                axes[idx].grid(True, alpha=0.3)

        plt.suptitle('Latent Space Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_reconstructions(self, dataloader, n_samples=8,
                            figsize=(15, 10), save_path=None, show=True):
        """
        Plota reconstruções de cada beta para comparação visual.

        Args:
            dataloader: DataLoader com dados
            n_samples (int): Número de amostras
            figsize (tuple): Tamanho da figura
            save_path (str): Caminho para salvar
            show (bool): Se deve mostrar

        Exemplo:
            >>> comparison.plot_reconstructions(test_loader, n_samples=10)
        """
        if not self.results:
            raise ValueError("No results to plot. Run compare_betas() first.")

        betas = sorted(self.results.keys())
        n_betas = len(betas)

        # Pega um batch
        data, labels = next(iter(dataloader))
        data = data[:n_samples].to(self.device)

        fig, axes = plt.subplots(n_betas + 1, n_samples, figsize=figsize)

        # Primeira linha: originais
        for i in range(n_samples):
            axes[0, i].imshow(data[i].cpu().squeeze(), cmap='gray')
            axes[0, i].axis('off')
            if i == 0:
                axes[0, i].set_title('Original', fontsize=11, fontweight='bold',
                                    loc='left')

        # Linhas seguintes: reconstruções para cada beta
        for row_idx, beta in enumerate(betas):
            model = self.results[beta]['model'].to(self.device)
            model.eval()

            with torch.no_grad():
                data_flat = data.view(-1, model.input_dim)
                reconstructed, _, _, _ = model(data_flat)
                reconstructed = reconstructed.view(-1, 1, 28, 28)

            for i in range(n_samples):
                axes[row_idx + 1, i].imshow(reconstructed[i].cpu().squeeze(), cmap='gray')
                axes[row_idx + 1, i].axis('off')
                if i == 0:
                    axes[row_idx + 1, i].set_title(f'β={beta}', fontsize=11,
                                                   fontweight='bold', loc='left')

        plt.suptitle('Reconstruction Quality Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def get_summary_table(self):
        """
        Retorna uma tabela resumida com métricas finais de cada beta.

        Returns:
            pd.DataFrame: Tabela com métricas

        Exemplo:
            >>> summary = comparison.get_summary_table()
            >>> print(summary)
        """
        if not self.results:
            raise ValueError("No results. Run compare_betas() first.")

        data = []
        for beta in sorted(self.results.keys()):
            history = self.results[beta]['history']

            row = {
                'Beta': beta,
                'Final Loss': history['train_loss'][-1],
                'Final Recon': history['train_recon'][-1],
                'Final KL': history['train_kl'][-1],
                'Min Loss': min(history['train_loss']),
                'Min Recon': min(history['train_recon']),
            }

            if history['val_loss']:
                row['Val Loss'] = history['val_loss'][-1]

            data.append(row)

        df = pd.DataFrame(data)
        return df

    def export_results(self, filepath):
        """
        Exporta resultados para arquivo.

        Args:
            filepath (str): Caminho do arquivo (CSV ou pickle)

        Exemplo:
            >>> comparison.export_results('beta_comparison_results.csv')
        """
        df = self.get_summary_table()

        if filepath.endswith('.csv'):
            df.to_csv(filepath, index=False)
        elif filepath.endswith('.pkl'):
            df.to_pickle(filepath)
        else:
            raise ValueError("Filepath must end with .csv or .pkl")

        print(f"Results exported to {filepath}")

    def __repr__(self):
        n_results = len(self.results)
        betas = sorted(self.results.keys()) if self.results else []
        return (f"BetaVAEComparison(n_experiments={n_results}, "
                f"betas={betas}, device={self.device})")
