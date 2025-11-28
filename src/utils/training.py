"""
Training: Funções de treinamento para Autoencoder e VAE
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from typing import Optional, Dict, List, Callable


class EarlyStopping:
    """
    Early stopping para interromper o treinamento quando a validação parar de melhorar.

    Args:
        patience (int): Quantas épocas esperar após última melhora
        min_delta (float): Mínima mudança para considerar melhora
        mode (str): 'min' ou 'max' (minimizar ou maximizar métrica)
        verbose (bool): Se deve imprimir mensagens

    Exemplo:
        >>> early_stopping = EarlyStopping(patience=5, mode='min')
        >>> for epoch in range(100):
        ...     val_loss = train_epoch()
        ...     if early_stopping(val_loss):
        ...         print("Early stopping triggered!")
        ...         break
    """

    def __init__(self, patience=7, min_delta=0.0, mode='min', verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose

        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0

        if mode == 'min':
            self.monitor_op = lambda x, y: x < y - min_delta
            self.best_score = np.inf
        else:
            self.monitor_op = lambda x, y: x > y + min_delta
            self.best_score = -np.inf

    def __call__(self, score, epoch=None):
        """
        Verifica se deve fazer early stopping.

        Args:
            score (float): Métrica atual (loss ou accuracy)
            epoch (int): Número da época atual (opcional)

        Returns:
            bool: True se deve parar o treinamento
        """
        if self.monitor_op(score, self.best_score):
            self.best_score = score
            self.counter = 0
            self.best_epoch = epoch if epoch is not None else self.best_epoch
            if self.verbose:
                print(f"Validation improved to {score:.6f}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")

            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"Early stopping triggered! Best epoch: {self.best_epoch}")
                return True

        return False

    def reset(self):
        """Reseta o estado do early stopping."""
        self.counter = 0
        self.early_stop = False
        if self.mode == 'min':
            self.best_score = np.inf
        else:
            self.best_score = -np.inf


def train_model(model, train_loader, val_loader=None,
                num_epochs=50, learning_rate=1e-3, device='cpu',
                early_stopping_patience=None, save_path=None,
                verbose=True):
    """
    Treina um Autoencoder padrão.

    Args:
        model: Modelo Autoencoder
        train_loader: DataLoader de treino
        val_loader: DataLoader de validação (opcional)
        num_epochs (int): Número de épocas
        learning_rate (float): Taxa de aprendizado
        device (str): 'cpu' ou 'cuda'
        early_stopping_patience (int): Paciência para early stopping (None = desabilitado)
        save_path (str): Caminho para salvar melhor modelo
        verbose (bool): Se deve mostrar progresso

    Returns:
        dict: Histórico de treinamento com 'train_loss' e 'val_loss'

    Exemplo:
        >>> from src.models.autoencoder import Autoencoder
        >>> from src.utils.data_loader import load_mnist
        >>>
        >>> model = Autoencoder(latent_dim=2)
        >>> train_loader, val_loader, _ = load_mnist(batch_size=128)
        >>> history = train_model(model, train_loader, val_loader,
        ...                       num_epochs=20, device='cuda')
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Early stopping
    early_stopper = None
    if early_stopping_patience is not None:
        early_stopper = EarlyStopping(patience=early_stopping_patience,
                                     mode='min', verbose=verbose)

    # Histórico
    history = {
        'train_loss': [],
        'val_loss': []
    }

    # Loop de treinamento
    for epoch in range(num_epochs):
        # Fase de treino
        model.train()
        train_loss = 0.0

        iterator = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}') if verbose else train_loader

        for batch_idx, (data, _) in enumerate(iterator):
            data = data.view(-1, model.input_dim).to(device)

            # Forward pass
            reconstructed, _ = model(data)

            # Loss (MSE ou BCE)
            loss = F.mse_loss(reconstructed, data, reduction='sum')

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # Atualiza barra de progresso
            if verbose and isinstance(iterator, tqdm):
                iterator.set_postfix({'loss': loss.item() / len(data)})

        # Loss médio de treino
        avg_train_loss = train_loss / len(train_loader.dataset)
        history['train_loss'].append(avg_train_loss)

        # Fase de validação
        avg_val_loss = None
        if val_loader is not None:
            model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for data, _ in val_loader:
                    data = data.view(-1, model.input_dim).to(device)
                    reconstructed, _ = model(data)
                    loss = F.mse_loss(reconstructed, data, reduction='sum')
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader.dataset)
            history['val_loss'].append(avg_val_loss)

            if verbose:
                print(f'Epoch {epoch+1}: Train Loss = {avg_train_loss:.6f}, '
                      f'Val Loss = {avg_val_loss:.6f}')

            # Early stopping
            if early_stopper is not None:
                if early_stopper(avg_val_loss, epoch):
                    print(f"Early stopping at epoch {epoch+1}")
                    break

                # Salva melhor modelo
                if save_path and early_stopper.counter == 0:
                    torch.save(model.state_dict(), save_path)
                    if verbose:
                        print(f"Model saved to {save_path}")
        else:
            if verbose:
                print(f'Epoch {epoch+1}: Train Loss = {avg_train_loss:.6f}')

    return history


def train_vae(model, train_loader, val_loader=None,
              num_epochs=50, learning_rate=1e-3, beta=1.0,
              device='cpu', early_stopping_patience=None,
              save_path=None, verbose=True):
    """
    Treina um VAE (Variational Autoencoder).

    Args:
        model: Modelo VAE ou BetaVAE
        train_loader: DataLoader de treino
        val_loader: DataLoader de validação (opcional)
        num_epochs (int): Número de épocas
        learning_rate (float): Taxa de aprendizado
        beta (float): Peso da KL divergence (1.0 para VAE padrão, >1 para Beta-VAE)
        device (str): 'cpu' ou 'cuda'
        early_stopping_patience (int): Paciência para early stopping
        save_path (str): Caminho para salvar melhor modelo
        verbose (bool): Se deve mostrar progresso

    Returns:
        dict: Histórico com 'train_loss', 'train_recon', 'train_kl',
              'val_loss', 'val_recon', 'val_kl'

    Exemplo:
        >>> from src.models.vae import VAE
        >>> from src.utils.data_loader import load_mnist
        >>>
        >>> vae = VAE(latent_dim=10)
        >>> train_loader, val_loader, _ = load_mnist(batch_size=128)
        >>> history = train_vae(vae, train_loader, val_loader,
        ...                     num_epochs=30, beta=4.0, device='cuda')
    """
    from src.models.vae import vae_loss

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Early stopping
    early_stopper = None
    if early_stopping_patience is not None:
        early_stopper = EarlyStopping(patience=early_stopping_patience,
                                     mode='min', verbose=verbose)

    # Histórico
    history = {
        'train_loss': [],
        'train_recon': [],
        'train_kl': [],
        'val_loss': [],
        'val_recon': [],
        'val_kl': []
    }

    # Loop de treinamento
    for epoch in range(num_epochs):
        # Fase de treino
        model.train()
        train_total_loss = 0.0
        train_recon_loss = 0.0
        train_kl_loss = 0.0

        iterator = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}') if verbose else train_loader

        for batch_idx, (data, _) in enumerate(iterator):
            data = data.view(-1, model.input_dim).to(device)

            # Forward pass
            reconstructed, mu, logvar, z = model(data)

            # Loss
            loss_dict = vae_loss(reconstructed, data, mu, logvar, beta=beta)
            loss = loss_dict['total']

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Acumula losses
            train_total_loss += loss.item()
            train_recon_loss += loss_dict['reconstruction'].item()
            train_kl_loss += loss_dict['kl'].item()

            # Atualiza barra de progresso
            if verbose and isinstance(iterator, tqdm):
                iterator.set_postfix({
                    'loss': loss.item() / len(data),
                    'recon': loss_dict['reconstruction'].item() / len(data),
                    'kl': loss_dict['kl'].item() / len(data)
                })

        # Losses médios de treino
        n_train = len(train_loader.dataset)
        avg_train_loss = train_total_loss / n_train
        avg_train_recon = train_recon_loss / n_train
        avg_train_kl = train_kl_loss / n_train

        history['train_loss'].append(avg_train_loss)
        history['train_recon'].append(avg_train_recon)
        history['train_kl'].append(avg_train_kl)

        # Fase de validação
        avg_val_loss = None
        if val_loader is not None:
            model.eval()
            val_total_loss = 0.0
            val_recon_loss = 0.0
            val_kl_loss = 0.0

            with torch.no_grad():
                for data, _ in val_loader:
                    data = data.view(-1, model.input_dim).to(device)
                    reconstructed, mu, logvar, z = model(data)

                    loss_dict = vae_loss(reconstructed, data, mu, logvar, beta=beta)

                    val_total_loss += loss_dict['total'].item()
                    val_recon_loss += loss_dict['reconstruction'].item()
                    val_kl_loss += loss_dict['kl'].item()

            n_val = len(val_loader.dataset)
            avg_val_loss = val_total_loss / n_val
            avg_val_recon = val_recon_loss / n_val
            avg_val_kl = val_kl_loss / n_val

            history['val_loss'].append(avg_val_loss)
            history['val_recon'].append(avg_val_recon)
            history['val_kl'].append(avg_val_kl)

            if verbose:
                print(f'Epoch {epoch+1}:')
                print(f'  Train - Loss: {avg_train_loss:.4f}, '
                      f'Recon: {avg_train_recon:.4f}, KL: {avg_train_kl:.4f}')
                print(f'  Val   - Loss: {avg_val_loss:.4f}, '
                      f'Recon: {avg_val_recon:.4f}, KL: {avg_val_kl:.4f}')

            # Early stopping
            if early_stopper is not None:
                if early_stopper(avg_val_loss, epoch):
                    print(f"Early stopping at epoch {epoch+1}")
                    break

                # Salva melhor modelo
                if save_path and early_stopper.counter == 0:
                    torch.save(model.state_dict(), save_path)
                    if verbose:
                        print(f"Model saved to {save_path}")
        else:
            if verbose:
                print(f'Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, '
                      f'Recon: {avg_train_recon:.4f}, KL: {avg_train_kl:.4f}')

    return history


def evaluate_model(model, test_loader, device='cpu', is_vae=False, beta=1.0):
    """
    Avalia um modelo no conjunto de teste.

    Args:
        model: Modelo Autoencoder ou VAE
        test_loader: DataLoader de teste
        device (str): 'cpu' ou 'cuda'
        is_vae (bool): Se é um VAE (usa vae_loss ao invés de MSE)
        beta (float): Beta para VAE

    Returns:
        dict: Métricas de avaliação

    Exemplo:
        >>> metrics = evaluate_model(model, test_loader, device='cuda', is_vae=True)
        >>> print(f"Test Loss: {metrics['loss']:.4f}")
    """
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    recon_loss = 0.0
    kl_loss = 0.0

    if is_vae:
        from src.models.vae import vae_loss

    with torch.no_grad():
        for data, _ in test_loader:
            data = data.view(-1, model.input_dim).to(device)

            if is_vae:
                reconstructed, mu, logvar, z = model(data)
                loss_dict = vae_loss(reconstructed, data, mu, logvar, beta=beta)
                total_loss += loss_dict['total'].item()
                recon_loss += loss_dict['reconstruction'].item()
                kl_loss += loss_dict['kl'].item()
            else:
                reconstructed, _ = model(data)
                loss = F.mse_loss(reconstructed, data, reduction='sum')
                total_loss += loss.item()

    n_samples = len(test_loader.dataset)

    if is_vae:
        return {
            'loss': total_loss / n_samples,
            'reconstruction': recon_loss / n_samples,
            'kl': kl_loss / n_samples
        }
    else:
        return {
            'loss': total_loss / n_samples
        }
