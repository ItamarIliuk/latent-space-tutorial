"""
Data Loading: MNIST e outros datasets
"""

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import os


def load_mnist(batch_size=128, data_dir='./data', num_workers=0, 
               val_split=0.1, download=True):
    """
    Carrega o dataset MNIST com data loaders prontos para uso.
    
    Args:
        batch_size (int): Tamanho do batch
        data_dir (str): Diretório para salvar/carregar os dados
        num_workers (int): Número de workers para o DataLoader
        val_split (float): Proporção dos dados de treino para validação
        download (bool): Se deve baixar o dataset se não existir
    
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    
    Exemplo:
        >>> train_loader, val_loader, test_loader = load_mnist(batch_size=64)
        >>> for images, labels in train_loader:
        ...     print(f"Batch shape: {images.shape}")
        ...     break
    """
    # Transformações
    transform = transforms.Compose([
        transforms.ToTensor(),
        # Normalização opcional: transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Dataset de treino
    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=download,
        transform=transform
    )
    
    # Dataset de teste
    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=download,
        transform=transform
    )
    
    # Split treino/validação
    if val_split > 0:
        val_size = int(len(train_dataset) * val_split)
        train_size = len(train_dataset) - val_size
        
        train_dataset, val_dataset = random_split(
            train_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
    else:
        val_dataset = None
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader, test_loader


class MNISTDataModule:
    """
    Data Module para MNIST (padrão PyTorch Lightning style).
    
    Encapsula toda a lógica de carregamento de dados.
    
    Exemplo:
        >>> dm = MNISTDataModule(batch_size=64)
        >>> dm.setup()
        >>> train_loader = dm.train_dataloader()
        >>> test_loader = dm.test_dataloader()
    """
    
    def __init__(self, batch_size=128, data_dir='./data', 
                 num_workers=0, val_split=0.1):
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.num_workers = num_workers
        self.val_split = val_split
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def setup(self):
        """Prepara os datasets."""
        transform = transforms.Compose([transforms.ToTensor()])
        
        # Dataset de treino
        full_train = datasets.MNIST(
            root=self.data_dir,
            train=True,
            download=True,
            transform=transform
        )
        
        # Split treino/validação
        if self.val_split > 0:
            val_size = int(len(full_train) * self.val_split)
            train_size = len(full_train) - val_size
            
            self.train_dataset, self.val_dataset = random_split(
                full_train,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42)
            )
        else:
            self.train_dataset = full_train
            self.val_dataset = None
        
        # Dataset de teste
        self.test_dataset = datasets.MNIST(
            root=self.data_dir,
            train=False,
            download=True,
            transform=transform
        )
    
    def train_dataloader(self):
        """Retorna o data loader de treino."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available()
        )
    
    def val_dataloader(self):
        """Retorna o data loader de validação."""
        if self.val_dataset is None:
            return None
        
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available()
        )
    
    def test_dataloader(self):
        """Retorna o data loader de teste."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available()
        )
    
    def get_sample_batch(self, split='train'):
        """
        Obtém um batch de exemplo.
        
        Args:
            split (str): 'train', 'val' ou 'test'
        
        Returns:
            tuple: (images, labels)
        """
        if split == 'train':
            loader = self.train_dataloader()
        elif split == 'val':
            loader = self.val_dataloader()
        else:
            loader = self.test_dataloader()
        
        return next(iter(loader))
