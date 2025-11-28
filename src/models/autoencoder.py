"""
Autoencoder: Modelo básico para compressão e reconstrução
"""

import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    """
    Autoencoder simples para demonstração didática.
    
    Comprime imagens MNIST (784 pixels) para um espaço latente de dimensão reduzida
    e depois reconstrói as imagens.
    
    Args:
        input_dim (int): Dimensão da entrada (padrão: 784 para MNIST)
        latent_dim (int): Dimensão do espaço latente (padrão: 2 para visualização 2D)
        hidden_dims (list): Lista com dimensões das camadas ocultas
    
    Exemplo:
        >>> model = Autoencoder(input_dim=784, latent_dim=2)
        >>> x = torch.randn(32, 784)  # Batch de 32 imagens
        >>> reconstructed, latent = model(x)
        >>> print(f"Entrada: {x.shape}, Latente: {latent.shape}, Saída: {reconstructed.shape}")
    """
    
    def __init__(self, input_dim=784, latent_dim=2, hidden_dims=None):
        super(Autoencoder, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        
        # Encoder: input_dim → hidden_dims[0] → ... → latent_dim
        encoder_layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
            ])
            prev_dim = hidden_dim
        
        # Camada final do encoder
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder: latent_dim → hidden_dims[-1] → ... → input_dim
        decoder_layers = []
        prev_dim = latent_dim
        
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
            ])
            prev_dim = hidden_dim
        
        # Camada final do decoder
        decoder_layers.extend([
            nn.Linear(prev_dim, input_dim),
            nn.Sigmoid()  # Para normalizar saída entre [0, 1]
        ])
        
        self.decoder = nn.Sequential(*decoder_layers)
    
    def encode(self, x):
        """
        Codifica a entrada para o espaço latente.
        
        Args:
            x (torch.Tensor): Tensor de entrada (batch_size, input_dim)
        
        Returns:
            torch.Tensor: Representação latente (batch_size, latent_dim)
        """
        return self.encoder(x)
    
    def decode(self, z):
        """
        Decodifica do espaço latente para reconstruir a entrada.
        
        Args:
            z (torch.Tensor): Códigos latentes (batch_size, latent_dim)
        
        Returns:
            torch.Tensor: Reconstrução (batch_size, input_dim)
        """
        return self.decoder(z)
    
    def forward(self, x):
        """
        Forward pass completo: encode → decode
        
        Args:
            x (torch.Tensor): Entrada (batch_size, input_dim)
        
        Returns:
            tuple: (reconstrução, código latente)
        """
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        return reconstructed, latent
    
    def get_latent_codes(self, dataloader, device='cpu'):
        """
        Extrai códigos latentes para um dataset inteiro.
        
        Args:
            dataloader: DataLoader com os dados
            device: Dispositivo ('cpu' ou 'cuda')
        
        Returns:
            tuple: (latent_codes, labels)
        """
        self.eval()
        latent_codes = []
        labels = []
        
        with torch.no_grad():
            for data, label in dataloader:
                data = data.view(-1, self.input_dim).to(device)
                _, latent = self(data)
                latent_codes.append(latent.cpu())
                labels.append(label)
        
        latent_codes = torch.cat(latent_codes, dim=0)
        labels = torch.cat(labels, dim=0)
        
        return latent_codes, labels
    
    def reconstruct(self, x):
        """
        Reconstrói uma entrada (atalho para forward()[0]).
        
        Args:
            x (torch.Tensor): Entrada
        
        Returns:
            torch.Tensor: Reconstrução
        """
        return self.forward(x)[0]
    
    def compression_ratio(self):
        """
        Calcula a taxa de compressão do modelo.
        
        Returns:
            float: Taxa de compressão (input_dim / latent_dim)
        """
        return self.input_dim / self.latent_dim
    
    def __repr__(self):
        return (f"Autoencoder(input_dim={self.input_dim}, "
                f"latent_dim={self.latent_dim}, "
                f"compression={self.compression_ratio():.1f}x)")
