# API Reference

## Models

### Autoencoder
```python
from src.models.autoencoder import Autoencoder

model = Autoencoder(input_dim=784, latent_dim=2, hidden_dims=[512, 256, 128])
reconstructed, latent = model(x)
```

### VAE
```python
from src.models.vae import VAE, vae_loss

vae = VAE(input_dim=784, latent_dim=10, hidden_dims=[512, 256])
recon, mu, logvar, z = vae(x)
loss_dict = vae_loss(recon, x, mu, logvar, beta=1.0)
```

### Beta-VAE
```python
from src.models.beta_vae import BetaVAE, AnnealedBetaVAE

beta_vae = BetaVAE(input_dim=784, latent_dim=10, beta=4.0)
annealed = AnnealedBetaVAE(input_dim=784, latent_dim=10, max_beta=4.0, anneal_steps=10)
```

## Training

```python
from src.utils.training import train_model, train_vae, EarlyStopping

# Autoencoder
history = train_model(model, train_loader, val_loader, num_epochs=20, device='cuda')

# VAE
history = train_vae(vae, train_loader, val_loader, num_epochs=30, beta=1.0, device='cuda')
```

## Visualization

```python
from src.utils.visualization import *

visualize_latent_space(model, test_loader, device='cuda')
plot_reconstructions(model, test_loader, n_samples=10)
plot_vae_results(vae, test_loader)
```

## Experiments

```python
from src.experiments import LatentExplorer, BetaVAEComparison

explorer = LatentExplorer(vae, latent_dim=10)
explorer.launch_interactive()

comparison = BetaVAEComparison(VAE, {'latent_dim': 10})
results = comparison.compare_betas([1.0, 2.0, 4.0], train_loader, val_loader)
```

## Ver Código Fonte

Consulte `src/` para implementações completas e docstrings detalhadas.
