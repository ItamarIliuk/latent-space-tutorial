# Arquitetura do Projeto

## Visão Geral

```
latent-space-tutorial/
├── src/                    # Código fonte principal
│   ├── models/            # Modelos (Autoencoder, VAE, Beta-VAE)
│   ├── utils/             # Utilitários (data, training, visualization)
│   └── experiments/       # Experimentos (explorer, comparisons)
├── examples/              # Scripts de exemplo
├── notebooks/             # Jupyter notebooks educacionais
├── docs/                  # Documentação técnica
├── data/                  # Datasets (auto-download)
└── output/                # Outputs de treinamento
```

---

## Módulos Principais

### 1. `src/models/`

#### `autoencoder.py`
```python
class Autoencoder(nn.Module):
    """Autoencoder básico determinístico."""

    def __init__(self, input_dim=784, latent_dim=2, hidden_dims=[512, 256, 128])
    def encode(x) → z
    def decode(z) → x'
    def forward(x) → (x', z)
    def get_latent_codes(dataloader) → (codes, labels)
```

**Arquitetura:**
```
Input (784) → FC(512) → ReLU → BN → FC(256) → ReLU → BN →
FC(128) → ReLU → BN → FC(latent_dim)
                                      ↓
                                    Latent
                                      ↓
FC(latent_dim) → FC(128) → ReLU → BN → FC(256) → ReLU → BN →
FC(512) → ReLU → BN → FC(784) → Sigmoid
```

#### `vae.py`
```python
class VAE(nn.Module):
    """Variational Autoencoder probabilístico."""

    def __init__(self, input_dim=784, latent_dim=2, hidden_dims=[512, 256])
    def encode(x) → (μ, logvar)
    def reparameterize(μ, logvar) → z
    def decode(z) → x'
    def forward(x) → (x', μ, logvar, z)
    def sample(num_samples) → generated_images
    def interpolate(x1, x2, num_steps) → interpolated_images

def vae_loss(x_recon, x_orig, mu, logvar, beta=1.0) → dict
```

**Arquitetura:**
```
Input → Encoder Base → ┬─→ fc_mu → μ
                       └─→ fc_logvar → logvar
                                      ↓
                        z = μ + ε·exp(0.5·logvar), ε ~ N(0,1)
                                      ↓
                                   Decoder → Output
```

#### `beta_vae.py`
```python
class BetaVAE(VAE):
    """Beta-VAE com regularização ajustável."""
    def __init__(..., beta=1.0)

class AnnealedBetaVAE(VAE):
    """Beta-VAE com annealing."""
    def __init__(..., max_beta=4.0, anneal_steps=10)
    def get_beta() → current_beta
```

---

### 2. `src/utils/`

#### `data_loader.py`
```python
def load_mnist(batch_size=128, data_dir='./data',
               val_split=0.1) → (train_loader, val_loader, test_loader)

class MNISTDataModule:
    def setup()
    def train_dataloader() → DataLoader
    def val_dataloader() → DataLoader
    def test_dataloader() → DataLoader
```

#### `training.py`
```python
class EarlyStopping:
    """Early stopping com patience."""
    def __call__(score, epoch) → bool (stop?)

def train_model(model, train_loader, val_loader,
                num_epochs, ...) → history

def train_vae(model, train_loader, val_loader,
              num_epochs, beta=1.0, ...) → history

def evaluate_model(model, test_loader, is_vae=False) → metrics
```

#### `visualization.py`
```python
def visualize_latent_space(model, dataloader, ...)
def plot_reconstructions(model, dataloader, n_samples=10, ...)
def plot_vae_results(model, dataloader, ...)
def plot_latent_grid(model, n_samples=20, ...)
def plot_interpolation(model, x1, x2, num_steps=10, ...)
def plot_training_history(history, ...)
```

---

### 3. `src/experiments/`

#### `latent_explorer.py`
```python
class LatentExplorer:
    """Exploração interativa do espaço latente."""

    def __init__(model, latent_dim, device='cpu')
    def decode_latent(z) → image
    def sample_random() → z
    def launch_interactive(backend='ipywidgets')
    def generate_grid_walk(dim1, dim2, n_steps)
    def interpolate_between_samples(n_samples, steps)
    def find_nearest_sample(target_z, dataloader, n_nearest)
```

#### `beta_comparison.py`
```python
class BetaVAEComparison:
    """Comparação de diferentes valores de beta."""

    def __init__(model_class, model_kwargs, device='cpu')
    def train_single_beta(beta, train_loader, ...) → results
    def compare_betas(betas, train_loader, ...) → all_results
    def plot_comparison()
    def plot_latent_spaces(dataloader)
    def plot_reconstructions(dataloader)
    def get_summary_table() → DataFrame
    def export_results(filepath)
```

---

## Fluxo de Dados

### Treinamento

```
1. load_mnist() → DataLoaders
                      ↓
2. Create model (VAE/Autoencoder)
                      ↓
3. train_vae() ou train_model()
   ├─ Loop: epochs
   │   ├─ Loop: batches
   │   │   ├─ Forward pass
   │   │   ├─ Calculate loss
   │   │   ├─ Backward pass
   │   │   └─ Update weights
   │   ├─ Validation
   │   └─ Early stopping check
   └─ Return history
                      ↓
4. Visualization
   ├─ plot_training_history()
   ├─ visualize_latent_space()
   └─ plot_vae_results()
```

### Inference / Geração

```
Trained model
      ↓
Option 1: Encode existing image
      x → model.encode() → (μ, logvar) → z

Option 2: Sample from prior
      z ~ N(0, I)

      ↓
model.decode(z) → x'
      ↓
Display / Save
```

---

## Design Patterns

### 1. Herança
```python
Autoencoder (base)
      ↑
     VAE (adiciona probabilidade)
      ↑
   BetaVAE (adiciona beta)
      ↑
AnnealedBetaVAE (adiciona annealing)
```

### 2. Composição
```python
MNISTDataModule
    ├─ train_dataset
    ├─ val_dataset
    └─ test_dataset

LatentExplorer
    └─ model (VAE ou Autoencoder)

BetaVAEComparison
    └─ results {beta: {model, history}}
```

### 3. Strategy Pattern
```python
# Diferentes loss functions
train_model()  # MSE/BCE
train_vae()    # ELBO (BCE + KL)

# Diferentes backends
LatentExplorer.launch_interactive()
    ├─ backend='ipywidgets' (Jupyter)
    └─ backend='matplotlib' (standalone)
```

---

## Extensibilidade

### Adicionar Novo Modelo

```python
# 1. Criar classe herdando de VAE ou nn.Module
class MyCustomVAE(VAE):
    def __init__(self, ...):
        super().__init__(...)
        # Adicionar camadas customizadas

    def custom_method(self):
        pass

# 2. Usar funções existentes
history = train_vae(MyCustomVAE(...), ...)
visualize_latent_space(model, ...)
```

### Adicionar Nova Visualização

```python
# Em src/utils/visualization.py
def my_custom_plot(model, data, **kwargs):
    """Nova visualização."""
    # Implementação
    plt.show()
```

### Adicionar Novo Dataset

```python
# Em src/utils/data_loader.py
def load_custom_dataset(batch_size=128, **kwargs):
    """Carrega dataset customizado."""
    # Implementação
    return train_loader, val_loader, test_loader
```

---

## Convenções de Código

### Nomenclatura
- **Classes:** PascalCase (`VAE`, `LatentExplorer`)
- **Funções:** snake_case (`train_vae`, `load_mnist`)
- **Constantes:** UPPER_CASE (`DEVICE`, `BATCH_SIZE`)
- **Privado:** _underscore (`_internal_method`)

### Docstrings
```python
def function_name(arg1, arg2):
    """
    Descrição breve.

    Descrição detalhada (opcional).

    Args:
        arg1 (type): Descrição
        arg2 (type): Descrição

    Returns:
        type: Descrição do retorno

    Example:
        >>> result = function_name(1, 2)
        >>> print(result)
    """
```

### Type Hints
```python
def train_vae(
    model: nn.Module,
    train_loader: DataLoader,
    num_epochs: int = 50,
    device: str = 'cpu'
) -> Dict[str, List[float]]:
    ...
```

---

## Performance

### Otimizações
1. **DataLoader:** `num_workers`, `pin_memory`
2. **Batch Size:** Ajustar conforme GPU
3. **Mixed Precision:** torch.cuda.amp (futuro)
4. **Gradient Accumulation:** Para batch size efetivo maior

### Benchmarks (MNIST, latent_dim=10)

| Modelo | Epochs | Train Time (CPU) | Train Time (GPU) |
|--------|--------|------------------|------------------|
| Autoencoder | 20 | ~15 min | ~2 min |
| VAE | 30 | ~25 min | ~3 min |
| Beta-VAE (β=4) | 30 | ~25 min | ~3 min |

**Hardware:** Intel i7, 16GB RAM, NVIDIA RTX 3060

---

## Próximos Passos

- Ler [treinamento.md](treinamento.md) para guias de treinamento
- Ver [api_reference.md](api_reference.md) para API completa
- Explorar código em `src/`
