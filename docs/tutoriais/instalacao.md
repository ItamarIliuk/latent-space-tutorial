# Guia de Instalação

## Requisitos do Sistema

### Hardware Mínimo
- **CPU:** Intel Core i5 ou equivalente
- **RAM:** 8 GB (16 GB recomendado)
- **Armazenamento:** 5 GB de espaço livre
- **GPU (Opcional):** NVIDIA com CUDA support para treinamento rápido

### Software
- **Sistema Operacional:** Windows 10/11, macOS 10.15+, ou Linux
- **Python:** 3.8, 3.9, 3.10, ou 3.11
- **CUDA:** 11.7+ (opcional, para GPU)

---

## Instalação

### Método 1: pip (Recomendado)

#### Passo 1: Clone o repositório
```bash
git clone https://github.com/your-repo/latent-space-tutorial.git
cd latent-space-tutorial
```

#### Passo 2: Crie ambiente virtual
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

#### Passo 3: Instale dependências
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### Passo 4: Verifique instalação
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import src; print('Installation successful!')"
```

---

### Método 2: Conda

#### Passo 1: Clone o repositório
```bash
git clone https://github.com/your-repo/latent-space-tutorial.git
cd latent-space-tutorial
```

#### Passo 2: Crie ambiente conda
```bash
conda env create -f environment.yml
conda activate latent-space-tutorial
```

#### Passo 3: Verifique instalação
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
```

---

### Método 3: Google Colab (Sem instalação local)

Para usar no Google Colab:

```python
# No Colab, execute:
!git clone https://github.com/your-repo/latent-space-tutorial.git
%cd latent-space-tutorial
!pip install -r requirements.txt

# Importar módulos
import sys
sys.path.append('/content/latent-space-tutorial')
from src.models.vae import VAE
```

---

## Configuração GPU (Opcional)

### NVIDIA CUDA

#### Windows
1. Instale [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
2. Instale [cuDNN](https://developer.nvidia.com/cudnn)
3. Instale PyTorch com CUDA:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### Linux
```bash
# Ubuntu/Debian
sudo apt-get install nvidia-cuda-toolkit

# Instalar PyTorch com CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Verificar GPU
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"Device name: {torch.cuda.get_device_name(0)}")
```

---

## Jupyter Notebooks

### Instalação
```bash
pip install jupyter ipywidgets
jupyter nbextension enable --py widgetsnbextension
```

### Uso
```bash
jupyter notebook notebooks/
```

Navegue até `01_analogia_espaco_latente.ipynb` e comece!

---

## Troubleshooting

### Erro: "No module named 'torch'"
**Solução:**
```bash
pip install torch torchvision
```

### Erro: "ModuleNotFoundError: No module named 'src'"
**Solução:**
Certifique-se de estar no diretório raiz do projeto:
```bash
pwd  # ou cd (Windows)
# Deve mostrar: .../latent-space-tutorial
```

### Erro: ImportError com ipywidgets
**Solução:**
```bash
pip install ipywidgets
jupyter nbextension enable --py widgetsnbextension
# Reinicie o Jupyter
```

### GPU não detectada
**Solução:**
1. Verifique drivers NVIDIA
2. Reinstale PyTorch com CUDA:
```bash
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## Próximos Passos

✅ Instalação completa!

**Agora você pode:**
1. Executar quick start: `python examples/quick_start.py`
2. Abrir notebooks: `jupyter notebook notebooks/`
3. Ler documentação: [arquitetura.md](arquitetura.md)
4. Treinar modelos: Ver [treinamento.md](treinamento.md)
