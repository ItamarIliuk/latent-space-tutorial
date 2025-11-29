# 🧠 Tutorial de Espaço Latente: Autoencoders e VAEs

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Made with ❤️](https://img.shields.io/badge/Made%20with-❤️-red.svg)](https://github.com/itamar15/latent-space-tutorial)

> 📚 Material didático completo sobre espaço latente, autoencoders e Variational Autoencoders (VAEs), desenvolvido para aulas de IA Generativa e Aprendizado Profundo.

## 🎯 Sobre o Projeto

Este repositório contém material didático completo para ensinar os conceitos de **espaço latente**, **autoencoders** e **VAEs** (Variational Autoencoders). Foi desenvolvido com foco em clareza, visualizações interativas e exemplos práticos que podem ser usados diretamente em sala de aula.

### ✨ Destaques

- 📊 **6 Notebooks Jupyter** progressivos e interativos
- 🎨 **Visualizações ricas** para exploração do espaço latente
- 🔬 **Experimentos comparativos** (Beta-VAE)
- 🛠️ **Código modular** e reutilizável
- 📖 **Documentação completa** com fundamentos matemáticos
- 🎮 **Interface interativa** para exploração
- ✅ **Testes unitários** incluídos

## 📋 Conteúdo

### 📓 Notebooks

1. **[Analogia: O Mapa do Tesouro](notebooks/01_analogia_espaco_latente.ipynb)** - Introdução conceitual visual
2. **[Autoencoder Básico](notebooks/02_autoencoder_basico.ipynb)** - Implementação e compressão 784→2
3. **[VAE Explicativo](notebooks/03_vae_explicativo.ipynb)** - Espaço latente probabilístico
4. **[Experimento Beta-VAE](notebooks/04_beta_vae_experimento.ipynb)** - Análise comparativa de β
5. **[Exploração Interativa](notebooks/05_exploracao_interativa.ipynb)** - Interface para navegar no espaço latente
6. **[Exemplos Avançados](notebooks/06_exemplos_avancados.ipynb)** - Aplicações práticas

### 📚 Documentação

- [Conceitos Fundamentais](docs/conceitos.md)
- [Fundamentos Matemáticos](docs/matematica.md)
- [Referências e Leituras](docs/referencias.md)
- [Tutoriais Passo a Passo](docs/tutoriais/)

### 🔧 Módulos Python

```
src/
├── models/          # Implementações de Autoencoder, VAE, Beta-VAE
├── utils/           # Utilidades para dados, visualização e treinamento
└── experiments/     # Experimentos e análises
```

## 🚀 Instalação Rápida

### Pré-requisitos

- Python 3.8 ou superior
- pip ou conda

### Opção 1: pip

```bash
# Clone o repositório
git clone https://github.com/itamar15/latent-space-tutorial.git
cd latent-space-tutorial

# Crie um ambiente virtual
python -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate

# Instale as dependências
pip install -r requirements.txt

# Instale o pacote em modo desenvolvimento
pip install -e .
```

### Opção 2: conda

```bash
# Clone o repositório
git clone https://github.com/itamar15/latent-space-tutorial.git
cd latent-space-tutorial

# Crie o ambiente
conda env create -f environment.yml
conda activate latent-space

# Instale o pacote
pip install -e .
```

## 🎓 Guia Rápido para Professores

### Plano de Aula Sugerido (90 minutos)

| Tempo | Atividade | Notebook |
|-------|-----------|----------|
| 0-15 min | Introdução conceitual | `01_analogia_espaco_latente.ipynb` |
| 15-35 min | Autoencoder prático | `02_autoencoder_basico.ipynb` |
| 35-40 min | ☕ Pausa | - |
| 40-65 min | VAE e probabilidade | `03_vae_explicativo.ipynb` |
| 65-80 min | Experimento Beta-VAE | `04_beta_vae_experimento.ipynb` |
| 80-90 min | Exploração livre | `05_exploracao_interativa.ipynb` |

### Material Complementar

- 📊 Slides (em `docs/slides/`)
- 🎯 Exercícios práticos
- 📝 Gabaritos de resolução

## 💻 Uso Rápido

### Exemplo 1: Treinar um Autoencoder

```python
from src.models import Autoencoder
from src.utils import load_mnist, train_model, visualize_latent_space

# Carregar dados
train_loader, test_loader = load_mnist(batch_size=128)

# Criar modelo
model = Autoencoder(input_dim=784, latent_dim=2)

# Treinar
train_model(model, train_loader, epochs=10)

# Visualizar espaço latente
visualize_latent_space(model, test_loader)
```

### Exemplo 2: Treinar um VAE

```python
from src.models import VAE
from src.utils import load_mnist, train_vae, plot_vae_results

# Carregar dados
train_loader, test_loader = load_mnist(batch_size=128)

# Criar VAE
vae = VAE(input_dim=784, latent_dim=2)

# Treinar
train_vae(vae, train_loader, epochs=20)

# Visualizar resultados
plot_vae_results(vae, test_loader)
```

### Exemplo 3: Explorar Espaço Latente

```python
from src.experiments import LatentExplorer

# Criar explorador
explorer = LatentExplorer(vae)

# Explorar dimensão específica
explorer.explore_dimension(dim=0, n_steps=10)

# Interpolar entre pontos
explorer.interpolate(point_a=[1.0, 1.0], point_b=[-1.0, -1.0], steps=10)

# Criar caminho no espaço latente
explorer.create_path([[0, 0], [2, 1], [1, -2]], steps_between=5)
```

## 📊 Resultados Esperados

### Autoencoder
- Compressão: 784 → 2 dimensões (~392x)
- Reconstrução: MSE < 0.05
- Tempo de treinamento: ~2 min (GPU) / ~10 min (CPU)

### VAE
- Geração de novas amostras: ✅
- Interpolação suave: ✅
- Espaço latente contínuo: ✅

## 🔬 Experimentos Incluídos

### 1. Comparação Beta-VAE

```python
from src.experiments import BetaVAEComparison

experiment = BetaVAEComparison()
experiment.compare_betas([0.5, 1.0, 2.0, 4.0])
```

**Resultado esperado:** 
- β baixo → melhor reconstrução, espaço latente menos estruturado
- β alto → pior reconstrução, melhor separação de conceitos

### 2. Análise de Disentanglement

```python
from src.experiments import analyze_disentanglement

results = analyze_disentanglement(vae, test_loader)
print(f"Score: {results['score']:.3f}")
```

## 🧪 Executar Testes

```bash
# Todos os testes
pytest tests/

# Testes específicos
pytest tests/test_models.py
pytest tests/test_utils.py

# Com cobertura
pytest --cov=src tests/
```

## 📖 Documentação Adicional

- [Conceitos Fundamentais](docs/conceitos.md) - O que é espaço latente?
- [Matemática dos VAEs](docs/matematica.md) - Derivações e provas
- [Tutoriais Detalhados](docs/tutoriais/) - Guias passo a passo
- [Referências](docs/referencias.md) - Papers e recursos

## 🤝 Contribuindo

Contribuições são bem-vindas! Por favor:

1. Faça um fork do projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

Veja [CONTRIBUTING.md](CONTRIBUTING.md) para mais detalhes.

## 📝 Citação

Se você usar este material em suas aulas ou pesquisas, por favor cite:

```bibtex
@misc{latent_space_tutorial,
  author = {Profa. Itamar},
  title = {Tutorial de Espaço Latente: Autoencoders e VAEs},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/itamar15/latent-space-tutorial}
}
```

## 👥 Autora

- **Profa. Itamar** - Professora de Ciência da Computação, UTFPR Campus Ponta Grossa - [GitHub](https://github.com/itamar15)

## 📄 Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

## 🙏 Agradecimentos

- Dataset MNIST (Yann LeCun et al.)
- PyTorch team
- Comunidade de Machine Learning
- Meus alunos da UTFPR

## 📬 Contato

Profa. Itamar - UTFPR Campus Ponta Grossa

Link do Projeto: [https://github.com/itamar15/latent-space-tutorial](https://github.com/itamar15/latent-space-tutorial)

---

<p align="center">
  Feito com ❤️ para ensino de IA Generativa
</p>

<p align="center">
  <a href="#-sobre-o-projeto">Topo</a> •
  <a href="#-instalação-rápida">Instalação</a> •
  <a href="#-uso-rápido">Uso</a> •
  <a href="#-documentação-adicional">Docs</a> •
  <a href="#-licença">Licença</a>
</p>
