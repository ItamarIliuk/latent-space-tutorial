# Documentação Técnica - Latent Space Tutorial

> Tutorial educacional sobre Autoencoders, VAEs e Espaços Latentes
> **Autora:** Profa. Itamar | **Instituição:** UTFPR | **Ano:** 2024

---

## 📚 Guias de Documentação

### Começando

1. **[Instalação](tutoriais/instalacao.md)**
   - Requisitos do sistema
   - Instalação com pip/conda
   - Configuração GPU
   - Troubleshooting

2. **[Arquitetura do Projeto](tutoriais/arquitetura.md)**
   - Visão geral dos módulos
   - Estrutura de diretórios
   - Design patterns
   - Extensibilidade

3. **[Guia de Treinamento](tutoriais/treinamento.md)**
   - Quick start
   - Configurações recomendadas
   - Hyperparâmetros

4. **[Referência de API](tutoriais/api_reference.md)**
   - Models (Autoencoder, VAE, Beta-VAE)
   - Training functions
   - Visualization functions
   - Experiments classes

### Fundamentação Teórica

5. **[Conceitos Fundamentais](conceitos.md)**
   - Espaço Latente
   - Autoencoders
   - Variational Autoencoders (VAE)
   - Beta-VAE
   - Disentanglement
   - Reparameterization Trick

6. **[Fundamentos Matemáticos](matematica.md)**
   - Notação
   - ELBO (Evidence Lower Bound)
   - KL Divergence
   - Derivações completas
   - Gradientes

7. **[Referências Bibliográficas](referencias.md)**
   - Papers fundamentais
   - Papers avançados
   - Livros
   - Tutoriais e cursos
   - Implementações
   - Datasets

---

## 🚀 Quick Navigation

### Para Iniciantes
👉 **Começar aqui:**
1. [Instalação](tutoriais/instalacao.md) → Configurar ambiente
2. Notebook 01 (`notebooks/01_analogia_espaco_latente.ipynb`) → Entender conceitos
3. [Quick Start](../examples/quick_start.py) → Primeiro código
4. Notebook 02 (`notebooks/02_autoencoder_basico.ipynb`) → Implementação prática

### Para Praticantes
👉 **Treinar modelos:**
1. [Guia de Treinamento](tutoriais/treinamento.md) → Configurações
2. Scripts em `examples/` → Treinamento completo
3. [API Reference](tutoriais/api_reference.md) → Uso programático

### Para Pesquisadores
👉 **Exploração avançada:**
1. [Conceitos](conceitos.md) → Teoria
2. [Matemática](matematica.md) → Derivações
3. [Referências](referencias.md) → Papers
4. Notebook 06 (`notebooks/06_exemplos_avancados.ipynb`) → Aplicações

---

## 📋 Estrutura do Projeto

```
latent-space-tutorial/
├── 📘 docs/                    # Documentação técnica (você está aqui)
│   ├── index.md              # Este arquivo
│   ├── conceitos.md          # Conceitos fundamentais
│   ├── matematica.md         # Matemática e derivações
│   ├── referencias.md        # Bibliografia completa
│   └── tutoriais/           # Guias práticos
│       ├── instalacao.md
│       ├── arquitetura.md
│       ├── treinamento.md
│       └── api_reference.md
│
├── 💻 src/                    # Código fonte
│   ├── models/              # Autoencoder, VAE, Beta-VAE
│   ├── utils/               # Training, visualization, data
│   └── experiments/         # LatentExplorer, BetaComparison
│
├── 📓 notebooks/             # 6 notebooks educacionais
│   ├── 01_analogia_espaco_latente.ipynb
│   ├── 02_autoencoder_basico.ipynb
│   ├── 03_vae_explicativo.ipynb
│   ├── 04_beta_vae_experimento.ipynb
│   ├── 05_exploracao_interativa.ipynb
│   └── 06_exemplos_avancados.ipynb
│
├── 🎯 examples/              # Scripts de exemplo
│   ├── quick_start.py
│   ├── train_autoencoder.py
│   ├── train_vae.py
│   └── explore_latent_space.py
│
├── 📄 README.md              # Visão geral do projeto
├── 📄 requirements.txt       # Dependências pip
└── 📄 environment.yml        # Ambiente conda
```

---

## 🎯 Fluxo de Aprendizado Recomendado

### Nível 1: Fundamentos (1-2 semanas)
```
1. Ler README.md
2. Instalação → tutoriais/instalacao.md
3. Conceitos → conceitos.md
4. Notebook 01 → Analogias
5. Notebook 02 → Autoencoder prático
```

### Nível 2: VAE e Aplicações (2-3 semanas)
```
6. Matemática → matematica.md (seções básicas)
7. Notebook 03 → VAE explicativo
8. Treinar modelos → examples/train_vae.py
9. Notebook 04 → Beta-VAE experimentos
```

### Nível 3: Exploração Avançada (1-2 semanas)
```
10. Notebook 05 → Exploração interativa
11. Notebook 06 → Exemplos avançados
12. Matemática → matematica.md (derivações completas)
13. Referências → referencias.md (papers)
14. Implementar projeto próprio
```

---

## 📊 Tabela de Conteúdos por Tópico

| Tópico | Conceitos | Matemática | Notebook | Script |
|--------|-----------|------------|----------|--------|
| **Espaço Latente** | ✅ conceitos.md | ✅ matematica.md | 01 | - |
| **Autoencoder** | ✅ conceitos.md | ✅ matematica.md | 02 | train_autoencoder.py |
| **VAE** | ✅ conceitos.md | ✅ matematica.md | 03 | train_vae.py |
| **Beta-VAE** | ✅ conceitos.md | ✅ matematica.md | 04 | train_vae.py |
| **Disentanglement** | ✅ conceitos.md | ✅ matematica.md | 04 | - |
| **Reparametrização** | ✅ conceitos.md | ✅ matematica.md | 03 | - |
| **Exploração** | - | - | 05 | explore_latent_space.py |
| **Aplicações** | ✅ conceitos.md | - | 06 | - |

---

## 🛠️ Recursos Adicionais

### Código-Fonte Documentado
Todo código em `src/` contém docstrings detalhadas. Exemplo:
```python
from src.models.vae import VAE
help(VAE)  # Mostra documentação completa
```

### Visualizações
Todas as funções de visualização estão em `src/utils/visualization.py`:
- `visualize_latent_space()` - Espaço latente 2D
- `plot_reconstructions()` - Originais vs reconstruções
- `plot_vae_results()` - Overview completo
- E mais...

### Experimentos
Classes para experimentação sistemática:
- `LatentExplorer` - Exploração interativa
- `BetaVAEComparison` - Comparação de betas

---

## 🔗 Links Rápidos

**Documentação:**
- [Instalação](tutoriais/instalacao.md)
- [Arquitetura](tutoriais/arquitetura.md)
- [API Reference](tutoriais/api_reference.md)
- [Conceitos](conceitos.md)
- [Matemática](matematica.md)
- [Referências](referencias.md)

**Código:**
- [src/models/](../src/models/) - Modelos
- [src/utils/](../src/utils/) - Utilitários
- [examples/](../examples/) - Scripts de exemplo

**Educacional:**
- [notebooks/](../notebooks/) - Jupyter notebooks
- [README.md](../README.md) - Visão geral

---

## 📞 Suporte

**Encontrou um problema?**
- Verifique [Instalação - Troubleshooting](tutoriais/instalacao.md#troubleshooting)
- Consulte [API Reference](tutoriais/api_reference.md)
- Abra uma issue no repositório

**Quer contribuir?**
- Leia [CONTRIBUTING.md](../CONTRIBUTING.md)
- Veja [Arquitetura](tutoriais/arquitetura.md) para entender estrutura

---

## 📝 Como Citar

```bibtex
@misc{latentspace2025,
  author = {Professora Itamar},
  title = {Latent Space Tutorial: Autoencoders e VAEs Educacional},
  year = {2025},
  publisher = {UTFPR},
  howpublished = {\url{https://github.com/your-repo/latent-space-tutorial}}
}
```

---

## ✨ Começar Agora

**Novo no projeto?**
```bash
# 1. Clone e instale
git clone https://github.com/your-repo/latent-space-tutorial.git
cd latent-space-tutorial
pip install -r requirements.txt

# 2. Execute quick start
python examples/quick_start.py

# 3. Abra primeiro notebook
jupyter notebook notebooks/01_analogia_espaco_latente.ipynb
```

**Bons estudos!** 🚀
