# Referências Bibliográficas

## Índice
- [Papers Fundamentais](#papers-fundamentais)
- [Papers Avançados](#papers-avançados)
- [Livros](#livros)
- [Tutoriais e Cursos](#tutoriais-e-cursos)
- [Implementações](#implementações)
- [Datasets](#datasets)
- [Ferramentas](#ferramentas)

---

## Papers Fundamentais

### Variational Autoencoders

**1. Auto-Encoding Variational Bayes (VAE Original)**
- **Autores:** Kingma, D. P., & Welling, M.
- **Ano:** 2013
- **Conferência:** ICLR 2014
- **Link:** https://arxiv.org/abs/1312.6114
- **Citações:** 20,000+
- **Importância:** Paper original que introduziu VAEs
- **Contribuições:**
  - Reparameterization trick
  - ELBO como objetivo de treinamento
  - Framework probabilístico para autoencoders

**2. Stochastic Backpropagation and Approximate Inference**
- **Autores:** Rezende, D. J., Mohamed, S., & Wierstra, D.
- **Ano:** 2014
- **Conferência:** ICML 2014
- **Link:** https://arxiv.org/abs/1401.4082
- **Importância:** Desenvolveu ideias similares ao VAE independentemente
- **Contribuições:**
  - Gradientes estocásticos para variáveis latentes
  - Inference aproximada

### Beta-VAE e Disentanglement

**3. β-VAE: Learning Basic Visual Concepts**
- **Autores:** Higgins, I., et al.
- **Ano:** 2017
- **Conferência:** ICLR 2017
- **Link:** https://openreview.net/forum?id=Sy2fzU9gl
- **Importância:** Introduziu β-VAE para disentanglement
- **Contribuições:**
  - Parâmetro β para controlar regularização
  - Métricas de disentanglement
  - Conexão entre β e independência de fatores

**4. Understanding disentangling in β-VAE**
- **Autores:** Burgess, C. P., et al.
- **Ano:** 2018
- **Workshop:** NeurIPS 2018 Workshop
- **Link:** https://arxiv.org/abs/1804.03599
- **Contribuições:**
  - Análise teórica de β-VAE
  - Annealed β para melhor convergência
  - Experimentos sistemáticos

**5. Isolating Sources of Disentanglement (β-TCVAE)**
- **Autores:** Chen, R. T., et al.
- **Ano:** 2018
- **Conferência:** NeurIPS 2018
- **Link:** https://arxiv.org/abs/1802.04942
- **Contribuições:**
  - Decomposição da KL divergence
  - Foco em Total Correlation
  - Melhor disentanglement que β-VAE

---

## Papers Avançados

### Arquiteturas e Variantes

**6. Conditional VAE (CVAE)**
- **Título:** Learning Structured Output Representation using Deep Conditional Generative Models
- **Autores:** Sohn, K., Lee, H., & Yan, X.
- **Ano:** 2015
- **Link:** https://papers.nips.cc/paper/5775-learning-structured-output-representation-using-deep-conditional-generative-models
- **Contribuições:** VAE condicionado em labels

**7. Hierarchical VAE**
- **Título:** Ladder Variational Autoencoders
- **Autores:** Sønderby, C. K., et al.
- **Ano:** 2016
- **Link:** https://arxiv.org/abs/1602.02282
- **Contribuições:** Múltiplos níveis latentes hierárquicos

**8. VQ-VAE**
- **Título:** Neural Discrete Representation Learning
- **Autores:** van den Oord, A., Vinyals, O., & Kavukcuoglu, K.
- **Ano:** 2017
- **Link:** https://arxiv.org/abs/1711.00937
- **Contribuições:** Quantização vetorial do espaço latente

**9. FactorVAE**
- **Título:** Disentangling by Factorising
- **Autores:** Kim, H., & Mnih, A.
- **Ano:** 2018
- **Link:** https://arxiv.org/abs/1802.05983
- **Contribuições:** Discriminador para disentanglement

### Aplicações

**10. VAE for Anomaly Detection**
- **Título:** Variational Autoencoder based Anomaly Detection using Reconstruction Probability
- **Autores:** An, J., & Cho, S.
- **Ano:** 2015
- **Link:** https://arxiv.org/abs/1802.03903
- **Aplicação:** Detecção de anomalias

**11. Semi-Supervised Learning with VAE**
- **Título:** Semi-supervised Learning with Deep Generative Models
- **Autores:** Kingma, D. P., et al.
- **Ano:** 2014
- **Link:** https://arxiv.org/abs/1406.5298
- **Aplicação:** Aprendizado semi-supervisionado

---

## Livros

### Deep Learning

**1. Deep Learning**
- **Autores:** Goodfellow, I., Bengio, Y., & Courville, A.
- **Ano:** 2016
- **Editora:** MIT Press
- **Link:** https://www.deeplearningbook.org/
- **Capítulos Relevantes:**
  - Chapter 14: Autoencoders
  - Chapter 20: Deep Generative Models
- **Nível:** Intermediário a Avançado

**2. Pattern Recognition and Machine Learning**
- **Autor:** Bishop, C. M.
- **Ano:** 2006
- **Editora:** Springer
- **Capítulos Relevantes:**
  - Chapter 9: Mixture Models and EM
  - Chapter 10: Approximate Inference
- **Nível:** Avançado

**3. Probabilistic Machine Learning: An Introduction**
- **Autor:** Murphy, K. P.
- **Ano:** 2022
- **Editora:** MIT Press
- **Link:** https://probml.github.io/pml-book/book1.html
- **Capítulos Relevantes:**
  - Chapter 20: Variational Inference
  - Chapter 21: Variational Autoencoders
- **Nível:** Intermediário

---

## Tutoriais e Cursos

### Tutoriais Escritos

**1. Tutorial on Variational Autoencoders**
- **Autor:** Doersch, C.
- **Ano:** 2016
- **Link:** https://arxiv.org/abs/1606.05908
- **Descrição:** Tutorial matemático detalhado sobre VAEs
- **Nível:** Intermediário
- **Recomendação:** ⭐⭐⭐⭐⭐ Excelente introdução

**2. From Autoencoder to Beta-VAE**
- **Autor:** Lil'Log (Lilian Weng)
- **Link:** https://lilianweng.github.io/posts/2018-08-12-vae/
- **Descrição:** Blog post detalhado cobrindo VAE e variantes
- **Nível:** Intermediário
- **Recursos:** Visualizações, código, matemática

**3. Understanding Variational Autoencoders**
- **Autor:** Jaan Altosaar
- **Link:** https://jaan.io/what-is-variational-autoencoder-vae-tutorial/
- **Descrição:** Tutorial interativo com visualizações
- **Nível:** Iniciante a Intermediário

### Cursos Online

**1. CS231n: Convolutional Neural Networks (Stanford)**
- **Instrutor:** Fei-Fei Li, Andrej Karpathy
- **Link:** http://cs231n.stanford.edu/
- **Aula Relevante:** Lecture 13 - Generative Models
- **Materiais:** Slides, vídeos, assignments

**2. Deep Learning Specialization (Coursera)**
- **Instrutor:** Andrew Ng
- **Plataforma:** Coursera
- **Curso Específico:** Sequence Models (Course 5)
- **Tópico:** VAEs em Week 4

**3. Probabilistic Deep Learning (Udacity)**
- **Instrutor:** Jan-Willem van de Meent
- **Link:** https://www.udacity.com/
- **Conteúdo:** VAEs, normalizing flows, GANs

### Vídeos

**1. Ali Ghodsi's Lecture on VAE**
- **Plataforma:** YouTube
- **Link:** https://www.youtube.com/watch?v=uaaqyVS9-rM
- **Duração:** ~1h30min
- **Nível:** Intermediário
- **Descrição:** Derivação matemática completa

**2. Arxiv Insights: VAE**
- **Canal:** Arxiv Insights
- **Link:** https://www.youtube.com/watch?v=9zKuYvjFFS8
- **Duração:** ~15min
- **Nível:** Iniciante
- **Descrição:** Explicação visual e intuitiva

---

## Implementações

### Frameworks e Bibliotecas

**1. PyTorch VAE Collection**
- **Repositório:** https://github.com/AntixK/PyTorch-VAE
- **Autor:** Anand K
- **Modelos:** 20+ variantes de VAE
- **Qualidade:** ⭐⭐⭐⭐⭐
- **Inclui:** VAE, β-VAE, WAE, VQ-VAE, etc.

**2. Disentanglement Library (Google Research)**
- **Repositório:** https://github.com/google-research/disentanglement_lib
- **Autor:** Google Research
- **Recursos:**
  - Múltiplos modelos (VAE, β-VAE, Factor-VAE, etc.)
  - Métricas de disentanglement
  - Datasets sintéticos
- **Qualidade:** ⭐⭐⭐⭐⭐

**3. PyTorch Lightning VAE**
- **Repositório:** https://github.com/PyTorchLightning/lightning-bolts
- **Framework:** PyTorch Lightning
- **Vantagens:** Código limpo, fácil treinar

**4. TensorFlow Probability**
- **Link:** https://www.tensorflow.org/probability
- **Recursos:** Distribuições, camadas probabilísticas
- **Exemplos:** VAE tutorials oficiais

### Nosso Código

**latent-space-tutorial**
- **Repositório:** Este projeto
- **Modelos:** Autoencoder, VAE, Beta-VAE, Annealed Beta-VAE
- **Recursos:**
  - Código educacional bem documentado
  - 6 notebooks progressivos
  - Scripts de exemplo
  - Visualizações extensivas

---

## Datasets

### Para Experimentação

**1. MNIST**
- **Tipo:** Dígitos manuscritos
- **Tamanho:** 70,000 imagens (28×28)
- **Link:** http://yann.lecun.com/exdb/mnist/
- **Uso:** Benchmark padrão para VAEs

**2. Fashion-MNIST**
- **Tipo:** Roupas e acessórios
- **Tamanho:** 70,000 imagens (28×28)
- **Link:** https://github.com/zalandoresearch/fashion-mnist
- **Uso:** Alternativa mais desafiadora ao MNIST

**3. CIFAR-10**
- **Tipo:** Imagens coloridas (10 classes)
- **Tamanho:** 60,000 imagens (32×32×3)
- **Link:** https://www.cs.toronto.edu/~kriz/cifar.html
- **Uso:** Teste em imagens coloridas

**4. CelebA**
- **Tipo:** Faces de celebridades
- **Tamanho:** 202,599 imagens
- **Link:** http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
- **Uso:** Disentanglement de atributos faciais
- **Atributos:** 40 anotações binárias

**5. dSprites**
- **Tipo:** Formas sintéticas 2D
- **Fatores:** Forma, escala, rotação, posição x, posição y
- **Tamanho:** 737,280 imagens (64×64)
- **Link:** https://github.com/deepmind/dsprites-dataset
- **Uso:** Benchmark para disentanglement
- **Vantagem:** Ground truth de fatores conhecidos

**6. 3D Shapes**
- **Tipo:** Objetos 3D renderizados
- **Fatores:** Cor, forma, tamanho, rotação, posição
- **Link:** https://github.com/deepmind/3d-shapes
- **Uso:** Disentanglement 3D

---

## Ferramentas

### Visualização

**1. TensorBoard**
- **Link:** https://www.tensorflow.org/tensorboard
- **Uso:** Logging de métricas, imagens, grafos
- **Compatível:** PyTorch e TensorFlow

**2. Weights & Biases (W&B)**
- **Link:** https://wandb.ai/
- **Recursos:**
  - Tracking de experimentos
  - Comparação de modelos
  - Sweep de hyperparameters

**3. Matplotlib / Seaborn**
- **Links:** https://matplotlib.org/ | https://seaborn.pydata.org/
- **Uso:** Visualizações estáticas
- **Nosso Projeto:** Usado extensivamente

**4. Plotly**
- **Link:** https://plotly.com/python/
- **Uso:** Visualizações interativas
- **Vantagem:** Gráficos 3D interativos

### Desenvolvimento

**1. PyTorch**
- **Link:** https://pytorch.org/
- **Versão:** 2.0+
- **Usado Em:** Este projeto

**2. Jupyter / JupyterLab**
- **Link:** https://jupyter.org/
- **Uso:** Notebooks interativos
- **Nosso Projeto:** 6 notebooks educacionais

**3. Poetry / pip**
- **Gerenciamento:** Dependências
- **Nosso Projeto:** requirements.txt e environment.yml

---

## Recursos Online

### Comunidades

**1. Reddit**
- r/MachineLearning
- r/learnmachinelearning
- r/deeplearning

**2. Stack Overflow**
- Tag: `variational-autoencoder`
- Tag: `pytorch`

**3. Papers With Code**
- **Link:** https://paperswithcode.com/method/vae
- **Recursos:** Papers + código + benchmarks

### Blogs Técnicos

**1. Distill.pub**
- **Link:** https://distill.pub/
- **Artigos:** Visualizações interativas de ML
- **Qualidade:** Excepcional

**2. OpenAI Blog**
- **Link:** https://openai.com/blog/
- **Artigos:** Pesquisa de ponta

**3. Google AI Blog**
- **Link:** https://ai.googleblog.com/
- **Artigos:** Pesquisa e aplicações

---

## Papers Complementares

### Teoria

**Information Theory**
- Cover, T. M., & Thomas, J. A. (2006). *Elements of information theory*. Wiley.

**Variational Inference**
- Blei, D. M., Kucukelbir, A., & McAuliffe, J. D. (2017). Variational inference: A review for statisticians. *JASA*.

### Aplicações Específicas

**Molecular Generation**
- Gómez-Bombarelli, R., et al. (2018). Automatic chemical design using a data-driven continuous representation of molecules. *ACS*.

**Music Generation**
- Roberts, A., et al. (2018). A hierarchical latent vector model for learning long-term structure in music. *ICML*.

**Text VAE**
- Bowman, S. R., et al. (2016). Generating sentences from a continuous space. *CoNLL*.

---

## Como Citar Este Projeto

### BibTeX

```bibtex
@misc{latentspace2024,
  author = {Professora Itamar},
  title = {Latent Space Tutorial: Autoencoders e VAEs Educacional},
  year = {2025},
  publisher = {UTFPR},
  howpublished = {\url{https://github.com/your-repo/latent-space-tutorial}},
  note = {Tutorial educacional sobre espaços latentes}
}
```

### APA

```
Itamar. (2025). Latent Space Tutorial: Autoencoders e VAEs Educacional.
UTFPR. https://github.com/your-repo/latent-space-tutorial
```

---

## Atualização

Este documento é atualizado regularmente. Última atualização: **Novembro 2025**.

Para sugerir adições, abra uma issue no repositório.
