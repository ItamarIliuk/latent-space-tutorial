# Conceitos Fundamentais

## Índice
- [Espaço Latente](#espaço-latente)
- [Autoencoders](#autoencoders)
- [Variational Autoencoders (VAE)](#variational-autoencoders-vae)
- [Beta-VAE](#beta-vae)
- [Disentanglement](#disentanglement)
- [Reparameterization Trick](#reparameterization-trick)

---

## Espaço Latente

### Definição

Um **espaço latente** (latent space) é uma representação compacta e estruturada de dados de alta dimensionalidade. É uma transformação que mapeia dados complexos para um espaço de menor dimensão, preservando as características mais importantes.

### Características

1. **Dimensionalidade Reduzida**
   - Original: 784 dimensões (28×28 pixels)
   - Latente: 2-50 dimensões tipicamente
   - Compressão de informação

2. **Estrutura Semântica**
   - Pontos próximos representam dados similares
   - Continuidade: transições suaves
   - Navegabilidade: interpolações válidas

3. **Interpretabilidade**
   - Dimensões podem ter significado específico
   - Permite manipulação controlada
   - Facilita análise e visualização

### Propriedades Desejáveis

| Propriedade | Descrição | Importância |
|------------|-----------|-------------|
| **Compressão** | Representar dados com menos dimensões | Alta |
| **Continuidade** | Pequenas mudanças → pequenas variações | Alta |
| **Completude** | Todo ponto válido gera dado válido | Média |
| **Disentanglement** | Dimensões independentes | Alta (VAE) |
| **Suavidade** | Gradientes bem definidos | Alta |

### Aplicações

- **Compressão de Dados**: Armazenamento eficiente
- **Geração de Dados**: Criar novas amostras
- **Transferência de Estilo**: Manipular características
- **Detecção de Anomalias**: Identificar outliers
- **Visualização**: Entender estrutura dos dados

---

## Autoencoders

### Arquitetura

Um Autoencoder é uma rede neural composta por duas partes:

```
Input → [Encoder] → Latent Code (z) → [Decoder] → Reconstruction
```

#### Encoder
- **Função**: Comprimir entrada para espaço latente
- **Arquitetura**: Camadas fully-connected com ativação ReLU
- **Output**: Vetor latente z de dimensão fixa

#### Decoder
- **Função**: Reconstruir entrada a partir do código latente
- **Arquitetura**: Espelho do encoder
- **Output**: Reconstrução da entrada original

### Função de Perda

**Mean Squared Error (MSE):**

```
L(x, x') = (1/n) Σ(x - x')²
```

Onde:
- x = entrada original
- x' = reconstrução
- n = número de pixels

**Binary Cross-Entropy (BCE):**

```
L(x, x') = -Σ[x log(x') + (1-x) log(1-x')]
```

Usado quando pixels estão normalizados em [0,1].

### Tipos de Autoencoders

1. **Vanilla Autoencoder**
   - Arquitetura básica
   - Mapeamento determinístico
   - Foco em reconstrução

2. **Sparse Autoencoder**
   - Regularização L1 no código latente
   - Força esparsidade
   - Representações mais interpretáveis

3. **Denoising Autoencoder**
   - Treinado com entrada ruidosa
   - Aprende características robustas
   - Melhor generalização

4. **Contractive Autoencoder**
   - Penaliza variação do código latente
   - Mais robusto a perturbações
   - Representações mais estáveis

### Limitações

❌ **Espaço latente pode ter "buracos"**
- Alguns pontos não geram dados válidos
- Interpolações podem falhar
- Difícil gerar novas amostras

❌ **Overfitting**
- Pode memorizar dados de treino
- Reconstrução perfeita ≠ boa representação

❌ **Falta de estrutura probabilística**
- Não modela incerteza
- Mapeamento determinístico

---

## Variational Autoencoders (VAE)

### Diferença Fundamental

**Autoencoder:**
```
x → Encoder → z (ponto fixo) → Decoder → x'
```

**VAE:**
```
x → Encoder → (μ, σ²) → z ~ N(μ, σ²) → Decoder → x'
```

### Componentes

#### 1. Encoder Probabilístico

O encoder aprende **parâmetros de uma distribuição**:
- μ (mu): Média
- σ² (sigma²): Variância

```python
mu = fc_mu(h)
logvar = fc_logvar(h)  # log(σ²) para estabilidade numérica
```

#### 2. Reparameterization Trick

**Problema:** Amostragem não é diferenciável.

**Solução:** Reparametrizar:
```
z = μ + ε·σ, onde ε ~ N(0, 1)
```

Isso permite backpropagation através da operação de amostragem.

#### 3. Decoder Probabilístico

Gera reconstrução a partir de z amostrado.

### Função de Perda (ELBO)

A perda do VAE combina dois termos:

```
L = Reconstruction Loss + β·KL Divergence
```

#### Reconstruction Loss
Quão bem o modelo reconstrói a entrada:
```
L_recon = BCE(x, x')  ou  MSE(x, x')
```

#### KL Divergence
Força z a seguir distribuição prior N(0, I):

```
L_KL = -0.5 Σ(1 + log(σ²) - μ² - σ²)
```

**Interpretação:**
- Regulariza o espaço latente
- Previne collapse (todas amostras → mesmo z)
- Garante que prior p(z) = N(0,1) funcione

### Vantagens sobre Autoencoder

✅ **Espaço latente contínuo e estruturado**
✅ **Geração de novas amostras**: Amostra z ~ N(0,1) e decodifica
✅ **Interpolações suaves**: Todo ponto é válido
✅ **Modelagem probabilística**: Captura incerteza
✅ **Regularização natural**: KL divergence previne overfitting

### Hyperparâmetros

| Parâmetro | Descrição | Valores Típicos |
|-----------|-----------|-----------------|
| `latent_dim` | Dimensão do espaço latente | 2-100 |
| `beta` | Peso da KL divergence | 0.5-10.0 |
| `learning_rate` | Taxa de aprendizado | 1e-4 a 1e-3 |
| `hidden_dims` | Dimensões das camadas ocultas | [512, 256], [256, 128] |

---

## Beta-VAE

### Motivação

VAE padrão (β=1) equilibra reconstrução e regularização. **Beta-VAE** adiciona controle sobre esse trade-off.

### Formulação

```
L = Reconstruction Loss + β·KL Divergence
```

**Efeito do β:**

| β | Reconstrução | Regularização | Disentanglement | Uso |
|---|-------------|---------------|----------------|-----|
| **< 1** | ⭐⭐⭐ | ⭐ | ⭐ | Compressão |
| **= 1** | ⭐⭐ | ⭐⭐ | ⭐⭐ | VAE padrão |
| **> 1** | ⭐ | ⭐⭐⭐ | ⭐⭐⭐ | Disentanglement |

### Beta-VAE vs VAE

**β < 1 (0.5 típico):**
- Melhor qualidade de reconstrução
- Mais "liberdade" no espaço latente
- Menos regularização
- Uso: Quando qualidade visual é crítica

**β = 1 (VAE padrão):**
- Balanceamento teórico ideal
- ELBO correto
- Geração balanceada

**β > 1 (2-10 típico):**
- Forte regularização
- Espaço latente mais "organizado"
- Dimensões mais independentes (disentangled)
- Pior reconstrução
- Uso: Quando interpretabilidade é crítica

### Annealed Beta-VAE

Aumenta β gradualmente durante treinamento:

```
β(epoch) = min(max_β, β_inicial + (max_β - β_inicial)·epoch/anneal_steps)
```

**Vantagens:**
- Começa focando em reconstrução (β baixo)
- Gradualmente adiciona regularização
- Evita collapse do espaço latente
- Melhor convergência

**Implementação:**
```python
class AnnealedBetaVAE:
    def __init__(self, max_beta=4.0, anneal_steps=10):
        self.max_beta = max_beta
        self.anneal_steps = anneal_steps
        self.current_step = 0

    def get_beta(self):
        return min(self.max_beta,
                   self.current_step / self.anneal_steps * self.max_beta)
```

---

## Disentanglement

### Definição

**Disentanglement** = Dimensões latentes independentes, cada uma controlando um fator de variação específico.

### Exemplo Visual (Dataset de Faces)

```
z[0] → Cor do cabelo (loiro ← → moreno)
z[1] → Óculos (sem ← → com)
z[2] → Sorriso (sério ← → sorrindo)
z[3] → Idade (jovem ← → idoso)
```

Mudando **apenas z[0]**, só a cor do cabelo muda.

### Importância

✅ **Interpretabilidade**: Entender o que o modelo aprendeu
✅ **Controle**: Manipular características específicas
✅ **Transferência**: Reutilizar representações
✅ **Debugging**: Identificar problemas

### Como Medir

**1. Variance of Informativeness (SAP Score)**
- Mede quão informativa cada dimensão é para cada fator
- Score alto = bom disentanglement

**2. Mutual Information Gap (MIG)**
- Diferença entre informação mútua da top-1 e top-2 dimensões
- Score alto = dimensões especializadas

**3. Qualitativo**
- Variar uma dimensão por vez
- Observar mudanças visuais
- Verificar independência

### Como Alcançar

1. **Beta-VAE (β > 1)**
   - Método mais comum
   - β = 4-10 típico

2. **Factor-VAE**
   - Adiciona discriminador
   - Penaliza correlações

3. **β-TCVAE**
   - Decompõe KL divergence
   - Foca em independência

4. **Arquitetura**
   - Mais capacidade no encoder
   - Batch normalization
   - Dropout seletivo

---

## Reparameterization Trick

### Problema

Queremos treinar: x → Encoder → **z ~ N(μ, σ²)** → Decoder → x'

❌ **Amostragem não é diferenciável!**

Não podemos fazer backpropagation através de operação estocástica.

### Solução: Reparameterização

**Antes (não diferenciável):**
```python
z = sample_from(N(μ, σ²))  # ❌ Não permite gradientes
```

**Depois (diferenciável):**
```python
ε = sample_from(N(0, 1))   # Ruído externo
z = μ + ε·σ                 # ✅ Diferenciável!
```

### Implementação

```python
def reparameterize(mu, logvar):
    """
    Reparameterization trick: z = μ + ε·σ

    Args:
        mu: Média da distribuição [batch_size, latent_dim]
        logvar: Log-variância [batch_size, latent_dim]

    Returns:
        z: Amostra latente [batch_size, latent_dim]
    """
    std = torch.exp(0.5 * logvar)  # σ = exp(0.5·log(σ²))
    eps = torch.randn_like(std)     # ε ~ N(0, 1)
    z = mu + eps * std              # z = μ + ε·σ
    return z
```

### Por que funciona?

1. **μ e σ são diferenciáveis** (output do encoder)
2. **ε é constante** (ruído externo, não depende de parâmetros)
3. **z = μ + ε·σ é diferenciável** em relação a μ e σ

**Gradientes:**
```
∂z/∂μ = 1
∂z/∂σ = ε
```

Ambos bem definidos! ✅

### Variantes

**1. Log-variance (padrão):**
```python
logvar = fc_logvar(h)
std = torch.exp(0.5 * logvar)
```
✅ Estabilidade numérica (σ sempre positivo)

**2. Softplus:**
```python
std = F.softplus(fc_std(h))
```
✅ Garante positividade

**3. Direct std:**
```python
std = fc_std(h)
```
❌ Pode gerar σ negativo

---

## Comparação: Autoencoder vs VAE vs Beta-VAE

| Aspecto | Autoencoder | VAE | Beta-VAE |
|---------|------------|-----|----------|
| **Encoder Output** | z (ponto) | (μ, σ²) | (μ, σ²) |
| **Latente** | Determinístico | Probabilístico | Probabilístico |
| **Loss** | MSE/BCE | BCE + KL | BCE + β·KL |
| **Geração** | Difícil | ✅ Fácil | ✅ Fácil |
| **Interpolação** | ⚠️ Pode falhar | ✅ Suave | ✅ Suave |
| **Disentanglement** | ❌ Baixo | ⚠️ Médio | ✅ Alto (β>1) |
| **Reconstrução** | ✅ Melhor | ⚠️ Boa | ❌ Pior (β alto) |
| **Uso Principal** | Compressão | Geração | Interpretabilidade |

---

## Referências Rápidas

- **Paper Original VAE**: Kingma & Welling (2013)
- **Beta-VAE**: Higgins et al. (2017)
- **Reparameterization Trick**: Kingma & Welling (2013)
- **Tutorial**: Carl Doersch (2016)

Para detalhes matemáticos, veja [matematica.md](matematica.md).
Para referências completas, veja [referencias.md](referencias.md).
