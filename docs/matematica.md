# Fundamentos Matemáticos

## Índice
- [Notação](#notação)
- [Autoencoder](#autoencoder-matemática)
- [VAE: Fundamentos Probabilísticos](#vae-fundamentos-probabilísticos)
- [ELBO: Evidence Lower Bound](#elbo-evidence-lower-bound)
- [KL Divergence](#kl-divergence)
- [Reparameterization Trick](#reparameterization-trick-detalhado)
- [Beta-VAE](#beta-vae-matemática)
- [Derivações](#derivações)

---

## Notação

| Símbolo | Descrição |
|---------|-----------|
| x | Dado observado (imagem) |
| z | Variável latente (código) |
| θ | Parâmetros do decoder |
| φ | Parâmetros do encoder |
| p(x) | Distribuição dos dados |
| p(z) | Prior sobre z |
| p_θ(x\|z) | Likelihood (decoder) |
| q_φ(z\|x) | Distribuição aproximada (encoder) |
| μ | Média da distribuição latente |
| σ² | Variância da distribuição latente |
| N(μ, σ²) | Distribuição Gaussiana |
| KL(·\|\|·) | Divergência de Kullback-Leibler |
| E[·] | Valor esperado |

---

## Autoencoder: Matemática

### Formulação Geral

**Objetivo:** Aprender função que minimiza erro de reconstrução.

```
min_θ,φ L(x, x') = min_θ,φ ||x - f_θ(g_φ(x))||²
```

Onde:
- g_φ: **Encoder** (x → z)
- f_θ: **Decoder** (z → x')

### Funções de Perda

#### 1. Mean Squared Error (MSE)

```
L_MSE(x, x') = (1/n) Σᵢ₌₁ⁿ (xᵢ - x'ᵢ)²
```

**Uso:** Imagens com valores contínuos.

**Propriedades:**
- Diferenciável
- Penaliza erros grandes quadraticamente
- Equivalente a assumir ruído Gaussiano

#### 2. Binary Cross-Entropy (BCE)

```
L_BCE(x, x') = -Σᵢ₌₁ⁿ [xᵢ log(x'ᵢ) + (1-xᵢ) log(1-x'ᵢ)]
```

**Uso:** Imagens binárias ou normalizadas em [0,1].

**Propriedades:**
- Assume distribuição Bernoulli
- Melhor para dados binários
- Gradientes mais estáveis

### Compressão

**Taxa de compressão:**
```
r = dim(x) / dim(z)
```

Exemplo MNIST:
```
r = 784 / 2 = 392x
```

---

## VAE: Fundamentos Probabilísticos

### Modelo Generativo

O VAE assume o seguinte processo gerador:

```
z ~ p(z) = N(0, I)           (1) Prior
x|z ~ p_θ(x|z)               (2) Likelihood (decoder)
```

**Objetivo:** Maximizar a log-likelihood marginal:

```
log p_θ(x) = log ∫ p_θ(x|z) p(z) dz
```

❌ **Problema:** Integral intratável!

### Inference Aproximada

Introduzimos distribuição aproximada q_φ(z|x) (encoder):

```
q_φ(z|x) = N(z; μ_φ(x), σ²_φ(x))
```

Onde μ_φ e σ²_φ são outputs de redes neurais.

---

## ELBO: Evidence Lower Bound

### Derivação

Começando com log p(x):

```
log p_θ(x) = log ∫ p_θ(x|z) p(z) dz

           = log ∫ p_θ(x|z) p(z) · [q_φ(z|x) / q_φ(z|x)] dz

           = log E_q[p_θ(x|z) p(z) / q_φ(z|x)]

           ≥ E_q[log p_θ(x|z)] - KL(q_φ(z|x) || p(z))
```

**ELBO (Evidence Lower BOund):**

```
L(θ, φ; x) = E_q_φ(z|x)[log p_θ(x|z)] - KL(q_φ(z|x) || p(z))
              └─────────┬──────────┘    └─────────┬────────┘
                 Reconstruction            Regularization
```

### Interpretação

**Termo 1: Reconstruction Loss**
- Quão bem o decoder reconstrói x a partir de z
- Maximizar ⟺ Minimizar erro

**Termo 2: KL Divergence**
- Quão diferente q(z|x) é de p(z)
- Regulariza o espaço latente
- Força z a seguir prior N(0,I)

### Objetivo do Treinamento

```
max_θ,φ E_p_data(x)[L(θ, φ; x)]
```

Ou equivalentemente (minimização):

```
min_θ,φ -E_p_data(x)[L(θ, φ; x)]
     = min_θ,φ E_p_data(x)[-E_q[log p_θ(x|z)] + KL(q||p)]
```

---

## KL Divergence

### Definição Geral

```
KL(P || Q) = E_P[log(P(x)/Q(x))]
           = ∫ P(x) log(P(x)/Q(x)) dx
```

**Propriedades:**
- KL(P||Q) ≥ 0 (sempre não-negativo)
- KL(P||Q) = 0 ⟺ P = Q
- Não simétrico: KL(P||Q) ≠ KL(Q||P)

### KL entre Gaussianas

Para VAE, precisamos calcular:

```
KL(q_φ(z|x) || p(z))
```

Onde:
- q_φ(z|x) = N(μ, σ²) (aprendido)
- p(z) = N(0, I) (prior fixo)

**Fórmula Fechada:**

Para duas Gaussianas univariadas N(μ₁, σ₁²) e N(μ₂, σ₂²):

```
KL(N(μ₁, σ₁²) || N(μ₂, σ₂²)) = log(σ₂/σ₁) + (σ₁² + (μ₁-μ₂)²)/(2σ₂²) - 1/2
```

Para nosso caso (μ₂=0, σ₂=1):

```
KL(N(μ, σ²) || N(0, 1)) = -1/2 · (1 + log(σ²) - μ² - σ²)
```

**Para dimensão d:**

```
KL(q || p) = -1/2 Σᵢ₌₁ᵈ (1 + log(σᵢ²) - μᵢ² - σᵢ²)
```

### Implementação

```python
def kl_divergence(mu, logvar):
    """
    KL(N(μ, σ²) || N(0, 1))

    Args:
        mu: [batch, latent_dim]
        logvar: log(σ²) [batch, latent_dim]

    Returns:
        kl: [batch] - KL para cada amostra
    """
    # KL = -0.5 * sum(1 + log(σ²) - μ² - σ²)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    return kl
```

**Por que log(σ²)?**
- Estabilidade numérica
- σ² sempre positivo: exp(logvar) > 0
- Evita problemas com σ² → 0

---

## Reparameterization Trick: Detalhado

### Problema

Queremos calcular:

```
∇_φ E_q_φ(z|x)[f(z)]
```

Mas amostragem z ~ q_φ(z|x) não é diferenciável!

### Solução Matemática

**Ideia:** Separar ruído estocástico dos parâmetros.

**Antes:**
```
z ~ N(μ_φ(x), σ²_φ(x))
```

**Depois:**
```
ε ~ N(0, I)                    (ruído externo)
z = μ_φ(x) + σ_φ(x) ⊙ ε       (determinístico dado ε)
```

Onde ⊙ é produto elemento a elemento.

### Derivação do Gradiente

Queremos calcular:

```
∇_φ E_q_φ(z|x)[f(z)]
```

**Com reparametrização:**

```
∇_φ E_q_φ(z|x)[f(z)] = ∇_φ E_ε~N(0,I)[f(μ_φ(x) + σ_φ(x) ⊙ ε)]

                       = E_ε~N(0,I)[∇_φ f(μ_φ(x) + σ_φ(x) ⊙ ε)]
```

✅ **Agora podemos amostrar ε e calcular gradientes!**

### Estimador Monte Carlo

```
∇_φ E_q[f(z)] ≈ (1/L) Σₗ₌₁ᴸ ∇_φ f(zₗ)
```

Onde zₗ = μ_φ(x) + σ_φ(x) ⊙ εₗ, com εₗ ~ N(0,I)

**Na prática, L=1** (uma amostra por forward pass).

### Gradientes Específicos

```
∂z/∂μ = I                      (identidade)
∂z/∂σ = ε                      (ruído amostrado)
```

Ambos bem definidos e computáveis! ✅

---

## Beta-VAE: Matemática

### Formulação

```
L_β(θ, φ; x) = E_q_φ(z|x)[log p_θ(x|z)] - β · KL(q_φ(z|x) || p(z))
```

Onde β ∈ ℝ⁺ é hiperparâmetro.

### Análise Teórica

**β = 1:** ELBO padrão (correto bayesiano)

**β > 1:** Maior peso na regularização
- Força maior independência entre dimensões latentes
- Pressão para usar prior N(0,I)
- Trade-off: pior reconstrução

**β < 1:** Menor regularização
- Melhor reconstrução
- Espaço latente menos regularizado
- Pode degenerar

### Decomposição da KL (β-TCVAE)

KL pode ser decomposta em:

```
KL(q(z|x) || p(z)) = I(z; x) + KL(q(z) || Πᵢq(zᵢ)) + Σᵢ KL(q(zᵢ) || p(zᵢ))
                      └──┬──┘   └────────┬───────┘   └──────────┬─────────┘
                      Index        Total              Dimension-wise
                     Coding      Correlation            KL
```

Beta-VAE implicitamente penaliza Total Correlation (termo do meio).

### Disentanglement Score

**Mutual Information Gap (MIG):**

```
MIG = (1/K) Σₖ [I(zⱼ₍₁₎; vₖ) - I(zⱼ₍₂₎; vₖ)] / H(vₖ)
```

Onde:
- vₖ: k-ésimo fator de variação verdadeiro
- j(1), j(2): dimensões latentes mais correlacionadas
- H(vₖ): entropia de vₖ

**Interpretação:** Diferença entre informação mútua da melhor e segunda melhor dimensão.

---

## Derivações

### Derivação Completa do ELBO

**Partindo de log p(x):**

```
log p_θ(x) = log ∫ p_θ(x,z) dz

           = log ∫ p_θ(x,z) · [q_φ(z|x) / q_φ(z|x)] dz

           = log E_q_φ(z|x)[p_θ(x,z) / q_φ(z|x)]
```

**Aplicando desigualdade de Jensen:**

```
log E_q[f(z)] ≥ E_q[log f(z)]    (Jensen, log é côncavo)
```

Portanto:

```
log p_θ(x) ≥ E_q_φ(z|x)[log p_θ(x,z) / q_φ(z|x)]

           = E_q[log p_θ(x,z)] - E_q[log q_φ(z|x)]

           = E_q[log p_θ(x|z) + log p(z)] - E_q[log q_φ(z|x)]

           = E_q[log p_θ(x|z)] + E_q[log p(z)] - E_q[log q_φ(z|x)]

           = E_q[log p_θ(x|z)] - [E_q[log q_φ(z|x)] - E_q[log p(z)]]

           = E_q[log p_θ(x|z)] - KL(q_φ(z|x) || p(z))
```

✅ **ELBO derivado!**

### Gap entre log p(x) e ELBO

```
log p_θ(x) - ELBO = KL(q_φ(z|x) || p_θ(z|x))
```

**Interpretação:**
- Gap = quão boa é nossa aproximação q
- Gap = 0 ⟺ q_φ(z|x) = p_θ(z|x) (posterior verdadeiro)
- Maximizar ELBO ≈ minimizar gap

### Gradiente da ELBO

**Em relação a φ (encoder):**

```
∇_φ ELBO = ∇_φ E_q_φ(z|x)[log p_θ(x|z)] - ∇_φ KL(q_φ(z|x) || p(z))
```

Primeiro termo (com reparametrização):

```
∇_φ E_ε[log p_θ(x|z(ε))] ≈ ∇_φ log p_θ(x|z)    (z = μ + σ⊙ε, ε~N(0,I))
```

Segundo termo (KL analítico):

```
∇_φ KL = -1/2 ∇_φ Σ(1 + log σ² - μ² - σ²)
```

**Em relação a θ (decoder):**

```
∇_θ ELBO = ∇_θ E_q_φ(z|x)[log p_θ(x|z)]

         ≈ ∇_θ log p_θ(x|z)    (z amostrado)
```

---

## Aproximações Comuns

### Reconstruction Loss

**Assumindo p_θ(x|z) = N(x; f_θ(z), σ²I):**

```
log p_θ(x|z) = -||x - f_θ(z)||² / (2σ²) + const
```

Maximizar ⟺ Minimizar MSE

**Assumindo p_θ(x|z) = Πᵢ Bernoulli(xᵢ; f_θ(z)ᵢ):**

```
log p_θ(x|z) = Σᵢ [xᵢ log f_θ(z)ᵢ + (1-xᵢ) log(1-f_θ(z)ᵢ)]
```

Maximizar ⟺ Minimizar BCE

---

## Identidades Úteis

### Gaussiana

```
N(x; μ, σ²) = (1/√(2πσ²)) exp(-(x-μ)²/(2σ²))

log N(x; μ, σ²) = -1/2 log(2πσ²) - (x-μ)²/(2σ²)
```

### KL entre Gaussianas

```
KL(N(μ₁, Σ₁) || N(μ₂, Σ₂)) = 1/2 [log|Σ₂|/|Σ₁| - d + tr(Σ₂⁻¹Σ₁) + (μ₂-μ₁)ᵀΣ₂⁻¹(μ₂-μ₁)]
```

Para diagonal Σ₁ = diag(σ₁²), Σ₂ = I:

```
KL = 1/2 Σᵢ[log(1/σᵢ²) + σᵢ² + μᵢ² - 1]
   = -1/2 Σᵢ[1 + log σᵢ² - μᵢ² - σᵢ²]
```

---

## Referências Matemáticas

1. **VAE Paper Original:**
   - Kingma, D. P., & Welling, M. (2013). Auto-encoding variational bayes.

2. **Tutorial Detalhado:**
   - Doersch, C. (2016). Tutorial on variational autoencoders.

3. **β-VAE:**
   - Higgins, I., et al. (2017). β-VAE: Learning basic visual concepts with a constrained variational framework.

4. **β-TCVAE:**
   - Chen, R. T., et al. (2018). Isolating sources of disentanglement in variational autoencoders.

Para conceitos intuitivos, veja [conceitos.md](conceitos.md).
