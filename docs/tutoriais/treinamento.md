# Guia de Treinamento

## Quick Start

```bash
# Treinar VAE rapidamente
python examples/quick_start.py

# Treinar com configurações customizadas
python examples/train_vae.py --latent-dim 10 --epochs 30 --beta 1.0 --save-model
```

## Configurações Recomendadas

### MNIST (Baseline)
- **latent_dim:** 2 (visualização) ou 10 (performance)
- **epochs:** 20-30
- **batch_size:** 128
- **learning_rate:** 1e-3
- **beta:** 1.0 (VAE padrão)

### Experimentação
- **Disentanglement:** beta=4.0-10.0
- **Compressão:** beta=0.5-1.0
- **Geração:** beta=1.0-2.0

## Ver Documentação Completa

Para tutoriais completos, consulte os notebooks em `notebooks/`.

Para API detalhada, veja `api_reference.md`.
