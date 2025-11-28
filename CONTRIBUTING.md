# Contribuindo para o Latent Space Tutorial

Obrigada por considerar contribuir para este projeto! 🎉

## Como Contribuir

### Reportando Bugs

Se você encontrar um bug, por favor abra uma issue incluindo:

- Descrição clara do problema
- Passos para reproduzir
- Comportamento esperado vs comportamento atual
- Versão do Python e das dependências
- Sistema operacional

### Sugerindo Melhorias

Sugestões são bem-vindas! Abra uma issue com:

- Descrição clara da melhoria
- Justificativa (por que seria útil?)
- Exemplos de uso, se aplicável

### Pull Requests

1. Fork o projeto
2. Crie uma branch (`git checkout -b feature/MinhaFeature`)
3. Faça commit das mudanças (`git commit -m 'Add: MinhaFeature'`)
4. Push para a branch (`git push origin feature/MinhaFeature`)
5. Abra um Pull Request

### Padrões de Código

- Siga o PEP 8
- Use type hints quando possível
- Adicione docstrings para funções e classes
- Mantenha funções pequenas e focadas
- Adicione testes para novas funcionalidades

### Mensagens de Commit

Use prefixos claros:
- `Add:` para novas features
- `Fix:` para correções de bugs
- `Docs:` para documentação
- `Refactor:` para refatorações
- `Test:` para testes

## Desenvolvimento Local

```bash
# Clone seu fork
git clone https://github.com/SEU_USUARIO/latent-space-tutorial.git

# Instale em modo desenvolvimento
pip install -e ".[dev]"

# Execute os testes
pytest tests/

# Verifique o estilo
black src/ tests/
flake8 src/ tests/
```

## Código de Conduta

Este projeto segue o [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/).

Seja respeitoso e inclusivo com todos os contribuidores.

## Licença

Ao contribuir, você concorda que suas contribuições serão licenciadas sob a Licença MIT.
