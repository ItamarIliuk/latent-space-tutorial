# Data Directory

Este diretório contém os datasets usados no tutorial.

## MNIST

O dataset MNIST será automaticamente baixado aqui quando você executar os notebooks ou scripts de treinamento pela primeira vez.

Estrutura após o download:
```
data/
└── MNIST/
    ├── raw/
    │   ├── train-images-idx3-ubyte
    │   ├── train-labels-idx1-ubyte
    │   ├── t10k-images-idx3-ubyte
    │   └── t10k-labels-idx1-ubyte
    └── processed/
        ├── training.pt
        └── test.pt
```

## Outros Datasets

Se você quiser experimentar com outros datasets, adicione-os aqui e atualize os data loaders em `src/utils/data_loader.py`.

## Nota

Os arquivos de dados não são commitados no Git (veja `.gitignore`). Cada usuário deve baixar os dados localmente.
