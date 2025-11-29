#!/usr/bin/env python3
"""
Script de Verificação do Repositório
Execute este script para verificar se tudo está pronto para upload
"""

import os
from pathlib import Path
import sys

def check_file_exists(filepath, description):
    """Verifica se um arquivo existe"""
    if os.path.exists(filepath):
        size = os.path.getsize(filepath)
        print(f"✅ {description}: {size} bytes")
        return True
    else:
        print(f"❌ {description}: NÃO ENCONTRADO")
        return False

def check_directory_exists(dirpath, description):
    """Verifica se um diretório existe"""
    if os.path.isdir(dirpath):
        count = len(os.listdir(dirpath))
        print(f"✅ {description}: {count} itens")
        return True
    else:
        print(f"❌ {description}: NÃO ENCONTRADO")
        return False

def main():
    print("=" * 60)
    print("🔍 VERIFICAÇÃO DO REPOSITÓRIO LATENT SPACE TUTORIAL")
    print("=" * 60)
    print()
    
    base_dir = Path(r"C:\Users\Itama\Documents\latent-space-tutorial")
    
    if not os.path.exists(base_dir):
        print(f"❌ Diretório base não encontrado: {base_dir}")
        sys.exit(1)
    
    print(f"📁 Diretório base: {base_dir}")
    print()
    
    # Contador de sucessos
    checks_passed = 0
    total_checks = 0
    
    # Verificar arquivos principais
    print("📄 Arquivos de Configuração:")
    print("-" * 60)
    
    files_to_check = [
        ("README.md", "README principal"),
        ("LICENSE", "Licença MIT"),
        ("requirements.txt", "Dependências Python"),
        ("environment.yml", "Ambiente Conda"),
        ("setup.py", "Setup do pacote"),
        (".gitignore", "Git ignore"),
        ("CONTRIBUTING.md", "Guia de contribuição"),
        ("SUMARIO.md", "Sumário"),
    ]
    
    for filename, description in files_to_check:
        filepath = base_dir / filename
        total_checks += 1
        if check_file_exists(filepath, description):
            checks_passed += 1
    
    print()
    print("🐍 Código Python:")
    print("-" * 60)
    
    python_files = [
        ("src/__init__.py", "Módulo principal"),
        ("src/models/__init__.py", "Módulo models"),
        ("src/models/autoencoder.py", "Autoencoder"),
        ("src/models/vae.py", "VAE"),
        ("src/models/beta_vae.py", "Beta-VAE"),
        ("src/utils/__init__.py", "Módulo utils"),
        ("src/utils/data_loader.py", "Data Loader"),
        ("src/experiments/__init__.py", "Módulo experiments"),
    ]
    
    for filename, description in python_files:
        filepath = base_dir / filename
        total_checks += 1
        if check_file_exists(filepath, description):
            checks_passed += 1
    
    print()
    print("📂 Diretórios:")
    print("-" * 60)
    
    directories = [
        ("src", "Código fonte"),
        ("src/models", "Modelos"),
        ("src/utils", "Utilitários"),
        ("src/experiments", "Experimentos"),
        ("notebooks", "Notebooks Jupyter"),
        ("docs", "Documentação"),
        ("examples", "Exemplos"),
        ("tests", "Testes"),
        ("data", "Dados"),
    ]
    
    for dirname, description in directories:
        dirpath = base_dir / dirname
        total_checks += 1
        if check_directory_exists(dirpath, description):
            checks_passed += 1
    
    print()
    print("=" * 60)
    print(f"📊 RESULTADO: {checks_passed}/{total_checks} verificações passaram")
    print("=" * 60)
    print()
    
    if checks_passed == total_checks:
        print("✅ PERFEITO! Tudo está pronto para upload!")
        print()
        print("🚀 Próximos passos:")
        print("1. Execute os comandos git")
        print("2. Faça upload para o GitHub")
        print()
        return 0
    else:
        missing = total_checks - checks_passed
        print(f"⚠️  ATENÇÃO: {missing} itens não encontrados")
        print()
        print("💡 Verifique os itens marcados com ❌ acima")
        print()
        return 1

if __name__ == "__main__":
    sys.exit(main())
