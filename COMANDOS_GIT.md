# 🎯 COMANDOS GIT - COPIE E COLE

## Passo 1: Abrir Terminal
Abra o CMD ou PowerShell e navegue até a pasta:

```bash
cd C:\Users\Itama\Documents\latent-space-tutorial
```

## Passo 2: Inicializar Git

```bash
git init
```

## Passo 3: Configurar Git (se necessário)

```bash
git config user.name "Profa. Itamar"
git config user.email "seu.email@utfpr.edu.br"
```

## Passo 4: Adicionar Arquivos

```bash
git add .
```

## Passo 5: Fazer Commit

```bash
git commit -m "Initial commit: Latent Space Tutorial - Autoencoders e VAEs para ensino"
```

## Passo 6: Preparar Branch Main

```bash
git branch -M main
```

## Passo 7: Criar Repositório no GitHub
⚠️ **IMPORTANTE**: Agora vá ao GitHub!

1. Acesse: https://github.com/new
2. Nome: `latent-space-tutorial`
3. Descrição: `Tutorial completo sobre Espaço Latente, Autoencoders e VAEs`
4. Visibilidade: Public
5. ⚠️ **NÃO marque**: Add a README file
6. ⚠️ **NÃO marque**: Add .gitignore
7. ⚠️ **NÃO marque**: Choose a license
8. Clique em: **Create repository**

## Passo 8: Conectar Repositório Local ao GitHub

```bash
git remote add origin https://github.com/itamar15/latent-space-tutorial.git
```

## Passo 9: Fazer Push (Upload)

```bash
git push -u origin main
```

## ✅ PRONTO!

Seu repositório está online em:
**https://github.com/itamar15/latent-space-tutorial**

---

## 🔄 Comandos para Futuras Atualizações

Quando adicionar novos arquivos:

```bash
# Adicionar arquivos
git add .

# Fazer commit
git commit -m "Add: descrição da mudança"

# Enviar para GitHub
git push
```

---

## 🆘 Resolução de Problemas

### Se der erro de autenticação:
1. Configure um Personal Access Token no GitHub
2. Use o token como senha

### Se der erro "remote origin already exists":
```bash
git remote remove origin
git remote add origin https://github.com/itamar15/latent-space-tutorial.git
```

### Para ver o status:
```bash
git status
```

### Para ver o histórico:
```bash
git log --oneline
```

### Para ver os remotes:
```bash
git remote -v
```

---

## 📋 Checklist Final

Antes de fazer push, verifique:

- [ ] Todos os arquivos estão adicionados (`git status`)
- [ ] Commit foi feito (`git log`)
- [ ] Repositório foi criado no GitHub
- [ ] Remote foi configurado (`git remote -v`)
- [ ] Pronto para push!

---

## 🎓 Compartilhar com Alunos

Após o push, seus alunos podem clonar:

```bash
git clone https://github.com/itamar15/latent-space-tutorial.git
cd latent-space-tutorial
pip install -r requirements.txt
```

---

💡 **Dica**: Salve este arquivo para referência futura!
