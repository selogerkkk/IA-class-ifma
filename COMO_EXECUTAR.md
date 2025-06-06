# 🚀 Como Executar o Projeto

## 📋 Pré-requisitos

1. **Python 3.7+** instalado
2. **Acesso ao terminal/linha de comando**
3. **Repositório clonado** ou pasta extraída

## ⚙️ Configuração Inicial (Uma vez)

```bash
# 1. Navegar para a pasta do projeto
cd /caminho/para/ifma-IA

# 2. Criar ambiente virtual
python3 -m venv venv

# 3. Ativar ambiente virtual
# Linux/Mac:
source venv/bin/activate
# Windows:
# venv\Scripts\activate

# 4. Instalar dependências
pip install pandas seaborn matplotlib numpy
```

## 📊 Executar Regressão Linear (Análise de Vinhos)

```bash
# Navegar para a pasta
cd regressao_linear

# Ativar ambiente (se não estiver ativo)
source ../venv/bin/activate

# Executar análise
python wine_analysis_simple.py
```

### 📈 O que será executado:
- ✅ Carregamento do dataset de vinhos (1599 amostras)
- ✅ Análise exploratória com 5 gráficos
- ✅ Pré-processamento (normalização, divisão train/test)
- ✅ Treinamento do modelo de regressão linear
- ✅ Avaliação com métricas (R², RMSE, MAE)
- ✅ Geração de 5 gráficos de resultados

### 📁 Arquivos gerados:
- `analise_exploratoria.png` - Gráficos exploratórios
- `resultados_regressao_linear.png` - Gráficos de resultados

---

## 📈 Executar Regressão Logística (Anúncios)

```bash
# Navegar para a pasta
cd ../regressao_logistica

# Ativar ambiente (se não estiver ativo)
source ../venv/bin/activate

# Executar análise
python logistic_regression_analysis.py
```

### 📈 O que será executado:
- ✅ Carregamento do dataset de anúncios (400 usuários)
- ✅ Análise exploratória com 6 gráficos
- ✅ Pré-processamento (codificação, normalização, divisão)
- ✅ Treinamento do modelo de regressão logística
- ✅ Avaliação com métricas (Acurácia, Precisão, Recall, F1, AUC-ROC)
- ✅ Geração de 8 gráficos de resultados

### 📁 Arquivos gerados:
- `analise_exploratoria_logistica.png` - Gráficos exploratórios
- `resultados_regressao_logistica.png` - Gráficos de resultados

---

## 🔧 Solução de Problemas

### ❌ Erro: "No such file or directory: dataset/..."
**Solução:** Certifique-se de estar na pasta correta (`regressao_linear/` ou `regressao_logistica/`)

### ❌ Erro: "ModuleNotFoundError: No module named 'pandas'"
**Solução:** 
```bash
# Ativar ambiente virtual
source venv/bin/activate  # ou ../venv/bin/activate

# Reinstalar dependências
pip install pandas seaborn matplotlib numpy
```

### ❌ Erro: "Permission denied"
**Solução:**
```bash
# Dar permissão de execução
chmod +x venv/bin/activate
```

### ❌ Windows: Problemas com ambiente virtual
**Solução:**
```cmd
# Usar este comando no Windows
venv\Scripts\activate.bat
```

---

## 📊 Verificação dos Resultados

### ✅ Regressão Linear - Resultados Esperados:
```
📊 RESULTADOS:
MSE: ~0.40
RMSE: ~0.64
MAE: ~0.51
R²: ~0.35 (35% da variação explicada)
```

### ✅ Regressão Logística - Resultados Esperados:
```
📊 RESULTADOS:
Acurácia: ~85.8% 
Precisão: ~94.1%
Recall: ~68.1%
F1-Score: ~79.0%
```

---

## 📚 Documentação Adicional

### 📖 Regressão Linear:
- **`explicacao_projeto.md`** - Teoria e implementação detalhada
- **`explicacao_graficos.md`** - Como interpretar os gráficos

### 📖 Regressão Logística:
- **`explicacao_regressao_logistica.md`** - Teoria e implementação detalhada
- **`explicacao_graficos_logistica.md`** - Como interpretar os gráficos

---

## 🎯 Execução Rápida (Ambos os Projetos)

```bash
# Script completo para executar tudo
cd /caminho/para/ifma-IA

# Ativar ambiente
source venv/bin/activate

# Executar regressão linear
cd regressao_linear
python wine_analysis_simple.py
echo "✅ Regressão Linear concluída!"

# Executar regressão logística
cd ../regressao_logistica
python logistic_regression_analysis.py
echo "✅ Regressão Logística concluída!"

echo "🎉 Todos os projetos executados com sucesso!"
```

---

## 📁 Estrutura Final dos Arquivos

```
ifma-IA/
├── regressao_linear/
│   ├── wine_analysis_simple.py
│   ├── analise_exploratoria.png          ← Gerado
│   ├── resultados_regressao_linear.png   ← Gerado
│   ├── explicacao_projeto.md
│   └── explicacao_graficos.md
├── regressao_logistica/
│   ├── logistic_regression_analysis.py
│   ├── analise_exploratoria_logistica.png    ← Gerado
│   ├── resultados_regressao_logistica.png   ← Gerado
│   ├── explicacao_regressao_logistica.md
│   └── explicacao_graficos_logistica.md
├── dataset/
│   ├── winequality-red.csv
│   └── Social_Network_Ads.csv
├── venv/                                 ← Ambiente virtual
├── README.md                             ← Documentação geral
└── COMO_EXECUTAR.md                      ← Este arquivo
```

---

## 🎓 Para Uso Acadêmico

**✅ Requisitos Atendidos:**
- [x] Pré-processamento completo
- [x] Implementação manual (sem sklearn)
- [x] Todas as métricas solicitadas
- [x] Análise exploratória detalhada
- [x] Visualizações profissionais
- [x] Documentação acadêmica

**📋 Para Apresentação:**
1. Execute ambos os projetos
2. Abra os arquivos PNG gerados
3. Consulte os arquivos .md para explicações
4. Use os gráficos para demonstrar resultados

---

*Projeto organizado e testado - Janeiro 2025* 