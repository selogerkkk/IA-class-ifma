# 🤖 Projeto de Machine Learning - Regressão Linear e Logística

## 📋 Descrição do Projeto

Este projeto implementa algoritmos de **Regressão Linear** e **Regressão Logística** do zero, sem uso de bibliotecas como sklearn, seguindo os requisitos acadêmicos especificados. Ambos os algoritmos incluem análise exploratória completa, pré-processamento, implementação manual e avaliação detalhada.

## 📁 Estrutura do Projeto

```
ifma-IA/
├── dataset/
│   ├── winequality-red.csv        # Dataset de qualidade de vinhos
│   └── Social_Network_Ads.csv     # Dataset de anúncios de rede social
├── regressao_linear/               # 📊 Projeto Regressão Linear
│   ├── wine_analysis_simple.py    # Implementação principal
│   ├── explicacao_projeto.md      # Documentação detalhada
│   ├── explicacao_graficos.md     # Guia dos gráficos
│   ├── base_analise_exploratoria.png      # Gráficos exploratórios
│   └── base_resultados_regressao_linear.png # Gráficos de resultados
├── regressao_logistica/            # 📈 Projeto Regressão Logística
│   ├── logistic_regression_analysis.py    # Implementação principal
│   ├── explicacao_regressao_logistica.md  # Documentação detalhada
│   ├── explicacao_graficos_logistica.md   # Guia dos gráficos
│   ├── analise_exploratoria_logistica.png # Gráficos exploratórios
│   └── resultados_regressao_logistica.png # Gráficos de resultados
├── venv/                          # Ambiente virtual Python
├── README.md                      # Este arquivo (documentação geral)
└── requirements.txt               # Dependências do projeto
```

## 🚀 Como Executar

### 1. Configurar Ambiente

```bash
# Criar ambiente virtual
python3 -m venv venv

# Ativar ambiente virtual
source venv/bin/activate

# Instalar dependências
pip install pandas seaborn matplotlib numpy
```

### 2. Executar os Projetos

```bash
# Regressão Linear (Análise de Vinhos)
cd regressao_linear
python wine_analysis_simple.py

# Regressão Logística (Anúncios de Rede Social)
cd ../regressao_logistica
python logistic_regression_analysis.py
```

## 📊 Projeto 1: Regressão Linear

**📂 Localização:** `regressao_linear/`

### 🎯 Objetivo
Prever a **qualidade de vinhos** (3-8) baseada em características químicas.

### 📈 Dataset
- **1599 vinhos** com 11 características químicas
- **Arquivo:** `dataset/winequality-red.csv`

### 📊 Resultados
- **R² = 36.9%** - Explica boa parte da variação
- **RMSE = 0.647** - Erro médio de ±0.65 pontos
- **MAE = 0.513** - Erro absoluto médio baixo

### 📚 Documentação
- **`explicacao_projeto.md`** - Explicação completa do algoritmo
- **`explicacao_graficos.md`** - Guia para interpretar visualizações

---

## 📈 Projeto 2: Regressão Logística

**📂 Localização:** `regressao_logistica/`

### 🎯 Objetivo
Prever se um usuário vai **comprar um produto** (0/1) baseado em características demográficas.

### 📈 Dataset
- **400 usuários** com idade, gênero e salário
- **Arquivo:** `dataset/Social_Network_Ads.csv`

### 📊 Resultados
- **Acurácia = 85.8%** - Excelente performance geral
- **Precisão = 94.1%** - Poucos falsos positivos
- **Recall = 68.1%** - Encontra maioria dos compradores
- **F1-Score = 79.0%** - Bom equilíbrio

### 📚 Documentação
- **`explicacao_regressao_logistica.md`** - Explicação completa do algoritmo
- **`explicacao_graficos_logistica.md`** - Guia para interpretar visualizações

---

## ✅ Requisitos Atendidos

### 1. Pré-Processamento ✅
- [x] Análise exploratória inicial (head, describe, info, value_counts)
- [x] Detecção de outliers com boxplots
- [x] Tratamento de valores ausentes
- [x] Normalização/padronização (média 0, variância 1)
- [x] Codificação de variáveis categóricas
- [x] Divisão 70% treino / 30% teste

### 2. Processamento ✅
- [x] **Regressão Linear** com gradiente descendente (implementação própria)
- [x] **Regressão Logística** com gradiente descendente (implementação própria)
- [x] **Sem uso de bibliotecas como sklearn**

### 3. Pós-processamento ✅

#### Regressão Linear:
- [x] MSE (Mean Squared Error)
- [x] RMSE (Root Mean Squared Error)
- [x] MAE (Mean Absolute Error)
- [x] R² (Coeficiente de Determinação)

#### Regressão Logística:
- [x] Acurácia
- [x] Precisão (Precision)
- [x] Revocação/Sensibilidade (Recall)
- [x] F1-Score
- [x] Matriz de Confusão
- [x] AUC-ROC (implementação manual)

---

## 🛠️ Tecnologias Utilizadas

- **Python 3.12**
- **Pandas** - Manipulação de dados
- **NumPy** - Computação numérica
- **Matplotlib** - Visualizações básicas
- **Seaborn** - Visualizações estatísticas

---

## 📋 Instruções Detalhadas

### Configuração Inicial
```bash
# 1. Navegar para o diretório do projeto
cd /caminho/para/ifma-IA

# 2. Verificar datasets
ls dataset/
# Deve mostrar: Social_Network_Ads.csv  winequality-red.csv

# 3. Criar e ativar ambiente virtual
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# ou venv\Scripts\activate para Windows

# 4. Instalar dependências
pip install pandas seaborn matplotlib numpy
```

### Execução dos Projetos
```bash
# Projeto 1: Regressão Linear
cd regressao_linear
python wine_analysis_simple.py

# Projeto 2: Regressão Logística  
cd ../regressao_logistica
python logistic_regression_analysis.py
```

### Verificação dos Resultados
Após execução, verifique se foram gerados os gráficos:
- `regressao_linear/base_*.png`
- `regressao_logistica/*_logistica.png`

---

## 🎯 Resumo dos Resultados

### 📊 Regressão Linear (Vinhos):
- **Performance Moderada**: 36.9% da variação explicada
- **Insight Principal**: Álcool e acidez são fatores-chave
- **Aplicação**: Orientação para produtores de vinho

### 📈 Regressão Logística (Anúncios):
- **Performance Excelente**: 85.8% de acurácia
- **Insight Principal**: Idade e salário determinam compras
- **Aplicação**: Otimização de campanhas publicitárias

---

## 🚀 Próximos Passos

### Para Regressão Linear:
- [ ] Engenharia de features avançada
- [ ] Regularização (Ridge/Lasso)
- [ ] Modelos ensemble

### Para Regressão Logística:
- [ ] Otimização de threshold
- [ ] Balanceamento de classes
- [ ] Features de interação

---

## 👥 Equipe

**Desenvolvido para a disciplina de Inteligência Artificial - IFMA**

---

## 📚 Referências

- Dataset de Vinhos: UCI Machine Learning Repository
- Dataset de Anúncios: Simulação de dados de marketing
- Algoritmos: Implementação baseada na literatura clássica de ML

---

## 📝 Licença

Este projeto é desenvolvido para fins acadêmicos.

---

*Projeto organizado e documentado - Janeiro 2025* 