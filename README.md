# 🍷 Predição de Qualidade de Vinhos

Projeto de machine learning que usa **regressão linear** para prever a qualidade de vinhos tintos baseado em suas características químicas.

## 📋 Pré-requisitos

- Python 3.7+
- Bibliotecas: pandas, numpy, matplotlib, seaborn, scikit-learn

## 🚀 Como usar

### 1. Instalar dependências
```bash
pip install -r requirements.txt
```

### 2. Executar o projeto

**Opção A - Direto:**
```bash
python wine_analysis_simple.py
```

**Opção B - Com script (Linux/Mac):**
```bash
chmod +x run.sh
./run.sh
```

**Opção C - Com ambiente virtual:**
```bash
# Criar ambiente virtual
python -m venv wine_env

# Ativar (Linux/Mac)
source wine_env/bin/activate

# Ativar (Windows)
wine_env\Scripts\activate

# Instalar dependências
pip install -r requirements.txt

# Executar
python wine_analysis_simple.py
```

## 📊 O que o projeto faz

1. **Carrega** o dataset de vinhos tintos (1.599 amostras)
2. **Analisa** os dados e cria visualizações
3. **Normaliza** as características químicas
4. **Treina** um modelo de regressão linear
5. **Avalia** a performance do modelo
6. **Gera** gráficos dos resultados

## 📈 Resultados

O projeto gera dois arquivos de imagem:
- `analise_exploratoria.png` - Análise inicial dos dados
- `resultados_regressao_linear.png` - Resultados do modelo

## 📁 Estrutura do projeto

```
├── wine_analysis_simple.py    # Código principal
├── dataset/
│   └── winequality-red.csv   # Dataset dos vinhos
├── requirements.txt          # Dependências
├── README.md                # Este arquivo
├── explicacao_projeto.md    # Explicações detalhadas
└── explicacao_graficos.md   # Como interpretar os gráficos
```

## 🎯 Métricas principais

- **R²**: Porcentagem da variação explicada pelo modelo
- **RMSE**: Erro médio em pontos de qualidade
- **MAE**: Erro absoluto médio (mais fácil de interpretar)

## 📚 Arquivos de apoio

- `explicacao_projeto.md` - Explicação completa do projeto
- `explicacao_graficos.md` - Como interpretar cada gráfico

## 🔧 Personalização

Para ajustar o modelo, modifique os parâmetros em `gradient_descent()`:
- `learning_rate`: Taxa de aprendizado (padrão: 0.01)
- `epochs`: Número de iterações (padrão: 1000) 