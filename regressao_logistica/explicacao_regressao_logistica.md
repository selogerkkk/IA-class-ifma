# 📊 Regressão Logística - Predição de Compras em Rede Social

## 🎯 O que é este projeto?

Este projeto usa **regressão logística** para prever se um usuário vai comprar um produto anunciado em uma rede social, baseado em suas características demográficas (idade, gênero e salário).

## 📋 Problema de Negócio

**Situação:** Uma empresa quer otimizar suas campanhas publicitárias direcionando anúncios apenas para usuários com maior probabilidade de compra.

**Objetivo:** Criar um modelo que preveja se um usuário vai comprar (1) ou não comprar (0) um produto.

**Benefícios esperados:**
- 💰 Reduzir custos de publicidade
- 🎯 Aumentar taxa de conversão
- 📈 Melhorar ROI das campanhas

## 📊 Dataset: Social Network Ads

### Características do dataset:
- **400 usuários** analisados
- **5 variáveis** (User ID, Gender, Age, EstimatedSalary, Purchased)
- **Variável alvo:** Purchased (0 = Não comprou, 1 = Comprou)
- **Features:** Gênero, Idade, Salário Estimado

### Distribuição dos dados:
- **257 usuários (64.3%)** não compraram
- **143 usuários (35.7%)** compraram
- **Idades:** 18 a 60 anos
- **Salários:** R$ 15.000 a R$ 150.000

## 🔍 Por que Regressão Logística?

### Diferenças da Regressão Linear:
| Aspecto                | Regressão Linear             | Regressão Logística        |
| ---------------------- | ---------------------------- | -------------------------- |
| **Tipo de problema**   | Predição (valores contínuos) | Classificação (categorias) |
| **Variável alvo**      | Numérica (ex: preço, peso)   | Categórica (ex: sim/não)   |
| **Função de ativação** | Identidade (y = x)           | Sigmóide (0 ≤ y ≤ 1)       |
| **Interpretação**      | Valor direto                 | Probabilidade              |
| **Exemplo de saída**   | "Casa vale R$ 300.000"       | "85% de chance de comprar" |

### Vantagens da Regressão Logística:
- ✅ **Probabilístico**: Retorna probabilidades (0 a 100%)
- ✅ **Interpretável**: Coeficientes têm significado claro
- ✅ **Robusto**: Menos sensível a outliers
- ✅ **Eficiente**: Rápido para treinar e prever
- ✅ **Baseline**: Excelente ponto de partida

## ⚙️ Como Funciona?

### 1. Função Sigmóide
A regressão logística usa a função sigmóide para converter valores reais em probabilidades:

```
P(y=1) = 1 / (1 + e^(-z))
onde z = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ
```

**Características da sigmóide:**
- 📈 **Formato de S**: Suave transição de 0 para 1
- 🎯 **Limitada**: Sempre entre 0 e 1
- ⚖️ **Ponto médio**: 0.5 quando z = 0

### 2. Processo de Decisão
```
1. Calcular z = β₀ + β₁×gênero + β₂×idade + β₃×salário
2. Aplicar sigmóide: P(compra) = 1/(1 + e^(-z))
3. Decisão: Se P(compra) ≥ 0.5 → "Vai comprar"
            Se P(compra) < 0.5 → "Não vai comprar"
```

### 3. Treinamento (Gradiente Descendente)
O algoritmo ajusta os pesos (β) para minimizar o erro:

```python
for epoch in range(epochs):
    # 1. Calcular predições
    predictions = sigmoid(X.dot(weights))
    
    # 2. Calcular erro
    error = predictions - actual_values
    
    # 3. Calcular gradiente
    gradient = X.T.dot(error) / m
    
    # 4. Atualizar pesos
    weights -= learning_rate * gradient
```

## 📈 Resultados do Modelo

### 🎯 Métricas de Performance

| Métrica      | Valor | O que significa                               |
| ------------ | ----- | --------------------------------------------- |
| **Acurácia** | 85.8% | Acerta 85.8% de todas as predições            |
| **Precisão** | 94.1% | Quando prevê "compra", acerta 94.1% das vezes |
| **Recall**   | 68.1% | Encontra 68.1% de todos os compradores reais  |
| **F1-Score** | 79.0% | Média harmônica entre precisão e recall       |

### 📊 Matriz de Confusão Explicada

```
                 PREDIÇÕES
              Não    Sim
REAL  Não   [ 71] [ 2 ]  ← Especificidade: 97.3%
      Sim   [ 15] [32 ]  ← Sensibilidade: 68.1%
```

**Interpretação:**
- **71 Verdadeiros Negativos**: Não iam comprar e modelo acertou
- **32 Verdadeiros Positivos**: Iam comprar e modelo acertou  
- **2 Falsos Positivos**: Não iam comprar mas modelo disse que sim
- **15 Falsos Negativos**: Iam comprar mas modelo disse que não

### 💰 Impacto Financeiro (Simulação)

Assumindo:
- **Custo por anúncio:** R$ 10
- **Lucro por venda:** R$ 100
- **Teste em 120 usuários**

| Cenário                         | Resultado         | Valor         |
| ------------------------------- | ----------------- | ------------- |
| **Acertos (32 vendas)**         | Lucro             | +R$ 2.880     |
| **Falsos positivos (2)**        | Desperdício       | -R$ 20        |
| **Oportunidades perdidas (15)** | Não contabilizado | R$ 1.350      |
| **💰 LUCRO LÍQUIDO**             | **Total**         | **+R$ 2.860** |

## 🔍 Insights dos Dados

### 🏆 Características Mais Importantes

1. **🎂 IDADE (+2.072)**
   - **Maior influência** na decisão de compra
   - Pessoas mais velhas têm mais chance de comprar
   - A cada ano a mais, probabilidade aumenta significativamente

2. **💰 SALÁRIO (+1.123)**
   - **Segunda maior influência**
   - Maior renda = maior chance de compra
   - Faz sentido: mais dinheiro disponível

3. **⚧ GÊNERO (+0.136)**
   - **Menor influência**
   - Homens têm ligeira tendência a comprar mais
   - Diferença pequena, mas estatisticamente relevante

### 📊 Padrões Descobertos

**👴 Idade e Compras:**
- Usuários **mais jovens (18-30)**: Baixa taxa de compra
- Usuários **meia-idade (30-45)**: Taxa moderada
- Usuários **mais velhos (45-60)**: Alta taxa de compra

**💸 Salário e Compras:**
- **Baixa renda (<R$ 50k)**: Raramente compram
- **Renda média (R$ 50-80k)**: Taxa moderada
- **Alta renda (>R$ 80k)**: Frequentemente compram

**⚧ Gênero e Compras:**
- **Mulheres**: 34.5% de taxa de compra
- **Homens**: 37.1% de taxa de compra
- Diferença pequena, mas consistente

## 🎯 Aplicações Práticas

### 📢 Recomendações de Marketing

**🎯 Público-Alvo Ideal:**
- **Homens** de **40-55 anos**
- **Salário acima de R$ 70.000**
- **Probabilidade de compra: >80%**

**📈 Estratégias Sugeridas:**

1. **Campanha Premium**
   - Focar nos usuários de alta probabilidade (>70%)
   - Menor volume, maior conversão
   - ROI otimizado

2. **Campanha Ampla**
   - Incluir usuários de média probabilidade (40-70%)
   - Maior alcance, custos maiores
   - Boa para awareness de marca

3. **Segmentação por Idade**
   - **18-30 anos**: Produtos mais baratos, promoções
   - **30-45 anos**: Foco em qualidade e durabilidade  
   - **45+ anos**: Produtos premium, facilidade de uso

### 🔮 Casos de Uso do Modelo

**✅ Quando usar este modelo:**
- Campanhas de email marketing
- Segmentação de anúncios no Facebook/Google
- Personalização de ofertas
- Otimização de orçamento publicitário

**⚠️ Limitações:**
- Apenas 3 variáveis (poderia ter mais)
- Dataset pequeno (400 usuários)
- Não considera sazonalidade
- Não inclui histórico de compras

## 🚀 Próximos Passos

### 📊 Para Melhorar o Modelo:

1. **Mais Dados**
   - Histórico de navegação
   - Interações em redes sociais
   - Localização geográfica
   - Timing das visitas

2. **Engenharia de Features**
   - Interação idade×salário
   - Categorização de faixas etárias
   - Normalização por região

3. **Algoritmos Avançados**
   - Random Forest
   - Gradient Boosting
   - Redes Neurais

### 💼 Para Implementação:

1. **Coleta de Dados**
   - Integrar com Google Analytics
   - Pixels de conversão
   - CRM da empresa

2. **Monitoramento**
   - A/B testing
   - Métricas de negócio
   - Retreat do modelo

3. **Automatização**
   - API para predições em tempo real
   - Segmentação automática
   - Alertas de performance

## 🛠️ Aspectos Técnicos

### ⚙️ Implementação

**Pré-processamento:**
- ✅ Codificação de variáveis categóricas (Female=0, Male=1)
- ✅ Normalização Z-score (média=0, desvio=1)
- ✅ Divisão treino/teste (70%/30%)

**Hiperparâmetros:**
- **Learning Rate:** 0.1 (otimizado)
- **Épocas:** 1000 (convergência garantida)
- **Inicialização:** Pesos pequenos aleatórios

**Validação:**
- ✅ Sem overfitting detectado
- ✅ Convergência suave
- ✅ Métricas consistentes

### 📊 Interpretação dos Coeficientes

| Feature        | Coeficiente | Interpretação                            |
| -------------- | ----------- | ---------------------------------------- |
| **Intercepto** | -1.234      | Baseline (probabilidade base)            |
| **Gênero**     | +0.136      | Ser homem aumenta log-odds em 0.136      |
| **Idade**      | +2.072      | Cada ano aumenta log-odds em 2.072       |
| **Salário**    | +1.123      | Cada unidade de salário aumenta log-odds |

**Para interpretar em probabilidade:**
```
Odds = e^(coeficiente)
Probabilidade = Odds / (1 + Odds)
```

## 📚 Conclusões

### ✅ O que funcionou bem:
- **Alta precisão (94.1%)**: Poucos falsos positivos
- **Boa acurácia geral (85.8%)**: Modelo confiável
- **Interpretabilidade**: Coeficientes fazem sentido
- **Convergência rápida**: Treino eficiente

### ⚠️ Pontos de atenção:
- **Recall moderado (68.1%)**: Perde alguns compradores
- **Dataset pequeno**: Apenas 400 usuários
- **Features limitadas**: Só 3 variáveis preditoras
- **Balanceamento**: Classes desbalanceadas (64%/36%)

### 🎯 Valor para o negócio:
- **ROI positivo**: Modelo gera lucro estimado de R$ 2.860
- **Eficiência**: Reduz desperdício em 97% (apenas 2 falsos positivos)
- **Escalabilidade**: Pode ser aplicado a milhões de usuários
- **Transparência**: Resultados explicáveis para stakeholders

---

**🏁 Resultado Final:** Modelo de regressão logística eficaz para predição de compras, com performance sólida e aplicação prática imediata para otimização de campanhas publicitárias. 