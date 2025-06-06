# Explicação dos Gráficos - Análise de Regressão Logística

Este documento explica todos os gráficos gerados durante a análise de regressão logística para predição de compras em redes sociais.

## 1. Gráficos de Análise Exploratória

### 1.1 Distribuição das Variáveis
**Arquivo:** `distribuicao_variaveis.png`

Este gráfico mostra histogramas das principais variáveis:
- **Idade**: Mostra a distribuição etária dos usuários
- **Salário Estimado**: Revela a distribuição de renda
- **Compra**: Indica o balanceamento da variável target

**Como interpretar:**
- Procure por outliers ou valores extremos
- Verifique se as distribuições são normais ou enviesadas
- Observe se a variável target está balanceada

**Sinais de sucesso:**
- Distribuições sem outliers extremos
- Dados bem distribuídos sem concentrações anômalas
- Classes balanceadas na variável target

### 1.2 Matriz de Correlação
**Arquivo:** `matriz_correlacao.png`

Mostra as correlações entre todas as variáveis numéricas usando um heatmap.

**Como interpretar:**
- Valores próximos a 1: correlação positiva forte
- Valores próximos a -1: correlação negativa forte
- Valores próximos a 0: sem correlação linear

**O que procurar:**
- Correlações altas entre features (multicolinearidade)
- Correlações significativas com a variável target
- Padrões interessantes nos dados

### 1.3 Distribuição por Gênero
**Arquivo:** `distribuicao_genero.png`

Gráfico de barras mostrando:
- Quantidade de homens e mulheres no dataset
- Taxa de compra por gênero

**Interpretação:**
- Verifica se há balanceamento entre gêneros
- Mostra se um gênero tem maior propensão a comprar
- Importante para identificar viés nos dados

### 1.4 Análise de Idade vs Compra
**Arquivo:** `idade_vs_compra.png`

Box plot comparando a distribuição de idades entre:
- Pessoas que compraram (Purchased = 1)
- Pessoas que não compraram (Purchased = 0)

**Como interpretar:**
- Compare as medianas entre os grupos
- Observe a dispersão (tamanho das caixas)
- Identifique outliers (pontos fora dos whiskers)

**Insights esperados:**
- Pessoas mais velhas tendem a comprar mais
- Maior variabilidade em um dos grupos

### 1.5 Análise de Salário vs Compra
**Arquivo:** `salario_vs_compra.png`

Box plot similar ao anterior, mas para salários.

**Interpretação:**
- Pessoas com salários mais altos compram mais?
- Qual a dispersão de salários em cada grupo?
- Existem outliers de renda?

### 1.6 Scatter Plot: Idade vs Salário
**Arquivo:** `scatter_idade_salario.png`

Gráfico de dispersão colorido por status de compra.

**Como interpretar:**
- Pontos vermelhos: não compraram
- Pontos azuis: compraram
- Procure por padrões ou clusters
- Identifique regiões com maior concentração de compras

**Padrões típicos:**
- Zona de alta compra: idade alta + salário alto
- Zona de baixa compra: idade baixa + salário baixo

## 2. Gráficos de Treinamento e Resultados

### 2.1 Convergência do Treinamento
**Arquivo:** `convergencia_treinamento.png`

Mostra a evolução da função de custo durante o treinamento.

**Como interpretar:**
- Eixo X: número de iterações
- Eixo Y: valor da função de custo (log-loss)
- Curva deve ser decrescente e estabilizar

**Sinais de sucesso:**
- Curva suave e descendente
- Estabilização em um valor baixo
- Sem oscilações ou aumentos

**Sinais de problema:**
- Curva oscilante (learning rate muito alto)
- Não convergência (learning rate muito baixo)
- Aumento do custo (overfitting)

### 2.2 Matriz de Confusão
**Arquivo:** `matriz_confusao.png`

Tabela 2x2 mostrando as predições vs valores reais.

**Estrutura:**
```
                Predito
Real      0    1
  0      TN   FP
  1      FN   TP
```

**Métricas derivadas:**
- **Acurácia**: (TP + TN) / Total
- **Precisão**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1-Score**: 2 * (Precisão * Recall) / (Precisão + Recall)

### 2.3 Curva ROC
**Arquivo:** `curva_roc.png`

Gráfica a Taxa de Verdadeiros Positivos vs Taxa de Falsos Positivos.

**Como interpretar:**
- Linha diagonal: classificador aleatório (AUC = 0.5)
- Curva mais próxima do canto superior esquerdo: melhor modelo
- Área sob a curva (AUC): métrica de qualidade

**Valores de AUC:**
- 0.5: modelo aleatório
- 0.7-0.8: modelo razoável
- 0.8-0.9: modelo bom
- 0.9-1.0: modelo excelente

### 2.4 Distribuição de Probabilidades
**Arquivo:** `distribuicao_probabilidades.png`

Histograma das probabilidades preditas para cada classe.

**Como interpretar:**
- Duas distribuições separadas indicam boa separabilidade
- Sobreposição indica dificuldade de classificação
- Picos próximos a 0 e 1 indicam predições confiantes

**Ideal:**
- Distribuição bimodal
- Picos em 0 e 1
- Pouca sobreposição no meio

### 2.5 Análise de Resíduos
**Arquivo:** `analise_residuos.png`

Mostra os resíduos (diferença entre predito e real) vs valores preditos.

**Como interpretar:**
- Pontos devem estar aleatoriamente distribuídos
- Sem padrões sistemáticos
- Variância constante (homocedasticidade)

**Sinais de problema:**
- Padrões curvos (não-linearidade)
- Variância crescente (heterocedasticidade)
- Outliers extremos

### 2.6 Importância das Features
**Arquivo:** `importancia_features.png`

Gráfico de barras mostrando a importância relativa de cada variável.

**Como interpretar:**
- Barras maiores: features mais importantes
- Baseado nos coeficientes do modelo
- Ajuda a entender quais variáveis mais influenciam a decisão

### 2.7 Predições vs Realidade
**Arquivo:** `predicoes_vs_realidade.png`

Scatter plot comparando valores reais vs probabilidades preditas.

**Como interpretar:**
- Eixo X: valores reais (0 ou 1)
- Eixo Y: probabilidades preditas (0 a 1)
- Boa separação indica modelo eficaz

**Ideal:**
- Pontos de classe 0 concentrados em baixo (prob < 0.5)
- Pontos de classe 1 concentrados em cima (prob > 0.5)
- Clara separação entre os grupos

### 2.8 Análise de Limiar de Decisão
**Arquivo:** `analise_limiar.png`

Mostra como diferentes limiares afetam precisão, recall e F1-score.

**Como interpretar:**
- Eixo X: diferentes valores de limiar (0 a 1)
- Eixo Y: métricas de performance
- Ajuda a escolher o melhor limiar para o problema

**Trade-offs:**
- Limiar baixo: maior recall, menor precisão
- Limiar alto: maior precisão, menor recall
- F1-score: balanço entre precisão e recall

## 3. Dicas de Interpretação

### Sinais de um Modelo Bem Treinado:
1. **Convergência suave** na função de custo
2. **AUC-ROC > 0.8** 
3. **Matriz de confusão** com poucos falsos positivos/negativos
4. **Distribuição de probabilidades** bem separada
5. **Resíduos** aleatoriamente distribuídos

### Sinais de Problemas:
1. **Overfitting**: performance perfeita no treino, ruim no teste
2. **Underfitting**: performance ruim em ambos os conjuntos
3. **Desbalanceamento**: alta acurácia mas baixo recall para classe minoritária
4. **Outliers**: pontos extremos que podem distorcer o modelo

### Melhorias Possíveis:
1. **Feature Engineering**: criar novas variáveis relevantes
2. **Regularização**: L1 ou L2 para evitar overfitting
3. **Balanceamento**: técnicas como SMOTE para classes desbalanceadas
4. **Seleção de Features**: remover variáveis irrelevantes
5. **Ajuste de Hiperparâmetros**: otimizar learning rate e outros parâmetros

## 4. Aplicação Prática

### Para Tomada de Decisão:
- Use a **matriz de confusão** para entender erros
- Observe a **curva ROC** para avaliar performance geral
- Analise **importância das features** para insights de negócio
- Considere **análise de limiar** para otimizar métricas específicas

### Para Comunicação com Stakeholders:
- **Acurácia** é mais intuitiva para explicar performance
- **Gráficos de distribuição** mostram padrões nos dados
- **Importância das features** revela quais fatores mais influenciam
- **Análise de ROI** quantifica o impacto financeiro

Este conjunto de gráficos fornece uma visão completa do modelo de regressão logística, desde a análise exploratória até a avaliação final, permitindo tanto validação técnica quanto comunicação eficaz dos resultados. 