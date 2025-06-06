import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

print("=== CARREGANDO DADOS DA REDE SOCIAL ===")
df = pd.read_csv("../dataset/Social_Network_Ads.csv")

print("Primeiras 5 linhas do dataset:")
print(df.head())
print(f"\nTamanho do dataset: {df.shape[0]} usuários com {df.shape[1]} características")

print("\nEstatísticas descritivas:")
print(df.describe())

print(f"\nValores únicos na coluna 'Purchased' (nossa variável alvo):")
print(f"Classes possíveis: {sorted(df['Purchased'].unique())}")
print(f"Distribuição das classes:\n{df['Purchased'].value_counts()}")

print(f"\nVerificação de dados faltantes:")
missing_data = df.isnull().sum()
if missing_data.sum() == 0:
    print("✓ Não há dados faltantes no dataset!")
else:
    print(missing_data[missing_data > 0])

print(f"\nInformações sobre os tipos de dados:")
print(df.info())

print("\n=== ANÁLISE VISUAL DOS DADOS ===")

plt.figure(figsize=(16, 12))

plt.subplot(2, 3, 1)
numeric_cols = ['Age', 'EstimatedSalary']
df[numeric_cols].boxplot(ax=plt.gca())
plt.title('Detecção de Outliers - Variáveis Numéricas')
plt.xticks(rotation=45)

plt.subplot(2, 3, 2)
purchase_counts = df['Purchased'].value_counts()
plt.bar(['Não Comprou', 'Comprou'], purchase_counts.values, 
        color=['lightcoral', 'lightblue'], alpha=0.7, edgecolor='black')
plt.xlabel('Decisão de Compra')
plt.ylabel('Número de Usuários')
plt.title('Distribuição da Variável Alvo')
for i, v in enumerate(purchase_counts.values):
    plt.text(i, v + 5, str(v), ha='center', va='bottom')

plt.subplot(2, 3, 3)
gender_purchase = pd.crosstab(df['Gender'], df['Purchased'])
gender_purchase.plot(kind='bar', ax=plt.gca(), color=['lightcoral', 'lightblue'])
plt.title('Compras por Gênero')
plt.xlabel('Gênero')
plt.ylabel('Número de Usuários')
plt.legend(['Não Comprou', 'Comprou'])
plt.xticks(rotation=0)

plt.subplot(2, 3, 4)
plt.scatter(df[df['Purchased']==0]['Age'], df[df['Purchased']==0]['EstimatedSalary'], 
           alpha=0.6, color='red', label='Não Comprou', s=30)
plt.scatter(df[df['Purchased']==1]['Age'], df[df['Purchased']==1]['EstimatedSalary'], 
           alpha=0.6, color='blue', label='Comprou', s=30)
plt.xlabel('Idade')
plt.ylabel('Salário Estimado')
plt.title('Idade vs Salário (por Decisão)')
plt.legend()

plt.subplot(2, 3, 5)
df[df['Purchased']==0]['Age'].hist(alpha=0.6, color='red', label='Não Comprou', bins=20)
df[df['Purchased']==1]['Age'].hist(alpha=0.6, color='blue', label='Comprou', bins=20)
plt.xlabel('Idade')
plt.ylabel('Frequência')
plt.title('Distribuição de Idade')
plt.legend()

plt.subplot(2, 3, 6)
df[df['Purchased']==0]['EstimatedSalary'].hist(alpha=0.6, color='red', label='Não Comprou', bins=20)
df[df['Purchased']==1]['EstimatedSalary'].hist(alpha=0.6, color='blue', label='Comprou', bins=20)
plt.xlabel('Salário Estimado')
plt.ylabel('Frequência')
plt.title('Distribuição de Salário')
plt.legend()

plt.tight_layout()
plt.savefig('analise_exploratoria_logistica.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n=== PRÉ-PROCESSAMENTO DOS DADOS ===")

print("Codificando variáveis categóricas...")

df_processed = df.copy()

df_processed['Gender_encoded'] = (df_processed['Gender'] == 'Male').astype(int)

features_to_use = ['Gender_encoded', 'Age', 'EstimatedSalary']
features = df_processed[features_to_use].values
target = df_processed['Purchased'].values

print(f"Características (X): {features.shape[1]} variáveis")
print(f"  • Gender_encoded: 0=Feminino, 1=Masculino")
print(f"  • Age: Idade do usuário") 
print(f"  • EstimatedSalary: Salário estimado")
print(f"Variável alvo (y): decisão de compra (0=Não, 1=Sim)")

print("\nNormalizando os dados...")

mean_X = np.mean(features, axis=0)
std_X = np.std(features, axis=0)
X_normalized = (features - mean_X) / std_X

print("✓ Dados normalizados!")
print(f"Médias após normalização: {np.mean(X_normalized, axis=0)}")
print(f"Desvios padrão após normalização: {np.std(X_normalized, axis=0)}")

print("\n=== DIVISÃO DOS DADOS ===")

np.random.seed(42)
n_samples = len(X_normalized)
n_test = int(n_samples * 0.3)
indices = np.random.permutation(n_samples)
test_indices = indices[:n_test]
train_indices = indices[n_test:]

X_train = X_normalized[train_indices]
X_test = X_normalized[test_indices]
y_train = target[train_indices]
y_test = target[test_indices]

print(f"Dados de treino: {X_train.shape[0]} usuários")
print(f"Dados de teste: {X_test.shape[0]} usuários")

train_dist = np.bincount(y_train) / len(y_train)
test_dist = np.bincount(y_test) / len(y_test)
print(f"Distribuição treino: {train_dist[0]:.2%} Não Comprou, {train_dist[1]:.2%} Comprou")
print(f"Distribuição teste: {test_dist[0]:.2%} Não Comprou, {test_dist[1]:.2%} Comprou")

print("\n=== REGRESSÃO LOGÍSTICA ===")

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def cost_function(X, y, theta):
    m = len(y)
    h = sigmoid(X.dot(theta))
    epsilon = 1e-15
    h = np.clip(h, epsilon, 1 - epsilon)
    cost = -(1/m) * (y.dot(np.log(h)) + (1-y).dot(np.log(1-h)))
    return cost

def gradient_descent_logistic(X, y, learning_rate=0.01, epochs=1000):
    m, n = X.shape
    X_b = np.c_[np.ones((m, 1)), X]
    theta = np.random.randn(n + 1) * 0.01
    
    costs = []
    
    print("Iniciando treinamento da regressão logística...")
    
    for epoch in range(epochs):
        h = sigmoid(X_b.dot(theta))
        
        gradient = (1/m) * X_b.T.dot(h - y)
        
        theta -= learning_rate * gradient
        
        cost = cost_function(X_b, y, theta)
        costs.append(cost)
        
        if epoch % 200 == 0:
            print(f"Época {epoch:4d}: Custo = {cost:.4f}")
    
    return theta, costs

print("\nTreinando o modelo de regressão logística...")
theta_final, training_costs = gradient_descent_logistic(X_train, y_train, 
                                                       learning_rate=0.1, epochs=1000)

print(f"\n✓ Treinamento concluído!")

print("\n=== FAZENDO PREDIÇÕES ===")

def predict(X, theta):
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    probabilities = sigmoid(X_b.dot(theta))
    predictions = (probabilities >= 0.5).astype(int)
    return predictions, probabilities

y_pred, y_prob = predict(X_test, theta_final)

print(f"Predições feitas para {len(y_test)} usuários de teste.")

print("\n=== AVALIAÇÃO DO MODELO ===")

def confusion_matrix(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    return np.array([[TN, FP], [FN, TP]])

def calculate_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = cm.ravel()
    
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'confusion_matrix': cm
    }

def calculate_auc_roc(y_true, y_prob):
    sorted_indices = np.argsort(y_prob)[::-1]
    y_true_sorted = y_true[sorted_indices]
    
    thresholds = np.unique(y_prob)
    tpr_list = []
    fpr_list = []
    
    for threshold in thresholds:
        y_pred_thresh = (y_prob >= threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred_thresh)
        TN, FP, FN, TP = cm.ravel()
        
        tpr = TP / (TP + FN) if (TP + FN) > 0 else 0
        fpr = FP / (FP + TN) if (FP + TN) > 0 else 0
        
        tpr_list.append(tpr)
        fpr_list.append(fpr)
    
    tpr_list = [0] + tpr_list + [1]
    fpr_list = [0] + fpr_list + [1]
    
    auc = 0
    for i in range(1, len(fpr_list)):
        auc += (fpr_list[i] - fpr_list[i-1]) * (tpr_list[i] + tpr_list[i-1]) / 2
    
    return abs(auc), tpr_list, fpr_list

metrics = calculate_metrics(y_test, y_pred)
auc_score, tpr_list, fpr_list = calculate_auc_roc(y_test, y_prob)

print(f"\n📊 RESULTADOS:")
print(f"Acurácia: {metrics['accuracy']:.4f} ({metrics['accuracy']:.1%})")
print(f"Precisão: {metrics['precision']:.4f}")
print(f"Revocação (Recall): {metrics['recall']:.4f}")
print(f"F1-Score: {metrics['f1_score']:.4f}")
print(f"AUC-ROC: {auc_score:.4f}")

print(f"\nMatriz de Confusão:")
cm = metrics['confusion_matrix']
print(f"                 Predito")
print(f"              Não  Sim")
print(f"Real Não    [{cm[0,0]:3d}] [{cm[0,1]:3d}]")
print(f"     Sim    [{cm[1,0]:3d}] [{cm[1,1]:3d}]")

if metrics['accuracy'] > 0.9:
    print("→ 🟢 Modelo excelente!")
elif metrics['accuracy'] > 0.8:
    print("→ 🟡 Modelo muito bom!")
elif metrics['accuracy'] > 0.7:
    print("→ 🟠 Modelo bom.")
elif metrics['accuracy'] > 0.6:
    print("→ 🔴 Modelo razoável.")
else:
    print("→ ⚫ Modelo fraco.")

print("\n=== VISUALIZAÇÕES DOS RESULTADOS ===")

plt.figure(figsize=(18, 12))

plt.subplot(2, 4, 1)
plt.plot(training_costs, color='blue', linewidth=2)
plt.xlabel('Épocas')
plt.ylabel('Custo')
plt.title('Convergência do Treinamento')
plt.grid(True, alpha=0.3)

plt.subplot(2, 4, 2)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Não Comprou', 'Comprou'],
            yticklabels=['Não Comprou', 'Comprou'])
plt.title('Matriz de Confusão')
plt.ylabel('Real')
plt.xlabel('Predito')

plt.subplot(2, 4, 3)
plt.hist(y_prob[y_test==0], bins=20, alpha=0.6, color='red', label='Não Comprou', density=True)
plt.hist(y_prob[y_test==1], bins=20, alpha=0.6, color='blue', label='Comprou', density=True)
plt.axvline(x=0.5, color='black', linestyle='--', label='Threshold=0.5')
plt.xlabel('Probabilidade Predita')
plt.ylabel('Densidade')
plt.title('Distribuição das Probabilidades')
plt.legend()

plt.subplot(2, 4, 4)
plt.plot(fpr_list, tpr_list, color='blue', linewidth=2, label=f'ROC (AUC = {auc_score:.3f})')
plt.plot([0, 1], [0, 1], 'r--', label='Linha de Base')
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Curva ROC')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 4, 5)
X_vis = X_normalized[:, [1, 2]]
y_vis = target

h = 0.1
x_min, x_max = X_vis[:, 0].min() - 1, X_vis[:, 0].max() + 1
y_min, y_max = X_vis[:, 1].min() - 1, X_vis[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

theta_vis, _ = gradient_descent_logistic(X_vis, y_vis, learning_rate=0.1, epochs=500)

grid_points = np.c_[xx.ravel(), yy.ravel()]
grid_points_with_bias = np.c_[np.ones((grid_points.shape[0], 1)), grid_points]
Z = sigmoid(grid_points_with_bias.dot(theta_vis))
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, levels=50, alpha=0.6, cmap='RdYlBu')
plt.colorbar(label='P(Compra)')
scatter = plt.scatter(X_vis[:, 0], X_vis[:, 1], c=y_vis, cmap='RdYlBu', edgecolors='black')
plt.xlabel('Idade (normalizada)')
plt.ylabel('Salário (normalizado)')
plt.title('Fronteira de Decisão')

plt.subplot(2, 4, 6)
feature_names = ['Intercepto', 'Gênero', 'Idade', 'Salário']
weights = theta_final
colors = ['red' if w < 0 else 'blue' for w in weights]
plt.barh(range(len(weights)), weights, color=colors, alpha=0.7)
plt.yticks(range(len(weights)), feature_names)
plt.xlabel('Peso no Modelo')
plt.title('Importância das Características')
plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)

plt.subplot(2, 4, 7)
sample_indices = np.random.choice(len(y_test), 30, replace=False)
x_pos = range(len(sample_indices))
plt.bar([x - 0.2 for x in x_pos], y_test[sample_indices], width=0.4, 
        label='Real', alpha=0.7, color='blue')
plt.bar([x + 0.2 for x in x_pos], y_pred[sample_indices], width=0.4, 
        label='Predito', alpha=0.7, color='red')
plt.xlabel('Amostras')
plt.ylabel('Decisão')
plt.title('Comparação: 30 Amostras')
plt.legend()

plt.subplot(2, 4, 8)
metric_names = ['Acurácia', 'Precisão', 'Recall', 'F1-Score']
metric_values = [metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1_score']]
colors = ['green' if v > 0.8 else 'orange' if v > 0.6 else 'red' for v in metric_values]
bars = plt.bar(metric_names, metric_values, color=colors, alpha=0.7)
plt.ylim(0, 1)
plt.title('Métricas de Performance')
plt.xticks(rotation=45)
for bar, value in zip(bars, metric_values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{value:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('resultados_regressao_logistica.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\n" + "="*70)
print(f"RESUMO FINAL")
print(f"="*70)

print(f"\n📊 DADOS:")
print(f"   • {df.shape[0]} usuários analisados")
print(f"   • {len(features_to_use)} características: {', '.join(features_to_use)}")
print(f"   • {np.sum(target)} usuários compraram ({np.mean(target):.1%})")
print(f"   • {len(target) - np.sum(target)} usuários não compraram ({1-np.mean(target):.1%})")

print(f"\n📈 PERFORMANCE:")
print(f"   • Acurácia = {metrics['accuracy']:.3f} ({metrics['accuracy']:.1%} de acertos)")
print(f"   • Precisão = {metrics['precision']:.3f} (quando prediz compra, acerta {metrics['precision']:.1%})")
print(f"   • Recall = {metrics['recall']:.3f} (encontra {metrics['recall']:.1%} dos compradores)")
print(f"   • F1-Score = {metrics['f1_score']:.3f} (harmônica entre precisão e recall)")
print(f"   • AUC-ROC = {auc_score:.3f} (capacidade de discriminação)")

print(f"\n🔍 CARACTERÍSTICAS MAIS IMPORTANTES:")
feature_importance = list(zip(feature_names[1:], theta_final[1:]))
feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)

for i, (feature, weight) in enumerate(feature_importance):
    direction = "aumenta" if weight > 0 else "diminui"
    print(f"   {i+1}. {feature}: {direction} a chance de compra (peso: {weight:.3f})")

print(f"\n" + "="*70)

print(f"\n=== EXPLICAÇÃO DOS RESULTADOS ===")

print(f"\n🔍 ANÁLISE DAS MÉTRICAS:")

print(f"\n1. ACURÁCIA ({metrics['accuracy']:.3f}):")
if metrics['accuracy'] > 0.85:
    print(f"   ✅ Excelente! O modelo acerta {metrics['accuracy']:.1%} das predições")
elif metrics['accuracy'] > 0.75:
    print(f"   ✅ Muito bom! O modelo acerta {metrics['accuracy']:.1%} das predições")
elif metrics['accuracy'] > 0.65:
    print(f"   🟡 Bom! O modelo acerta {metrics['accuracy']:.1%} das predições")
else:
    print(f"   ⚠️ Pode melhorar. O modelo acerta apenas {metrics['accuracy']:.1%}")

print(f"\n2. PRECISÃO ({metrics['precision']:.3f}):")
if metrics['precision'] > 0.8:
    print(f"   ✅ Quando prediz que alguém vai comprar, acerta {metrics['precision']:.1%} das vezes")
elif metrics['precision'] > 0.6:
    print(f"   🟡 Quando prediz que alguém vai comprar, acerta {metrics['precision']:.1%} das vezes")
else:
    print(f"   ⚠️ Muitos falsos positivos - só acerta {metrics['precision']:.1%} quando prediz compra")

print(f"\n3. RECALL/SENSIBILIDADE ({metrics['recall']:.3f}):")
if metrics['recall'] > 0.8:
    print(f"   ✅ Consegue identificar {metrics['recall']:.1%} dos verdadeiros compradores")
elif metrics['recall'] > 0.6:
    print(f"   🟡 Consegue identificar {metrics['recall']:.1%} dos verdadeiros compradores")
else:
    print(f"   ⚠️ Perde muitos compradores - só identifica {metrics['recall']:.1%}")

print(f"\n4. F1-SCORE ({metrics['f1_score']:.3f}):")
if metrics['f1_score'] > 0.8:
    print(f"   ✅ Excelente equilíbrio entre precisão e recall")
elif metrics['f1_score'] > 0.6:
    print(f"   🟡 Bom equilíbrio entre precisão e recall")
else:
    print(f"   ⚠️ Desequilíbrio entre precisão e recall")

print(f"\n5. AUC-ROC ({auc_score:.3f}):")
if auc_score > 0.9:
    print(f"   ✅ Excelente capacidade de discriminação")
elif auc_score > 0.8:
    print(f"   ✅ Boa capacidade de discriminação")
elif auc_score > 0.7:
    print(f"   🟡 Capacidade moderada de discriminação")
else:
    print(f"   ⚠️ Capacidade fraca de discriminação")

print(f"\n💡 INTERPRETAÇÃO PRÁTICA:")

gender_weight = theta_final[1]
age_weight = theta_final[2]
salary_weight = theta_final[3]

if abs(salary_weight) > abs(age_weight) and abs(salary_weight) > abs(gender_weight):
    print(f"   💰 SALÁRIO é o fator mais importante para a decisão de compra")
elif abs(age_weight) > abs(gender_weight):
    print(f"   👥 IDADE é o fator mais importante para a decisão de compra")
else:
    print(f"   ⚧ GÊNERO é o fator mais importante para a decisão de compra")

if salary_weight > 0.1:
    print(f"   📈 Pessoas com maior salário têm mais chance de comprar")
elif salary_weight < -0.1:
    print(f"   📉 Pessoas com menor salário têm mais chance de comprar")

if age_weight > 0.1:
    print(f"   🧓 Pessoas mais velhas têm mais chance de comprar")
elif age_weight < -0.1:
    print(f"   👶 Pessoas mais novas têm mais chance de comprar")

if gender_weight > 0.1:
    print(f"   👨 Homens têm mais chance de comprar")
elif gender_weight < -0.1:
    print(f"   👩 Mulheres têm mais chance de comprar")

print(f"\n🎯 RECOMENDAÇÕES DE NEGÓCIO:")

if metrics['precision'] > 0.7 and metrics['recall'] > 0.7:
    print(f"   ✅ Modelo confiável para campanhas de marketing direcionado")
elif metrics['precision'] > metrics['recall']:
    print(f"   🎯 Modelo bom para campanhas premium (poucos erros, mas pode perder clientes)")
elif metrics['recall'] > metrics['precision']:
    print(f"   📢 Modelo bom para campanhas amplas (não perde clientes, mas tem custos extras)")

print(f"\n📊 MATRIZ DE CONFUSÃO:")
TN, FP, FN, TP = cm.ravel()
print(f"   • Verdadeiros Negativos (TN): {TN} - Não iam comprar e modelo acertou")
print(f"   • Falsos Positivos (FP): {FP} - Não iam comprar mas modelo disse que sim")
print(f"   • Falsos Negativos (FN): {FN} - Iam comprar mas modelo disse que não")
print(f"   • Verdadeiros Positivos (TP): {TP} - Iam comprar e modelo acertou")

print(f"\n💸 IMPACTO FINANCEIRO (exemplo):")
print(f"   • Se cada campanha custa R$ 10 e cada venda rende R$ 100:")
print(f"   • Lucro com acertos: {TP} × R$ 90 = R$ {TP * 90}")
print(f"   • Prejuízo com falsos positivos: {FP} × R$ 10 = R$ {FP * 10}")
print(f"   • Oportunidade perdida: {FN} × R$ 90 = R$ {FN * 90}")
print(f"   • Lucro líquido estimado: R$ {TP * 90 - FP * 10}")

print(f"\n" + "="*70) 