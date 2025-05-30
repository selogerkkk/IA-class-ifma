import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, confusion_matrix

# Carregar dados
df = pd.read_csv("dataset/winequality-red.csv", sep=";")

# 1. PRÉ-PROCESSAMENTO
print("=== ANÁLISE EXPLORATÓRIA ===")
print(df.head())
print("\n", df.describe())
print("\n", df.info())

print(f"\nValores únicos por coluna:")
for col in df.columns:
    print(f"{col}: {df[col].nunique()} valores únicos")

# Verificar valores ausentes
print(f"\nValores ausentes:\n{df.isnull().sum()}")

# Boxplot para outliers
plt.figure(figsize=(12, 6))
sns.boxplot(data=df)
plt.xticks(rotation=45)
plt.title('Detecção de Outliers')
plt.tight_layout()
plt.savefig('results_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Normalizar dados (exceto target)
scaler = StandardScaler()
features = df.drop('quality', axis=1)
df_scaled = scaler.fit_transform(features)
df_scaled = pd.DataFrame(df_scaled, columns=features.columns)
df_scaled['quality'] = df['quality'].values

print(f"\nDados normalizados!")

# 2. REGRESSÃO LINEAR
print("\n=== REGRESSÃO LINEAR ===")

# Dividir dados
X = df_scaled.drop("quality", axis=1).values
y = df_scaled["quality"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def gradient_descent(X, y, lr=0.01, epochs=1000):
    m, n = X.shape
    X_b = np.c_[np.ones((m, 1)), X]  # adiciona bias
    theta = np.random.randn(n + 1)  # inicializa pesos
    
    for epoch in range(epochs):
        y_pred = X_b.dot(theta)
        error = y_pred - y
        gradients = (2/m) * X_b.T.dot(error)
        theta -= lr * gradients
        
        if epoch % 200 == 0:
            loss = mean_squared_error(y, y_pred)
            print(f"Epoch {epoch}, MSE: {loss:.4f}")
    
    return theta

# Treinar modelo
print("Treinando regressão linear...")
theta = gradient_descent(X_train, y_train)

# Predições
X_test_b = np.c_[np.ones((X_test.shape[0], 1)), X_test]
y_pred = X_test_b.dot(theta)

# Métricas regressão
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n=== RESULTADOS REGRESSÃO LINEAR ===")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R²: {r2:.4f}")

# 3. REGRESSÃO LOGÍSTICA
print("\n=== REGRESSÃO LOGÍSTICA ===")

# Criar problema binário (qualidade >= 6 = boa)
y_binary = (df['quality'] >= 6).astype(int)
X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(X, y_binary, test_size=0.3, random_state=42)

def sigmoid(z):
    z = np.clip(z, -250, 250)  # evitar overflow
    return 1 / (1 + np.exp(-z))

def logistic_gradient_descent(X, y, lr=0.1, epochs=1000):
    m, n = X.shape
    X_b = np.c_[np.ones((m, 1)), X]
    theta = np.random.randn(n + 1)
    
    for epoch in range(epochs):
        z = X_b.dot(theta)
        y_pred = sigmoid(z)
        error = y_pred - y
        gradients = (1/m) * X_b.T.dot(error)
        theta -= lr * gradients
        
        if epoch % 200 == 0:
            cost = -np.mean(y * np.log(y_pred + 1e-15) + (1 - y) * np.log(1 - y_pred + 1e-15))
            print(f"Epoch {epoch}, Cost: {cost:.4f}")
    
    return theta

# Treinar modelo logístico
print("Treinando regressão logística...")
theta_log = logistic_gradient_descent(X_train_log, y_train_log)

# Predições logísticas
X_test_log_b = np.c_[np.ones((X_test_log.shape[0], 1)), X_test_log]
y_pred_proba = sigmoid(X_test_log_b.dot(theta_log))
y_pred_log = (y_pred_proba >= 0.5).astype(int)

# Métricas classificação
accuracy = accuracy_score(y_test_log, y_pred_log)
cm = confusion_matrix(y_test_log, y_pred_log)

print(f"\n=== RESULTADOS REGRESSÃO LOGÍSTICA ===")
print(f"Acurácia: {accuracy:.4f}")
print(f"Matriz de Confusão:\n{cm}")

# Calcular precision, recall, f1
tn, fp, fn, tp = cm.ravel()
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"Precisão: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

# 4. VISUALIZAÇÕES
plt.figure(figsize=(18, 12))

# Predições vs Real (Regressão)
plt.subplot(2, 3, 1)
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Valores Reais')
plt.ylabel('Predições')
plt.title('Regressão Linear: Predições vs Real')

# Resíduos
plt.subplot(2, 3, 2)
residuals = y_test - y_pred
plt.scatter(y_pred, residuals, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predições')
plt.ylabel('Resíduos')
plt.title('Análise de Resíduos')

# Distribuição da qualidade
plt.subplot(2, 3, 3)
plt.hist(df['quality'], bins=10, edgecolor='black', alpha=0.7)
plt.xlabel('Qualidade')
plt.ylabel('Frequência')
plt.title('Distribuição da Qualidade')

# Matriz de correlação
plt.subplot(2, 3, 4)
corr = df.corr()
# Aumentar o tamanho da fonte e melhorar a visualização
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, fmt='.2f', 
            annot_kws={'size': 8}, square=True, linewidths=0.5)
plt.title('Matriz de Correlação', fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(rotation=0, fontsize=9)

# Matriz de confusão
plt.subplot(2, 3, 5)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predito')
plt.ylabel('Real')
plt.title('Matriz de Confusão')

# Distribuição de probabilidades
plt.subplot(2, 3, 6)
plt.hist(y_pred_proba[y_test_log == 0], alpha=0.7, label='Qualidade Baixa', bins=20)
plt.hist(y_pred_proba[y_test_log == 1], alpha=0.7, label='Qualidade Alta', bins=20)
plt.xlabel('Probabilidade')
plt.ylabel('Frequência')
plt.title('Distribuição de Probabilidades')
plt.legend()

plt.tight_layout()
plt.savefig('results_analysis_simple.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\n=== RESUMO FINAL ===")
print(f"Regressão Linear - R²: {r2:.4f}, RMSE: {rmse:.4f}")
print(f"Regressão Logística - Acurácia: {accuracy:.4f}, F1: {f1:.4f}") 