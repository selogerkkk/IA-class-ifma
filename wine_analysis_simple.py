import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, confusion_matrix

print("=== CARREGANDO DADOS DO VINHO ===")
df = pd.read_csv("dataset/winequality-red.csv", sep=";")

print("Primeiras 5 linhas do dataset:")
print(df.head())
print(f"\nTamanho do dataset: {df.shape[0]} vinhos com {df.shape[1]} características")

print("\nEstatísticas descritivas:")
print(df.describe())

print(f"\nValores únicos na coluna 'quality' (nossa variável alvo):")
print(f"Qualidades possíveis: {sorted(df['quality'].unique())}")
print(f"Distribuição das qualidades:\n{df['quality'].value_counts().sort_index()}")

print(f"\nVerificação de dados faltantes:")
missing_data = df.isnull().sum()
if missing_data.sum() == 0:
    print("✓ Não há dados faltantes no dataset!")
else:
    print(missing_data[missing_data > 0])

print("\n=== ANÁLISE VISUAL DOS DADOS ===")

plt.figure(figsize=(16, 12))

plt.subplot(2, 3, 1)
sns.boxplot(data=df)
plt.xticks(rotation=45)
plt.title('Detecção de Outliers')
plt.tight_layout()

plt.subplot(2, 3, 2)
plt.hist(df['quality'], bins=10, edgecolor='black', alpha=0.7, color='skyblue')
plt.xlabel('Qualidade do Vinho')
plt.ylabel('Número de Vinhos')
plt.title('Distribuição da Qualidade')

plt.subplot(2, 3, 3)
corr = df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, fmt='.2f', 
            annot_kws={'size': 8}, square=True, linewidths=0.5)
plt.title('Matriz de Correlação', fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(rotation=0, fontsize=9)

plt.subplot(2, 3, 4)
plt.scatter(df['alcohol'], df['quality'], alpha=0.6, color='red')
plt.xlabel('Teor Alcoólico')
plt.ylabel('Qualidade')
plt.title('Álcool vs Qualidade')

plt.subplot(2, 3, 5)
plt.scatter(df['volatile acidity'], df['quality'], alpha=0.6, color='purple')
plt.xlabel('Acidez Volátil')
plt.ylabel('Qualidade')
plt.title('Acidez Volátil vs Qualidade')

plt.tight_layout()
plt.savefig('analise_exploratoria.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n=== PRÉ-PROCESSAMENTO DOS DADOS ===")

print("Normalizando os dados...")

features = df.drop('quality', axis=1)
target = df['quality']

print(f"Características (X): {features.shape[1]} variáveis")
print(f"Variável alvo (y): qualidade do vinho")

scaler = StandardScaler()
X_normalized = scaler.fit_transform(features)

print("✓ Dados normalizados!")

print("\n=== DIVISÃO DOS DADOS ===")

X_train, X_test, y_train, y_test = train_test_split(
    X_normalized, target, test_size=0.3, random_state=42
)

print(f"Dados de treino: {X_train.shape[0]} vinhos")
print(f"Dados de teste: {X_test.shape[0]} vinhos")

print("\n=== REGRESSÃO LINEAR ===")

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def gradient_descent(X, y, learning_rate=0.01, epochs=1000):
    m, n = X.shape
    X_b = np.c_[np.ones((m, 1)), X]
    theta = np.random.randn(n + 1)
    
    print("Iniciando treinamento...")
    
    for epoch in range(epochs):
        y_pred = X_b.dot(theta)
        error = y_pred - y
        gradients = (2/m) * X_b.T.dot(error)
        theta -= learning_rate * gradients
        
        if epoch % 200 == 0:
            loss = mean_squared_error(y, y_pred)
            print(f"Época {epoch:4d}: MSE = {loss:.4f}")
    
    return theta

print("\nTreinando o modelo de regressão linear...")
theta_final = gradient_descent(X_train, y_train, learning_rate=0.01, epochs=1000)

print(f"\n✓ Treinamento concluído!")

print("\n=== FAZENDO PREDIÇÕES ===")

X_test_b = np.c_[np.ones((X_test.shape[0], 1)), X_test]
y_pred = X_test_b.dot(theta_final)

print(f"Predições feitas para {len(y_test)} vinhos de teste.")

print("\n=== AVALIAÇÃO DO MODELO ===")

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n📊 RESULTADOS:")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R²: {r2:.4f}")

if r2 > 0.7:
    print("→ 🟢 Modelo muito bom!")
elif r2 > 0.5:
    print("→ 🟡 Modelo razoável.")
elif r2 > 0.3:
    print("→ 🟠 Modelo fraco, mas melhor que o acaso.")
else:
    print("→ 🔴 Modelo muito fraco.")

print("\n=== VISUALIZAÇÕES DOS RESULTADOS ===")

plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.scatter(y_test, y_pred, alpha=0.6, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Qualidade Real')
plt.ylabel('Qualidade Predita')
plt.title('Predições vs Realidade')

plt.subplot(2, 3, 2)
residuals = y_test - y_pred
plt.scatter(y_pred, residuals, alpha=0.6, color='green')
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predições')
plt.ylabel('Resíduos')
plt.title('Análise de Resíduos')

plt.subplot(2, 3, 3)
plt.hist(residuals, bins=20, edgecolor='black', alpha=0.7, color='orange')
plt.xlabel('Resíduos')
plt.ylabel('Frequência')
plt.title('Distribuição dos Erros')

plt.subplot(2, 3, 4)
feature_names = features.columns
weights = theta_final[1:]
plt.barh(range(len(weights)), weights)
plt.yticks(range(len(weights)), feature_names)
plt.xlabel('Peso no Modelo')
plt.title('Importância das Características')

plt.subplot(2, 3, 5)
sample_indices = np.random.choice(len(y_test), 20, replace=False)
x_pos = range(len(sample_indices))
plt.bar([x - 0.2 for x in x_pos], y_test.iloc[sample_indices], width=0.4, 
        label='Real', alpha=0.7, color='blue')
plt.bar([x + 0.2 for x in x_pos], y_pred[sample_indices], width=0.4, 
        label='Predito', alpha=0.7, color='red')
plt.xlabel('Amostras')
plt.ylabel('Qualidade')
plt.title('Comparação: 20 Amostras')
plt.legend()

plt.tight_layout()
plt.savefig('resultados_regressao_linear.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\n" + "="*60)
print(f"RESUMO FINAL")
print(f"="*60)

print(f"\n📊 DADOS:")
print(f"   • {df.shape[0]} vinhos analisados")
print(f"   • {df.shape[1]-1} características químicas")

print(f"\n📈 PERFORMANCE:")
print(f"   • R² = {r2:.3f} ({r2:.1%} da variação explicada)")
print(f"   • Erro médio = {mae:.2f} pontos de qualidade")
print(f"   • RMSE = {rmse:.2f}")

print(f"\n🔍 CARACTERÍSTICAS MAIS IMPORTANTES:")
feature_importance = list(zip(features.columns, theta_final[1:]))
feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)

for i, (feature, weight) in enumerate(feature_importance[:5]):
    direction = "aumenta" if weight > 0 else "diminui"
    print(f"   {i+1}. {feature}: {direction} a qualidade (peso: {weight:.3f})")

print(f"\n" + "="*60) 