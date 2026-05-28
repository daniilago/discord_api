import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from config import root

# 1. Carregar os dados (ajuste o caminho para o seu arquivo real)
df = pd.read_csv(f"{root}/channel_history/Talvez seja um servidor/geral.csv")

# 2. Criar novas features com base nos dados coletados
df['data'] = pd.to_datetime(df['data'])
df['hora_do_dia'] = df['data'].dt.hour
df['engajou'] = (df['reacoes'] > 0).astype(int)
df['tem_anexo'] = df['tem_anexo'].astype(int)
df['menciona_alguem'] = df['menciona_alguem'].astype(int)

# 3. Estatísticas básicas
features = ['message_length', 'hora_do_dia', 'tem_anexo', 'menciona_alguem']
print(df[features].describe()) # Gera min, max, média, desvio-padrão e quartis

# 4. Distribuição (Histogramas e Boxplots)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(df['message_length'], bins=30, kde=True)
plt.title("Distribuição do Tamanho das Mensagens")

plt.subplot(1, 2, 2)
sns.boxplot(x='engajou', y='message_length', data=df)
plt.title("Boxplot: Tamanho vs Engajamento")
plt.show()

# 5. Correlação
plt.figure(figsize=(8, 6))
sns.heatmap(df[features + ['engajou']].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Matriz de Correlação")
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Seleção de dados relevantes
X = df[features]
y = df['engajou']

# Separação em treino e teste (evita data leakage antes de normalizar)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Normalização
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Redução de dimensionalidade (exemplo reduzindo para 2 componentes)
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print(f"Variância explicada pelos componentes: {pca.explained_variance_ratio_}")

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# O parâmetro class_weight="balanced" resolve problemas de classes desbalanceadas 
# (ex: se 90% das mensagens não têm reações).
# max_depth=5 e min_samples_leaf=4 limitam a profundidade da árvore para evitar overfitting.
rf_model = RandomForestClassifier(
    n_estimators=100, 
    max_depth=5, 
    min_samples_leaf=4, 
    class_weight="balanced", 
    random_state=42
)

# Treinando o modelo (usando os dados normais em vez do PCA para manter explicabilidade)
rf_model.fit(X_train_scaled, y_train)

# Predições
y_pred = rf_model.predict(X_test_scaled)

# Imprimindo as métricas
print("\n=== AVALIAÇÃO DO MODELO ===")
print("Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred))
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))

# Importância de cada atributo para a decisão final
importancias = pd.Series(rf_model.feature_importances_, index=features)
print("\nImportância dos Atributos:")
print(importancias.sort_values(ascending=False))