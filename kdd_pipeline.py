import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from config import root
os.makedirs("kdd_plots", exist_ok=True)

# 1. Carregar os dados
df = pd.read_csv(f"{root}/channel_history/[LPS] Los Pombos/│📗│chat-geraldo.csv")

# 2. Criar novas features
df['data'] = pd.to_datetime(df['data'])
df['hora_do_dia'] = df['data'].dt.hour
df['engajou'] = (df['reacoes'] > 0).astype(int)
df['tem_anexo'] = df['tem_anexo'].astype(int)
df['menciona_alguem'] = df['menciona_alguem'].astype(int)

features = ['message_length', 'hora_do_dia', 'tem_anexo', 'menciona_alguem']

# 3. Estatísticas básicas
print("=== ESTATÍSTICAS BÁSICAS ===")
print(df[features].describe())

# 4. Distribuição (Histogramas e Boxplots)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(df['message_length'], bins=30, kde=True)
plt.title("Distribuição do Tamanho das Mensagens")
plt.subplot(1, 2, 2)
sns.boxplot(x='engajou', y='message_length', data=df)
plt.title("Boxplot: Tamanho vs Engajamento")
plt.tight_layout()
plt.savefig("kdd_plots/dist_message_length.png", dpi=150, bbox_inches='tight')

# Boxplots para hora_do_dia
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(df['hora_do_dia'], bins=24, kde=False)
plt.title("Distribuição por Hora do Dia")
plt.subplot(1, 2, 2)
sns.boxplot(x='engajou', y='hora_do_dia', data=df)
plt.title("Boxplot: Hora do Dia vs Engajamento")
plt.tight_layout()
plt.savefig("kdd_plots/dist_hora_do_dia.png", dpi=150, bbox_inches='tight')

# 5. Correlação entre atributos
plt.figure(figsize=(8, 6))
sns.heatmap(df[features + ['engajou']].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Matriz de Correlação")
plt.tight_layout()
plt.savefig("kdd_plots/correlacao_matriz.png", dpi=150, bbox_inches='tight')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 6. Correlação dos atributos com o target — justifica a seleção
print("\n=== CORRELAÇÃO DOS ATRIBUTOS COM O TARGET (engajou) ===")
correlacao_target = df[features + ['engajou']].corr()['engajou'].drop('engajou').sort_values(ascending=False)
print(correlacao_target)

plt.figure(figsize=(7, 4))
correlacao_target.plot(kind='bar', color=['#4CAF50' if v > 0 else '#EF5350' for v in correlacao_target])
plt.title("Correlação dos Atributos com Engajamento")
plt.ylabel("Correlação de Pearson")
plt.axhline(0, color='black', linewidth=0.8)
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("kdd_plots/correlacao_target.png", dpi=150, bbox_inches='tight')

# 7. Seleção de dados relevantes
X = df[features]
y = df['engajou']

# 8. Separação em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 9. Normalização
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 10. Redução de dimensionalidade (PCA)
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)
print(f"\nVariância explicada pelos componentes PCA: {pca.explained_variance_ratio_}")

# Visualização da projeção PCA
plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    X_train_pca[:, 0], X_train_pca[:, 1],
    c=y_train, cmap='coolwarm', alpha=0.6, edgecolors='k', linewidths=0.3
)
plt.colorbar(scatter, label='Engajou (0 = Não, 1 = Sim)')
plt.title("Projeção PCA — Treino (2 componentes)")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variância)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variância)")
plt.tight_layout()
plt.savefig("kdd_plots/pca_projecao.png", dpi=150, bbox_inches='tight')

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# 11. Comparação entre diferentes configurações (bias/variância)
configuracoes = [
    {"max_depth": 3,    "min_samples_leaf": 2,  "n_estimators": 50},
    {"max_depth": 5,    "min_samples_leaf": 4,  "n_estimators": 100},  # configuração final
    {"max_depth": 10,   "min_samples_leaf": 1,  "n_estimators": 100},
    {"max_depth": None, "min_samples_leaf": 1,  "n_estimators": 200},  # sem restrição
]

print("\n=== COMPARAÇÃO DE CONFIGURAÇÕES (Cross-Validation 5-fold) ===")
print(f"{'Config':<10} {'max_depth':<12} {'min_leaf':<10} {'n_est':<8} {'F1 médio':<12} {'Desvio'}")
print("-" * 65)

resultados_configs = []
for i, cfg in enumerate(configuracoes):
    modelo_cfg = RandomForestClassifier(**cfg, class_weight="balanced", random_state=42)
    scores = cross_val_score(modelo_cfg, X_train_scaled, y_train, cv=5, scoring='f1')
    resultados_configs.append(scores.mean())
    print(f"Config {i+1:<4} {str(cfg['max_depth']):<12} {cfg['min_samples_leaf']:<10} {cfg['n_estimators']:<8} {scores.mean():.4f}       ±{scores.std():.4f}")

plt.figure(figsize=(8, 5))
labels = [f"Config {i+1}\ndepth={c['max_depth']}" for i, c in enumerate(configuracoes)]
cores = ['#4CAF50' if v == max(resultados_configs) else '#90CAF9' for v in resultados_configs]
plt.bar(labels, resultados_configs, color=cores)
plt.title("F1-Score médio por configuração (Cross-Validation 5-fold)")
plt.ylabel("F1-Score (classe engajou=1)")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("kdd_plots/configuracoes_comparacao.png", dpi=150, bbox_inches='tight')

# 12. Modelo final
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    min_samples_leaf=4,
    class_weight="balanced",
    random_state=42
)
rf_model.fit(X_train_scaled, y_train)

# 13. Cross-validation no modelo final (estratégia anti-overfitting)
print("\n=== CROSS-VALIDATION — MODELO FINAL ===")
cv_scores = cross_val_score(rf_model, X_train_scaled, y_train, cv=5, scoring='f1')
print(f"F1 por fold: {[round(s, 4) for s in cv_scores]}")
print(f"F1 médio:    {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print("(Desvio baixo indica modelo estável, sem overfitting severo)")

# 14. Avaliação no conjunto de teste (Holdout)
y_pred = rf_model.predict(X_test_scaled)

print("\n=== AVALIAÇÃO DO MODELO ===")
print("Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred))
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))

# 15. Importância dos atributos
importancias = pd.Series(rf_model.feature_importances_, index=features).sort_values(ascending=False)
print("\nImportância dos Atributos:")
print(importancias)

plt.figure(figsize=(7, 4))
importancias.plot(kind='bar', color='#1565c0')
plt.title("Importância dos Atributos (Random Forest)")
plt.ylabel("Importância")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("kdd_plots/feature_importance.png", dpi=150, bbox_inches='tight')