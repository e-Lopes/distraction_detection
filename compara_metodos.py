import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Modelos a serem avaliados
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier

# Carregar o dataset
data = load_breast_cancer()
X = data.data
y = data.target
feature_names = data.feature_names
target_names = data.target_names

# Verificar distribuição das classes
print(f"Distribuição das classes: Maligno={sum(y==0)}, Benigno={sum(y==1)}")

# Definir os modelos
models = {
    "SVM": SVC(random_state=42),
    "Árvore de Decisão": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),    
    "MLP": MLPClassifier(random_state=42, max_iter=1000),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Bagging": BaggingClassifier(random_state=42),
    "AdaBoost": AdaBoostClassifier(random_state=42),
    "XGBoost": XGBClassifier(random_state=42, eval_metric='logloss')
}

# Configurar validação cruzada
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Avaliar cada modelo
results = {}
for name, model in models.items():
    # Criar pipeline com padronização (exceto para Naive Bayes)
    if name != "Naive Bayes":
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])
    else:
        pipeline = model
    
    # Calcular scores com validação cruzada
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy')
    results[name] = {
        'mean_accuracy': np.mean(scores),
        'std_accuracy': np.std(scores),
        'scores': scores
    }
    print(f"{name}: Acurácia média = {results[name]['mean_accuracy']:.4f} (±{results[name]['std_accuracy']:.4f})")

# Visualizar resultados
df_results = pd.DataFrame.from_dict(results, orient='index')
df_results.sort_values(by='mean_accuracy', ascending=False, inplace=True)

plt.figure(figsize=(12, 6))
plt.bar(df_results.index, df_results['mean_accuracy'], yerr=df_results['std_accuracy'], capsize=5)
plt.ylim(0.8, 1.0)
plt.title('Desempenho dos Modelos (Validação Cruzada 5-Folds)')
plt.ylabel('Acurácia Média')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Mostrar resultados detalhados
print("\nResultados detalhados (ordenados por acurácia):")
print(df_results[['mean_accuracy', 'std_accuracy']].sort_values(by='mean_accuracy', ascending=False))