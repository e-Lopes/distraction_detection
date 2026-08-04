import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import GridSearchCV, cross_val_predict, StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn import model_selection

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
import xgboost as xgb

# Carregar dados
dataset = load_breast_cancer()
X = dataset.data
y = dataset.target
target_names = dataset.target_names

cv_outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Modelos e parâmetros
models_params = {
    "Decision Tree": (
        DecisionTreeClassifier(random_state=42),
        [{'criterion':['gini', 'entropy', 'log_loss'],
          'max_depth': [None, 5, 10],
          'min_samples_split':[2, 5],
          'splitter':['random', 'best']}]
    ),
    "KNN": (
        KNeighborsClassifier(),
        [{'n_neighbors':[1,3,5],
          'weights': ['uniform', 'distance'],
          'p':[1,2]}]
    ),
    "Naive Bayes": (
        GaussianNB(),
        [{'var_smoothing':[1e-09]}]
    ),
    "Random Forest": (
        RandomForestClassifier(random_state=42),
        [{'criterion':['gini', 'entropy'],
          'max_depth': [5, 10],
          'min_samples_split':[2, 5],
          'max_samples': [0.3, 0.5],
          'n_estimators':[10, 50, 100, 200]}]
    ),
    "AdaBoost": (
        AdaBoostClassifier(random_state=42),
        [{'n_estimators': [10, 50, 100]}]
    ),
    "Bagging": (
        BaggingClassifier(random_state=42),
        [{'n_estimators': [10, 50, 100]}]
    ),
    "XGBoost": (
        xgb.XGBClassifier(eval_metric='logloss', random_state=42),
        [{'n_estimators': [10, 50, 100],
          'max_depth': [3, 5, 10],
          'learning_rate': [0.01, 0.1, 0.2]}]
    ),
    "MLP": (
        MLPClassifier(max_iter=1000, random_state=42),
        [{'hidden_layer_sizes':[8, (8, 4), 10],
          'learning_rate': ['constant', 'invscaling'],
          'learning_rate_init':[0.01, 0.001, 0.0001],
          'activation':['relu', 'logistic', 'tanh'],
          'random_state':[10, 46, 37]}]
    ),
    "SVM": (
        SVC(random_state=42),
        [{'C':[1,5,10,50],
          'kernel': ['rbf', 'poly', 'linear'],
          'gamma':[0.1, 0.001, 'scale']}]
    )
}

# Armazenar matrizes
matrizes = []
nomes_matrizes = []

def run_grid_search_evaluate(name, clf, params):
    print(f"\n--- {name} ---")
    gs = GridSearchCV(clf, params, scoring='accuracy', cv=5, n_jobs=-1)
    gs.fit(X, y)

    best_clf = gs.best_estimator_
    print(f"Melhores parâmetros: {gs.best_params_}")

    # Avaliação com melhor estimador
    y_pred = cross_val_predict(best_clf, X, y, cv=cv_outer)

    precision = precision_score(y, y_pred, average='macro')
    recall = recall_score(y, y_pred, average='macro')
    f1 = f1_score(y, y_pred, average='macro')
    acc = model_selection.cross_val_score(best_clf, X, y, cv=5).mean()

    print(f"Acurácia média: {acc:.3f}")
    print(f"F1-score (macro): {f1:.3f}")
    print(f"Precision (macro): {precision:.3f}")
    print(f"Recall (macro): {recall:.3f}")

    matrix = confusion_matrix(y, y_pred)
    matrizes.append(matrix)
    nomes_matrizes.append(name)

# Executa o experimento
for name, (clf, params) in models_params.items():
    run_grid_search_evaluate(name, clf, params)

# Exibir todas as matrizes no final
n = len(matrizes)
cols = 3
rows = int(np.ceil(n / cols))

fig, axs = plt.subplots(rows, cols, figsize=(15, 5 * rows))
axs = axs.flatten()

for i, (matrix, nome) in enumerate(zip(matrizes, nomes_matrizes)):
    disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=target_names)
    disp.plot(ax=axs[i], colorbar=False)
    axs[i].set_title(f'{nome}')

# Remover gráficos extras, se houver
for j in range(i + 1, len(axs)):
    fig.delaxes(axs[j])

plt.tight_layout()
plt.show()
