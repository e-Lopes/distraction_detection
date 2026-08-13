# Grid search dos modelos clássicos — janela de 60 frames

Cada fold externo seleciona hiperparâmetros exclusivamente no respectivo bloco de validação. O vídeo de teste do fold não participa da seleção.

## Desempenho externo médio

| Modelo | Ablação | Macro F1 (3 classes) | Balanced accuracy |
|---|---|---:|---:|
| svm | facial_only | 0.4469 | 0.5772 |
| svm | facial_plus_missingness | 0.4351 | 0.5763 |
| svm | missingness_only | 0.4752 | 0.5908 |
| random_forest | facial_only | 0.4708 | 0.5905 |
| random_forest | facial_plus_missingness | 0.4625 | 0.5776 |
| random_forest | missingness_only | 0.4617 | 0.5855 |
| xgboost | facial_only | 0.4674 | 0.5959 |
| xgboost | facial_plus_missingness | 0.4666 | 0.5927 |
| xgboost | missingness_only | 0.4721 | 0.5862 |

## Vencedores por fold de validação

| Modelo | Ablação | Fold | Macro F1 validação | Hiperparâmetros |
|---|---|---:|---:|---|
| random_forest | facial_only | 1 | 0.4350 | `{"class_weight": "balanced_subsample", "max_depth": null, "max_features": "sqrt", "min_samples_leaf": 3, "n_estimators": 500, "n_jobs": -1}` |
| random_forest | facial_only | 2 | 0.4096 | `{"class_weight": "balanced_subsample", "max_depth": null, "max_features": 0.8, "min_samples_leaf": 1, "n_estimators": 500, "n_jobs": -1}` |
| random_forest | facial_only | 3 | 0.4078 | `{"class_weight": "balanced_subsample", "max_depth": null, "max_features": "sqrt", "min_samples_leaf": 1, "n_estimators": 200, "n_jobs": -1}` |
| random_forest | facial_only | 4 | 0.4255 | `{"class_weight": "balanced_subsample", "max_depth": 16, "max_features": 0.8, "min_samples_leaf": 1, "n_estimators": 500, "n_jobs": -1}` |
| random_forest | facial_plus_missingness | 1 | 0.4287 | `{"class_weight": "balanced_subsample", "max_depth": null, "max_features": "sqrt", "min_samples_leaf": 1, "n_estimators": 200, "n_jobs": -1}` |
| random_forest | facial_plus_missingness | 2 | 0.4061 | `{"class_weight": "balanced_subsample", "max_depth": null, "max_features": 0.8, "min_samples_leaf": 1, "n_estimators": 200, "n_jobs": -1}` |
| random_forest | facial_plus_missingness | 3 | 0.4087 | `{"class_weight": "balanced_subsample", "max_depth": 16, "max_features": "sqrt", "min_samples_leaf": 1, "n_estimators": 500, "n_jobs": -1}` |
| random_forest | facial_plus_missingness | 4 | 0.4240 | `{"class_weight": "balanced_subsample", "max_depth": 16, "max_features": "sqrt", "min_samples_leaf": 1, "n_estimators": 500, "n_jobs": -1}` |
| random_forest | missingness_only | 1 | 0.4060 | `{"class_weight": "balanced_subsample", "max_depth": 8, "max_features": 0.8, "min_samples_leaf": 1, "n_estimators": 500, "n_jobs": -1}` |
| random_forest | missingness_only | 2 | 0.3803 | `{"class_weight": "balanced_subsample", "max_depth": 8, "max_features": "sqrt", "min_samples_leaf": 3, "n_estimators": 500, "n_jobs": -1}` |
| random_forest | missingness_only | 3 | 0.3897 | `{"class_weight": "balanced_subsample", "max_depth": null, "max_features": 0.8, "min_samples_leaf": 3, "n_estimators": 200, "n_jobs": -1}` |
| random_forest | missingness_only | 4 | 0.4060 | `{"class_weight": "balanced_subsample", "max_depth": 16, "max_features": "sqrt", "min_samples_leaf": 3, "n_estimators": 500, "n_jobs": -1}` |
| svm | facial_only | 1 | 0.4216 | `{"C": 10.0, "class_weight": "balanced", "kernel": "linear"}` |
| svm | facial_only | 2 | 0.3755 | `{"C": 1.0, "class_weight": "balanced", "gamma": "scale", "kernel": "rbf"}` |
| svm | facial_only | 3 | 0.3964 | `{"C": 0.1, "class_weight": "balanced", "gamma": "scale", "kernel": "rbf"}` |
| svm | facial_only | 4 | 0.4020 | `{"C": 0.1, "class_weight": "balanced", "gamma": "scale", "kernel": "rbf"}` |
| svm | facial_plus_missingness | 1 | 0.4323 | `{"C": 10.0, "class_weight": "balanced", "kernel": "linear"}` |
| svm | facial_plus_missingness | 2 | 0.3831 | `{"C": 1.0, "class_weight": "balanced", "gamma": 0.1, "kernel": "rbf"}` |
| svm | facial_plus_missingness | 3 | 0.3968 | `{"C": 10.0, "class_weight": "balanced", "kernel": "linear"}` |
| svm | facial_plus_missingness | 4 | 0.4034 | `{"C": 10.0, "class_weight": "balanced", "gamma": "scale", "kernel": "rbf"}` |
| svm | missingness_only | 1 | 0.4163 | `{"C": 10.0, "class_weight": "balanced", "gamma": 0.1, "kernel": "rbf"}` |
| svm | missingness_only | 2 | 0.3753 | `{"C": 1.0, "class_weight": "balanced", "kernel": "linear"}` |
| svm | missingness_only | 3 | 0.3862 | `{"C": 0.1, "class_weight": "balanced", "kernel": "linear"}` |
| svm | missingness_only | 4 | 0.3978 | `{"C": 10.0, "class_weight": "balanced", "kernel": "linear"}` |
| xgboost | facial_only | 1 | 0.4209 | `{"colsample_bytree": 0.8, "early_stopping_rounds": 30, "eval_metric": "mlogloss", "learning_rate": 0.1, "max_depth": 5, "min_child_weight": 1, "n_estimators": 300, "n_jobs": -1, "objective": "multi:softprob", "subsample": 0.8, "tree_method": "hist"}` |
| xgboost | facial_only | 2 | 0.3897 | `{"colsample_bytree": 0.8, "early_stopping_rounds": 30, "eval_metric": "mlogloss", "learning_rate": 0.1, "max_depth": 5, "min_child_weight": 1, "n_estimators": 300, "n_jobs": -1, "objective": "multi:softprob", "subsample": 0.8, "tree_method": "hist"}` |
| xgboost | facial_only | 3 | 0.3886 | `{"colsample_bytree": 0.8, "early_stopping_rounds": 30, "eval_metric": "mlogloss", "learning_rate": 0.1, "max_depth": 3, "min_child_weight": 1, "n_estimators": 300, "n_jobs": -1, "objective": "multi:softprob", "subsample": 0.8, "tree_method": "hist"}` |
| xgboost | facial_only | 4 | 0.4183 | `{"colsample_bytree": 1.0, "early_stopping_rounds": 30, "eval_metric": "mlogloss", "learning_rate": 0.1, "max_depth": 3, "min_child_weight": 1, "n_estimators": 300, "n_jobs": -1, "objective": "multi:softprob", "subsample": 0.8, "tree_method": "hist"}` |
| xgboost | facial_plus_missingness | 1 | 0.4211 | `{"colsample_bytree": 1.0, "early_stopping_rounds": 30, "eval_metric": "mlogloss", "learning_rate": 0.03, "max_depth": 5, "min_child_weight": 5, "n_estimators": 600, "n_jobs": -1, "objective": "multi:softprob", "subsample": 0.8, "tree_method": "hist"}` |
| xgboost | facial_plus_missingness | 2 | 0.3911 | `{"colsample_bytree": 1.0, "early_stopping_rounds": 30, "eval_metric": "mlogloss", "learning_rate": 0.03, "max_depth": 5, "min_child_weight": 1, "n_estimators": 600, "n_jobs": -1, "objective": "multi:softprob", "subsample": 0.8, "tree_method": "hist"}` |
| xgboost | facial_plus_missingness | 3 | 0.3914 | `{"colsample_bytree": 0.8, "early_stopping_rounds": 30, "eval_metric": "mlogloss", "learning_rate": 0.1, "max_depth": 5, "min_child_weight": 1, "n_estimators": 300, "n_jobs": -1, "objective": "multi:softprob", "subsample": 0.8, "tree_method": "hist"}` |
| xgboost | facial_plus_missingness | 4 | 0.4174 | `{"colsample_bytree": 1.0, "early_stopping_rounds": 30, "eval_metric": "mlogloss", "learning_rate": 0.1, "max_depth": 3, "min_child_weight": 1, "n_estimators": 300, "n_jobs": -1, "objective": "multi:softprob", "subsample": 0.8, "tree_method": "hist"}` |
| xgboost | missingness_only | 1 | 0.4023 | `{"colsample_bytree": 1.0, "early_stopping_rounds": 30, "eval_metric": "mlogloss", "learning_rate": 0.03, "max_depth": 3, "min_child_weight": 1, "n_estimators": 300, "n_jobs": -1, "objective": "multi:softprob", "subsample": 0.8, "tree_method": "hist"}` |
| xgboost | missingness_only | 2 | 0.3838 | `{"colsample_bytree": 0.8, "early_stopping_rounds": 30, "eval_metric": "mlogloss", "learning_rate": 0.03, "max_depth": 5, "min_child_weight": 1, "n_estimators": 300, "n_jobs": -1, "objective": "multi:softprob", "subsample": 0.8, "tree_method": "hist"}` |
| xgboost | missingness_only | 3 | 0.3764 | `{"colsample_bytree": 0.8, "early_stopping_rounds": 30, "eval_metric": "mlogloss", "learning_rate": 0.1, "max_depth": 5, "min_child_weight": 1, "n_estimators": 300, "n_jobs": -1, "objective": "multi:softprob", "subsample": 0.8, "tree_method": "hist"}` |
| xgboost | missingness_only | 4 | 0.3966 | `{"colsample_bytree": 1.0, "early_stopping_rounds": 30, "eval_metric": "mlogloss", "learning_rate": 0.1, "max_depth": 5, "min_child_weight": 1, "n_estimators": 300, "n_jobs": -1, "objective": "multi:softprob", "subsample": 0.8, "tree_method": "hist"}` |

A variação de parâmetros entre folds é parte da avaliação aninhada do procedimento. Nenhuma configuração final única deve ser escolhida olhando o desempenho de teste.
