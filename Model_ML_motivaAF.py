# Motiva-PA Machine Learning paper ADULTS
### Create by Timothy Cavazzotto
### Started at 24/06/2023
### Ended at 16/07/2023


# install
!pip install shap
!pip install pydot
!pip install hyperopt
!pip install numpy
!pip install pandas
!pip install xgboost
!pip install seaborn
!pip install sklearn
!pip install matplotlib
!pip install imblearn
!pip install openpyxl
!pip install catboost
!pip install lightgbm


# import
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
import pydot
import seaborn as sns
import sklearn
import matplotlib.pyplot as plt

# import from
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import ElasticNet
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from hyperopt import hp
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from xgboost import plot_tree



dados= pd.read_excel("dados_ml.xlsx")
dados


# divisão treino, validação e teste
barrier = dados["barrier"].copy()
dados = dados.drop(["barrier", "id"], axis=1)


# Divisão em treinamento, validação e teste (70/30)
X_train, X_test, y_train, y_test = train_test_split(dados, barrier,
                                                    test_size=0.30,
                                                    random_state=42,
                                                    stratify=barrier)


# Verificação dos tamanhos dos conjuntos
print("Tamanho do conjunto de treinamento:", X_train)
print("Tamanho do conjunto de validação:", X_test)
print("Tamanho do conjunto de teste:", y_train)
print("Tamanho do conjunto de teste:", y_test)

# Os dados atuais tem duas categorias de barriers (modificáveis e não modificáveis)

#  balancear as categorias para o treino
over = SMOTE(sampling_strategy=1.0)
X_train, y_train = over.fit_resample(X_train, y_train)
def getKtops(y_test, k_perc):
    return int(y_test.shape[0]*(k_perc/100))


##### XGBOOST ###########################
##### Estrutura dos hiperparâmetros #####

xgboost_space = {
    'learning_rate': [0.1, 0.01, 0.001],  # determina a contribuição de cada árvore no modelo final.
    'max_depth': [3, 5, 7],               # número de camadas das árvores
    'n_estimators': [50, 100, 150],       # total de árvores a serem construídas para formar o comitê de decisão
    'subsample': [0.80, 0.90, 0.95],      # percentual de amostra utilizada para treinara cada árvore (amostra com reposição) 
    'colsample_bytree': [0.80, 0.90, 1],  # percentual de variáveis nas árvores de treinamento
    'min_child_weight': [1, 5, 10],       # Peso mínimo total para divisão de um nó
    'scale_pos_weight': [1, 2, 3],        # Peso relativo das classes positivas
    'eval_metric': ['auc'],               # Métrica de avaliação
    'seed': [42]                          # Semente aleatória para reprodução dos resultados.
    }


# Criar o classificador XGBoost
xgb_classifier = XGBClassifier()
grid_search = GridSearchCV(estimator=xgb_classifier,
                           param_grid=xgboost_space,
                           scoring='roc_auc',
                           error_score='raise',
                           cv=5)


# Executar a busca em grade
grid_search.fit(X_train, y_train)

# Obter os melhores hiperparâmetros e o melhor desempenho
best_params = grid_search.best_params_
best_score = grid_search.best_score_
print("Melhores hiperparâmetros:", best_params)
print("Melhor ROC AUC:", best_score)

# Treinar o modelo com os melhores hiperparâmetros
best_xgb_classifier = XGBClassifier(**best_params)
best_xgb_classifier.fit(X_train, y_train)


##### CATBOOST ################################
from catboost import CatBoostClassifier

### Estrutura de hiperparâmetros ##############
cat = {
  'iterations': [100, 130, 150],
  'learning_rate': [0.1, 0.01, 0.001],  
  'max_depth': [3, 5, 7],
  'subsample': [0.80, 0.90, 0.95],      
  'scale_pos_weight': [1, 2, 3],       
  'random_state': [42]
  }


catboostclassifier = CatBoostClassifier()
grid_search_cat = GridSearchCV(estimator=catboostclassifier,
                           param_grid=cat,
                           scoring='roc_auc',
                           error_score='raise',
                           cv=5)

grid_search_cat.fit(X_train, y_train)


# Obter os melhores hiperparâmetros e o melhor desempenho
best_params_cat = grid_search_cat.best_params_
best_score_cat = grid_search_cat.best_score_
print("Melhores hiperparâmetros:", best_params_cat)
print("Melhor ROC AUC:", best_score_cat)

# Treinar o modelo com os melhores hiperparâmetros
best_cat_classifier = CatBoostClassifier(**best_params_cat)
best_cat_classifier.fit(X_train, y_train)


#### lightGBM #################################
import lightgbm as lgb
from lightgbm import LGBMClassifier
### Estrutura de hiperparâmetros ##############
space_lgb = {
    'n_estimators': [100],
    'learning_rate': [1],
    'max_depth': [5],
    'min_child_samples': [10],
    'subsample': [1],
    'metric': ['auc'],
    'random_state': [42]
}
lbg_class = LGBMClassifier()
grid_search_lgb = GridSearchCV(estimator=lbg_class,
                           param_grid=space_lgb,
                           scoring='roc_auc',
                           error_score='raise',
                           cv=5)

grid_search_lgb.fit(X_train, y_train)


# Obter os melhores hiperparâmetros e o melhor desempenho
best_params_lgb = grid_search_lgb.best_params_
best_score_lgb = grid_search_lgb.best_score_
print("Melhores hiperparâmetros:", best_params_lgb)
print("Melhor ROC AUC:", best_score_lgb)

# Treinar o modelo com os melhores hiperparâmetros
best_lgb_classifier = LGBMClassifier(**best_params_lgb)
best_lgb_classifier.fit(X_train, y_train)





# Predição com os dados de teste

# Fazer previsões no conjunto de teste
y_pred_xgb = best_xgb_classifier.predict(X_test)


# Avaliar a performance do modelo no conjunto de teste
roc_auc_xgb = roc_auc_score(y_test, y_pred_xgb)


# Imprimir as métricas de avaliação
print("ROC AUC no conjunto de teste:", roc_auc_xgb)

# Calcular a matriz de confusão
cm = confusion_matrix(y_test, y_pred_xgb)

# Imprimir a matriz de confusão
print("Matriz de Confusão:")
print(cm)

# Definir rótulos das classes
class_names = ['Não modificáveis', 'Modificáveis']

# Plotar a matriz de confusão
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=class_names, yticklabels=class_names)

plt.title('Matriz de Confusão')
plt.xlabel('Rótulos Previstos')
plt.ylabel('Rótulos Verdadeiros')
plt.show()






# Fazer previsões no conjunto de teste
y_pred = best_xgb_classifier.predict(X_test)

# Calcular a acurácia
accuracy = accuracy_score(y_test, y_pred)
print("Acurácia:", accuracy)

# Calcular a precisão
precision = precision_score(y_test, y_pred)
print("Precisão:", precision)

# Calcular o recall
recall = recall_score(y_test, y_pred)
print("Recall:", recall)

# Calcular o F1-score
f1 = f1_score(y_test, y_pred)
print("F1-score:", f1)

# Calcular a área sob a curva ROC
roc_auc = roc_auc_score(y_test, y_pred)
print("Área sob a curva ROC:", roc_auc)





# Obter as importâncias das variáveis
importances = best_xgb_classifier.feature_importances_

# Criar uma lista de tuplas com o nome da variável e a importância
feature_importances = [(feature, importance) for feature,
                       importance in zip(X_train, importances)]

# Ordenar a lista pelas importâncias em ordem decrescente
feature_importances = sorted(feature_importances,
                             key=lambda x: x[1],
                             reverse=True)


x_test_labels = X_test.copy()

# Imprimir as variáveis mais importantes
print("Variáveis mais importantes:")
for feature, importance in feature_importances:
    print(f"{feature}: {importance}")


# Plotar a importância das variáveis
shap.summary_plot(shap_values, X_train)

# Calcular os valores de Shapley
explainer = shap.Explainer(best_xgb_classifier)
shap_values = explainer(X_train)

# Plotar a importância das variáveis
shap.summary_plot(shap_values, X_train, show=False)
fig = plt.gcf()
fig.savefig('shap_values.tiff', dpi=300)

