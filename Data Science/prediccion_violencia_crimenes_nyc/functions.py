from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from sklearn.pipeline import Pipeline
from feature_engine.encoding import OrdinalEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score

class SelectFeatures(BaseEstimator, TransformerMixin):

    def __init__(self, features_names) :

        self.features_names = features_names   
######
    def fit(self, X, y = None):       
        return self
        
    def transform(self, X, y = None):
        X_ = X.copy()
        X_= X_.loc[:,self.features_names]

        return X_

def del_first_zero(string):
    check = '0'

    if check == string[0]:
        string = string[1:]      
          
    return string

def identify_high_correlations(df, threshold=.8):
    """
    identify_high_correlations: Genera un reporte sobre las correlaciones existentes entre variables, condicional a un nivel arbitrario.

    Parámetros de ingreso:
        - df: un objeto pd.DataFrame, por lo general es la base de datos a trabajar.
        - threshold: Nivel de correlaciones a considerar como altas. Por defecto es .7.

    Retorno:
        - Un pd.DataFrame con los nombres de las variables y sus correlaciones
    """

    # extraemos la matriz de correlación con una máscara booleana
    tmp = df.corr().mask(abs(df.corr()) < .8, df)
    # convertimos a long format
    tmp = pd.melt(tmp)
    # agregamos una columna extra que nos facilitará los cruces entre variables
    tmp['var2'] = list(df.columns) * len(df.columns)
    # reordenamos
    tmp = tmp[['variable', 'var2', 'value']].dropna()
    # eliminamos valores duplicados
    tmp = tmp[tmp['value'].duplicated()]
    # eliminamos variables con valores de 1
    return tmp[tmp['value'] < 1.00]

def gridsearch_train(dict, X_train, X_test, y_train, y_test, features):
    

    best_score_list = []    
    best_params_list = []
    score_validation = []
    index_names = []

    dict_modelos = {}
    
    for key, item in dict.items():          
        

        pipe = Pipeline([
                ('select', SelectFeatures(features)),
                ('ode', OrdinalEncoder(encoding_method='arbitrary')),
                ('model', item[0])
            ])
        
        search = GridSearchCV(pipe, item[1], cv = 5, scoring = 'f1_weighted', n_jobs = -1, return_train_score=True)
        search.fit(X_train, y_train)

        y_pred = search.best_estimator_.predict(X_test)
        score_validacion = f1_score(y_test, y_pred, average='weighted')        
                
        best_score_list.append(search.best_score_.round(4))
        best_params_list.append(search.best_params_)
        score_validation.append(score_validacion)
        index_names.append(key)

        dict_modelos[key] = search

    data = pd.DataFrame(data={'best_score':best_score_list,'best_params': best_params_list, 'f1_weighted': score_validation}, index=index_names)    
    return data, dict_modelos