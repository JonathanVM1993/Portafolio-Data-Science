import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_curve, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from feature_engine.wrappers import SklearnTransformerWrapper
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import statsmodels.api as sm
import numpy as np
from IPython.display import display

def plot_hist(dataframe, dicc):

    '''
    Función que permite graficar histogramas.

    Parámetros
    ----------
    dataframe: Dataframe de pandas.
    dicc: Diccionario que debe contener como Key los títulos y Value las variables a gráficar del DataFrame.
    '''

    plt.figure(figsize=(14,5))
    for i, item in enumerate(dicc.items()):    
        plt.subplot(1, 3, i+1)          
        sns.distplot(dataframe[item[0]])
        plt.xlabel(item[0])
        plt.axvline(dataframe[item[0]].mean(), color='tomato')
        plt.title(item[1])
        plt.tight_layout()
        plt.ylabel('Frecuencia')

def plot_box(dataframe, dicc):

    '''
    Función que permite graficar boxplots.

    Parámetros
    ----------
    dataframe: Dataframe de pandas.
    dicc: Diccionario que debe contener como Key los títulos y Value las variables a gráficar del DataFrame.    
    '''

    plt.figure(figsize=(14,5))
    for i, item in enumerate(dicc.items()):    
        plt.subplot(1, 3, i+1)          
        sns.boxplot(dataframe[item[0]])
        plt.xlabel(item[0])    
        plt.title(item[1])
        plt.tight_layout()       


def plot_bar(dataframe, dicc):
    '''
    Función que permite graficar barras.

    Parámetros
    ----------
    dataframe: Dataframe de pandas.
    dicc: Diccionario que debe contener como Key los títulos y Value las variables a gráficar del DataFrame.    
    '''

    plt.figure(figsize=(18,15))
    for i, item in enumerate(dicc.items()):    
        plt.subplot(5, 4, i+1)          
        sns.countplot(data=dataframe , x = item[0])
        plt.xlabel(item[0])
        #plt.axvline(dataframe[item[0]].mean(), color='tomato')
        plt.title(item[1])
        plt.ylabel('Frecuencia')
        plt.tight_layout()

def plot_vector(dataframe, listado):
    '''
    Función que permite hacer gráficos countplot.

    Parámetros
    ----------
    dataframe: Dataframe de pandas.
    dicc: Diccionario que debe contener como Key los títulos y Value las variables a gráficar del DataFrame.    
    '''

    plt.figure(figsize=(11, 4))

    for i, item in enumerate(listado):    
            plt.subplot(1, 3, i+1)          
            sns.countplot(data=dataframe , x = item)
            plt.xlabel(item)
            #plt.axvline(dataframe[item[0]].mean(), color='tomato')
            if item == 'Stroke':
                plt.title('Accidente cerebrovascular')
            elif item == 'Hypertension':
                plt.title('Hipertensión')
            else:
                plt.title(item)
            plt.ylabel('Frecuencia')
            plt.tight_layout()

def train_model(dataframe, vector_objetivo, features, diccionario_grilla, rand_state, scoring='accuracy', scalar=False, scalar_variables=None):
    
    '''
    Función que busca los mejores hiperparámetros de un grupo de modelos y devuelve la métrica solicitada.

    Parámetros
    ----------
    dataframe: Dataframe pandas.
    vector_objetivo: Vector objetivo a entrenar y modelar.
    features: Variables independientes que se usarán en el entrenamiento.
    diccionario_grilla: Diccionario que contiene los modelos a entrenar junto con su grilla de hiperparámetros a buscar.
    rand_state: Semilla Aleatoria que se usará en el entrenamiento.
    scoring: Tipo de métrica a utilizar.
    scalar: Booleano. True: Escalar variables. False: No escalar variables.
    scalar_variables: Variables que se escalarán.

    Returns
    -------
    data: dataframe con columnas "best_score", "best_params" y métrica seleccionada por modelo evaluado
    dict_modelos: diccionario de modelos entrenados
    '''

    X = dataframe.loc[:,features]

    X = X.drop(columns=[vector_objetivo])

    y = dataframe[vector_objetivo]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=rand_state)


    list_score_validation = []
    list_score_train = []
    list_best_params = []
    list_auc_score = []
    index_names = []
    dict_modelos = {}

    for key, item in diccionario_grilla.items():        

        if scalar == True:
            pipe = Pipeline([
                ('sc', SklearnTransformerWrapper(StandardScaler(), variables=scalar_variables)),
                ('model', item[0])
            ])
            search = GridSearchCV(pipe, item[1], cv = 5, scoring=scoring, n_jobs=-1, return_train_score=True)
            search.fit(X_train, y_train)
        else:
            search = GridSearchCV(item[0], item[1], cv = 5, scoring=scoring, n_jobs=-1, return_train_score=True)
            search.fit(X_train, y_train)
        

        y_pred = search.best_estimator_.predict(X_test)
        y_pred_train = search.best_estimator_.predict(X_train)
        y_pred_roc = search.best_estimator_.predict_proba(X_test)[:,1]

        roc_auc_score_1 = roc_auc_score(y_test, y_pred_roc)
        

        if scoring == 'accuracy':
            score_validacion = accuracy_score(y_test, y_pred)
            score_train = accuracy_score(y_train, y_pred_train)
            list_score_validation.append(score_validacion.round(4))
            list_score_train.append(score_train.round(4))
        else:
            score_validacion = f1_score(y_test, y_pred, average='weighted')
            score_train = f1_score(y_train, y_pred_train, average='weighted')
            list_score_validation.append(score_validacion.round(4))
            list_score_train.append(score_train.round(4))
        
        list_auc_score.append(roc_auc_score_1.round(4))
        list_best_params.append(search.best_params_)
        index_names.append(key)

        dict_modelos[key] = search
        

    data = pd.DataFrame(data={
        'best_params': list_best_params,
        f'{scoring}_validacion' : list_score_validation,
        f'{scoring}_train': list_score_train,
        'auc_score': list_auc_score
        }
        ,        
         index=index_names)

    return data, dict_modelos, X_test, y_test


def model_logit(dataframe, vector_objetivo, features):
    '''
    Función que imprime un modelo descriptivo, seleccionando el vector objetivo y las variables a modelar.

    Parámetros
    ----------
    dataframe: Dataframe pandas.
    vector_objetivo: Vector objetivo a modelar.
    features: Variables independientes que se usarán en el modelado.    

    Returns
    -------
    modelo_descriptivo: Modelo econométrico.
    '''

    X = dataframe.loc[:,features]
    X = X.drop(columns=vector_objetivo)
    y = dataframe[vector_objetivo]
    
    X_constant = sm.add_constant(X)

    modelo_descriptivo = sm.Logit(y, X_constant).fit()

    print(modelo_descriptivo.summary())

    return modelo_descriptivo


def countplot_graph(dataframe, vector):
    '''
    Función que permite graficar countplots.

    Parámetros
    ----------
    dataframe: Dataframe de pandas.
    vector: Variable a gráficar.    
    '''
    ax = sns.countplot(data=dataframe, x=vector)
    ax.bar_label(ax.containers[0])
    ax.set_title(f'Countplot Variable Objetivo {vector}')
    ax.set_ylabel('Frecuencia')

def corr_vector(dataframe, vector, umbral=0.1):   
    '''
    Función que retorna e imprime una lista de variables junto con las correlaciones asociadas al vector deseado, estableciendo un umbral.

    Parámetros
    ----------
    dataframe: Dataframe de pandas.
    vector: Vector al cuál se le calcularán las correlaciones con las variables independientes.

    Returns
    -------
    list_corr: Lista de correlaciones obtenidas y definidas por el umbral.
    '''
    df_corr = pd.DataFrame(dataframe.corr()[[vector]].abs().sort_values(by = vector, ascending = False))    

    print(df_corr.loc[df_corr[vector]>umbral])

    list_corr = list(df_corr.loc[df_corr[vector]>umbral].index)    

    return list_corr

def regplot(dataframe, vector, feature_list):
    '''
    Función que permite mostrar gráficos regplot.

    Parámetros
    ----------
    dataframe: Dataframe de pandas.
    vector: Vector objetivo.
    feature_list: Lista de variables a contrastar con el vector.
    '''

    plt.figure(figsize=(12,9))

    for i, column in enumerate(feature_list):
        plt.subplot(5, 4, i+1)
        sns.regplot(data=dataframe, y=vector, x=column)
        plt.tight_layout()
        plt.title(column)

def invlogit(x):
    '''
    Función logistica inversa.

    Parámetros
    ----------
    x: Número Entero.

    Returns
    -------
    Retorna la logistica inversa.

    '''

    return 1 / (1 + np.exp(-x))

def coef_prob_logit(serie):
    '''
    Función que imprime un valor aproximado de cuanto el coeficiente está afectando la probabilidad de ocurrencia de la Clase a evaluar en el modelo econométrico,
    dividiendo el coeficiente por 4.

    Parámetros
    ----------
    serie: Serie Pandas con los coeficientes del sumario del modelo descriptivo.

    Print
    -------
    Imprime el porcentaje con el cual está afectando a la ocurrencia de la clase, además indica si disminuye o aumenta.

    '''
    for index, item in serie.iteritems():    
        if item > 0:
            print(f'Variable {index} aumenta en {(item/4)*100:.2f}% aproximadamente la probabilidad de tener diabetes')
        else:
            print(f'Variable {index} disminuye en {(item/4)*100:.2f}% aproximadamente la probabilidad de tener diabetes')


def cat_num_rate_analysis(df):
    '''
    Función que permite evaluar la calidad de los datos, devolviendo una tabla que indica la cantidad de valores únicos y su tipo de dato.

    Parámetros
    ----------
    df: Dataframe Pandas.
    '''
    cat_num_rate = df.apply(lambda col: (len(col.unique()), len(col),col.dtype ,  col.unique()))
    cmr = pd.DataFrame(cat_num_rate.T)
    cmr.columns=["len of unique", "len of data", "col type", "unique of col"]
    max_rows = pd.get_option('display.max_rows')
    max_width = pd.get_option('display.max_colwidth')
    pd.set_option('display.max_colwidth', 150)
    pd.set_option('display.max_rows', None)
    display(cmr.sort_values(by="len of unique",ascending=False))
    pd.set_option('display.max_rows', max_rows)
    pd.set_option('display.max_colwidth', max_width)

def plot_roc_curve(X_test, y_test, modelo, title, auc):
    '''
    Función que gráfica la curva ROC de un modelo ingresado.

    Parámetros
    ----------
    X_test: Conjunto de validación donde se predice.
    y_test: Respuestas correctas del conjunto de validación.
    modelo: Modelo usado para redecir en el Conjunto de validación.
    title: Título del gráfico.
    '''    

    yhat_pr_roc = modelo.predict_proba(X_test)[:,1]

    false_positive, true_positive, threshold = roc_curve(y_test, yhat_pr_roc)

    plt.plot(false_positive, true_positive, lw=1)
    plt.plot([0, 1], linestyle='--', lw=1, color='tomato')
    plt.ylabel('Verdaderos Positivos')
    plt.xlabel('Falsos Positivos')
    plt.title(f'{title}')
    plt.annotate(f"Área bajo la Curva: {auc}", xy=(1, 0), ha='right')

def cross_plot_2(data, barra, variable, categorias, size=(8,5), xlim=(0,1), ylim=(0.1,0.8), titulo = None, order=1, medias=0, directorio='./'):

    '''
    Función que gráfica la curva ROC de un modelo ingresado.
    
    Parámetros
    ----------
    X_test: Conjunto de validación donde se predice.
    y_test: Respuestas correctas del conjunto de validación.
    modelo: Modelo usado para redecir en el Conjunto de validación.
    title: Título del gráfico.
    '''  

    fig, ax1 = plt.subplots(figsize=size)
    ax2 = ax1.twinx()
    if order==1:
        data = data.sort_values(barra).reset_index(drop=True)
    data[barra].plot(kind='bar', color='b', ax=ax1, label=barra)
    try:
        for v in variable:
            data[v].plot(kind='line', marker='d', ax=ax2, label=v)
    except:
        data[variable].plot(kind='line',color='r', marker='d', ax=ax2, label=variable)
    ax1.yaxis.tick_left()
    ax2.yaxis.tick_right()
    ticks = data[categorias]
    plt.xticks(np.arange(ticks.unique().shape[0]),ticks)
    plt.xlim(xlim)
    plt.ylim(ylim)
    if medias==1:
        cc = ['red','green','gray']
        j=0
        try:
            for v in variable:
                plt.axhline(data[v].mean(), label=v, color=cc[j])
                j=j+1
        except:
            plt.axhline(data[variable].mean(), label=variable, color=cc[j])
    plt.title(titulo)# , fontdict=font)
    ax1.set_xlabel(categorias)#, fontdict=font1)
    ax1.set_ylabel(barra)#, fontdict=font1)
    ax2.set_ylabel(variable)#, fontdict=font1)
    plt.legend()
    plt.grid()
    fig.tight_layout()  
    #plt.savefig(directorio+'Crossplot {}.png'.format(titulo), dpi=100)
    plt.show()

def crossplot(dataframe, list_corr, vector_objetivo, xlim=(0, 1)):

    '''
    Función que realiza un crossplot entre dos variables categóricas.

    Parámetros
    ----------
    dataframe: Pandas dataframe.
    list_corr: Lista de variables a cruzar con el vector objetivo.
    vector_objetivo: Vector objetivo.
    xlim: Límites del eje X del gráfico.
    '''  

    X = dataframe.copy() 
    
    list_corr.append(vector_objetivo)

    X = X.loc[:,list_corr]

    for col in list_corr:    
        X[col] = X[col].astype('O')
    
    calidad = pd.DataFrame({'tipo':X.dtypes.values}, index=X.dtypes.index)   
    var_obj = vector_objetivo
    
    for col in calidad.loc[calidad.tipo == 'object'].index:
        if col == vector_objetivo:
            continue;
        else:
            df_group = X.groupby(col).agg({var_obj:['count', 'mean']})    
            df_group.columns = ['cantidad', 'media']
            df_group = df_group.reset_index()

        titulo = f'{col} vs {var_obj}'

        cross_plot_2(df_group, 'cantidad', 'media', col,xlim=xlim, ylim=(0, 1), titulo=titulo, order=0) 

def imc_to_cat(imc):
    '''
    Función que clasifica el IMC.

    Parámetros
    ----------
    imc: IMC del individuo

    Retorna
    -------
    clasificacion(string): Clasficicación del individuo
    '''  

    clasificacion = ""

    if imc < 18.5:
            clasificacion = 'Bajo peso'
    elif imc < 25:
            clasificacion = 'Adecuado'
    elif imc < 30:
            clasificacion = 'Sobrepeso'
    elif imc < 35:
            clasificacion = 'Obesidad Grado I'
    elif imc < 40:
            clasificacion = 'Obesidad Grado II'
    else:
            clasificacion = 'Obesidad Grado III'

    return clasificacion

def check_nulls(dataframe):
    '''
    Función que comprueba la existencia de datos nulos en un Dataframe de Pandas.

    Parámetros
    ----------
    dataframe: Dataframe pandas.

    Retorna
    -------
    DataFrame de pandas con la cantidad de nulos por columna
    '''  
    dict_col = {}

    for col in dataframe:    
        if len(dataframe[col].isna().value_counts()) == 1:
            dict_col[col] = 'Sin nulos'
        else:
            dict_col[col] = dataframe[col].isna().value_counts()[True]

    return pd.DataFrame({'Cantidad nulos': dict_col.values()}, index=dict_col.keys())

def age_cat():
    '''
    Función que crea un diccionario con la categoría de edades y rangos.   

    Retorna
    -------
    cat_age: Diccionario con la categoría de edades y su rango.
    list: Lista de valores de las categorías.
    '''  
    cat_age = {}
    start = 18
    categoria = ""

    for i in range(1, 14):
        categoria = ""
        if i == 1:
            categoria = "Edad "+str(start)+" - "
            start+=6
            categoria = categoria + str(start)
        elif i == 13:
            categoria = "Edad 80 o mayor"
        else:
            start+=1
            categoria = "Edad "+str(start)+" - "
            start+=4
            categoria = categoria + str(start)        
        cat_age[i] = categoria
    
    return cat_age, list(cat_age.values())

def age_to_cat(number):
    '''
    Función que categoriza un número en rango de edades.

    Parámetros
    ----------

    Retorna
    -------
    value: Valor categorizado.
    '''  
    
    dict_age, _ = age_cat()

    for key, value in dict_age.items():
        if number == key:
            return value