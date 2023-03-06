import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor


def replace_nullstring(dataframe, vector, stringnull):

    """
    Función que reemplaza valores Strings por Nulos de Numpy.

    Parámetros
    ----------
    dataframe: Dataframe pandas.
    vector: Nombre del Vector Serie Pandas.
    stringnull: String a reemplazar por Numpy

    Returns
    -------
    dataframe
        dataframe[vector]: Vector serie pandas con valores reemplazados.

    """

    dataframe[vector] = np.where(dataframe[vector]==stringnull,np.nan,dataframe[vector])
    
    return dataframe[vector]

def replace_nullstring_list(dataframe, vector, listastrings):
    """
    Función que reemplaza valores Strings por Nulos de Numpy ingresando una lista de valores Strings.

    Parámetros
    ----------
    dataframe: Dataframe pandas.
    vector: Nombre del Vector Serie Pandas.
    listastrings: Lista de Strings a reemplazar por valores Numpy.

    Returns
    -------
    dataframe
        dataframe[vector]: Vector serie pandas con valores reemplazados.

    """
    for item in listastrings:
        dataframe[vector] = np.where(dataframe[vector] == item, np.nan, dataframe[vector])
    
    return dataframe[vector]

def obs_perdidas(dataframe, var, print_list = False):

    """
    Función que retorna la cantidad de datos perdidos de un vector. Se puede imprimir la lista al ingresar True como parámetro en print_list

    Parámetros
    ----------
    dataframe : Dataframe pandas.
    var: Nombre del Vector Serie Pandas

    Returns    
    -------
    float        
        casos_perdidos: Cantidad de casos perdidos.
        casos_perdidos_porc: Cantidad de casos perdidos en porcentaje.
    """

    df = dataframe

    if len(df[var].isna().value_counts()) == 1:
        casos_perdidos = 0
    else:
        casos_perdidos = df[var].isna().value_counts()[True]
        
    casos_perdidos_porc = (casos_perdidos/len(df)*100)
    lista_casos = df[df[var].isna()][var]

    if print_list == True:
        print(f'Lista de casos perdidos en {var}')
        print(lista_casos)
        return casos_perdidos, casos_perdidos_porc
    else:
        print(f'Cantidad de casos perdidos en {var}: {casos_perdidos}, {casos_perdidos_porc:.2f}%')        
        return casos_perdidos, casos_perdidos_porc

def change_nomenclature(dataframe, column, diccionario):

    """
    Función que reemplaza la nomenclatura de datos dentro de un Vector Serie Pandas a través de un diccionario ingresado.

    Parámetros
    ----------
    dataframe: Dataframe pandas.
    column: Nombre del Vector Serie Pandas.
    diccionario: Diccionario que contiene la información para cambiar la nomenclatura de datos.

    Returns
    -------
    dataframe
        dataframe[vector]: Vector serie pandas con valores reemplazados.

    """

    for key, item in diccionario.items():
        dataframe[column] = np.where(np.isin(dataframe[column], item), key, dataframe[column])

    return dataframe[column]

def countplot_sns(data, x, title):

    """
    Función que crea un gráfico countplot de seaborn.

    Parámetros
    ----------
    data: Dataframe pandas.
    x: Nombre del Vector Serie Pandas.
    title: Título del gráfico

    Returns
    -------   

    """

    sns.countplot(data=data, x=data[x])
    plt.title(title)
    plt.xlabel('Categorías')


def histplot_sns(data, x, title):

    """
    Función que crea un gráfico histplot de seaborn.

    Parámetros
    ----------
    data: Dataframe pandas.
    x: Nombre del Vector Serie Pandas.
    title: Título del gráfico

    Returns
    -------   

    """

    plt.figure(figsize=(10,8))
    sns.histplot(data=data, x=data[x], kde=True)
    plt.axvline(x = data[x].mean(), color='tomato')    
    plt.title(title)

def histplot_log(data, x, title):    

    """
    Función que crea un gráfico histplot de seaborn, pero aplicando logaritmo.

    Parámetros
    ----------
    data: Dataframe pandas.
    x: Nombre del Vector Serie Pandas.
    title: Título del gráfico

    Returns
    -------   

    """   

    df_filtered = data[data[x]>0]

    data_log = np.log(df_filtered[x])    
    
    plt.figure(figsize=(10,8))
    sns.histplot(data=data, x=data_log, kde=True)
    plt.axvline(x = data_log.mean(), color='tomato')
    plt.title(title);

def number_in_marks_to_number(dataframe, vector):

    """
    Función que quita doble comillas a datos dentro de un Vector Serie Pandas.

    Parámetros
    ----------
    dataframe: Dataframe pandas.
    vector: Nombre del Vector Serie Pandas.   

    Returns
    -------
    dataframe
        dataframe[vector]: Vector Serie Pandas con datos sin comillas.

    """

    dataframe[vector] = dataframe[vector].str.replace('"','')

    dataframe[vector] = dataframe[vector].astype('float64')

    return dataframe[vector]

def categorical_to_number(df, vector):

    """
    Función que transforma categorías en números, dejando la mayoritaria en el valor más alto y la minoritaria en el valor más bajo(0).

    Parámetros
    ----------
    df: Dataframe pandas.
    vector: Nombre del Vector Serie Pandas.   

    Returns
    -------
    Vector Serie Pandas
        df[vector]: Vector Serie Pandas con categorías numerizadas de menor a mayor.   

    """

    dict_map = {}

    for i, j in enumerate(list(df[vector].value_counts().index.to_list())[::-1]):
        dict_map[j]=i

    df[vector] = df[vector].map(dict_map)

    return df[vector]

def binarize_categorical(dataframe, vector):

    """
    Función que binariza categorías dejando la mayoritaria como número 0 y categoría minoritaria como 1.

    Parámetros
    ----------
    dataframe: Dataframe pandas.
    vector: Nombre del Vector Serie Pandas.   

    Returns
    -------
    Vector Serie Pandas
        df[vector]: Vector Serie Pandas con categorías numerizadas de forma binaria.
    """
    dict_map = {}

    for i, j in enumerate(dataframe[vector].value_counts().index.to_list()):
        dict_map[j]=i

    dataframe[vector] = dataframe[vector].map(dict_map)

    return dataframe[vector]


def VIF(predictors , vars_to_ignore = None):

    """
    Función entrega el variance inflation factor de predictores. Opcionalmente se puede ignorar columnas ingresando en el parámetro vars_to_ignore.

    Parámetros
    ----------
    predictors: Dataframe Pandas.
    vars_to_ignore: Nombre del Vector Serie Pandas a ignorar.   

    Returns
    -------
    float
        output: Nombre del vector
        output.index. Valor del vector    
    """

    x = predictors

    if vars_to_ignore is not None:
        x = x.drop(columns= vars_to_ignore)

    output = pd.Series([variance_inflation_factor(x.to_numpy(), col) for col in range(x.shape[1])], index=x.columns)

    return output, output.index

def model_metrics(model):

    """
    Función que imprime las bondades de ajuste de modelos de Regresión Lineal.

    Parámetros
    ----------
    model: Modelo de Regresión lineal OLS.

    Print
    -------
    R cuadrado del modelo
    AIC del modelo
    BIC del modelo
    Conditional Number del modelo
    """

    model.rsquared_adj
    model.aic
    model.bic
    model.condition_number

    print(f'R cuadrado ajustado: {model.rsquared_adj:.2f}')
    print(f'Model AIC: {model.aic:.2f}')
    print(f'Model BIC: {model.bic:.2f}')
    print(f'Model Cond. no: {model.condition_number:.2f}')
    
def info_utilidades_desafio2(x):
    """
    replace_nullstring
    Función que reemplaza valores Strings por Nulos de Numpy.

    replace_nullstring_list    
    Función que reemplaza valores Strings por Nulos de Numpy ingresando una lista de valores Strings.

    obs_perdidas
    Función que retorna la cantidad de datos perdidos de un vector. Se puede imprimir la lista al ingresar True como parámetro en print_list

    change_nomenclature    
    Función que reemplaza la nomenclatura de datos dentro de un Vector Serie Pandas a través de un diccionario ingresado.

    countplot_sns
    Función que crea un gráfico countplot de seaborn.

    histplot_sns
    Función que crea un gráfico histplot de seaborn.

    histplot_log
    Función que crea un gráfico histplot de seaborn, pero aplicando logaritmo.

    number_in_marks_to_number
    Función que quita doble comillas a datos dentro de un Vector Serie Pandas.

    categorical_to_number
    Función que transforma categorías en números, dejando la mayoritaria en el valor más alto y la minoritaria en el valor más bajo(0).

    binarize_categorical
    Función que binariza categorías dejando la mayoritaria como número 0 y categoría minoritaria como 1.

    VIF
    Función entrega el variance inflation factor de predictores. Opcionalmente se puede ignorar columnas ingresando en el parámetro vars_to_ignore.

    model_metrics
    Función que imprime las bondades de ajuste de modelos de Regresión Lineal.
    """

def info_utilidades_desafio1(x):
    """
    obs_perdidas
    Función que retorna la cantidad de datos perdidos de un vector. Se puede imprimir la lista al ingresar True como parámetro en print_list    

    countplot_sns
    Función que crea un gráfico countplot de seaborn.

    histplot_sns
    Función que crea un gráfico histplot de seaborn.

    histplot_log
    Función que crea un gráfico histplot de seaborn, pero aplicando logaritmo.

    change_nomenclature    
    Función que reemplaza la nomenclatura de datos dentro de un Vector Serie Pandas a través de un diccionario ingresado.
    """
