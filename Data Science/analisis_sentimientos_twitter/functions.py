import numpy as np
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import string
from wordcloud import WordCloud
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

np.random.seed(5)

stop_words = set(stopwords.words('english'))

def random_neutral(word): # Función para randomizar sentimientos neutrales

    """
    Función que randomiza observación Neutral entre Clase 0 y 1

    Parámetros
    ----------
    word: Palabra neutral

    Returns
    -------
    String
        word: Clase 0 o 1
    """

    if word == 'neutral':
        return np.random.choice([0, 1])
    return word

def random_neutral_string(word): # Función para randomizar sentimientos neutrales
    """
    Función que randomiza observación Neutral entre Clase 0 y 1

    Parámetros
    ----------
    word: Palabra neutral

    Returns
    -------
    String
        word: Clase 0 o 1
    """    
    if word == 'neutral':
        return np.random.choice(['positiva','negativa'])
    return word

def recod_emotions(df, vector, list, bin_number):

    """
    Función que recodifica emociones mediante una lista y la binarización deseada

    Parámetros
    ----------
    df: DataFrame Pandas.
    vector: Vector Pandas.
    list: Lista de recodificación.
    bin_number: Binarización deseada.

    Returns
    -------
    String
        df[vector]: Vector recodificado
    """

    df[vector] = np.where(np.isin(df[vector], list), bin_number, df[vector])
    return df[vector]

def countplot_sns(dataframe, vector, title='', xlabel=''):
    ax = sns.countplot(data=dataframe, x=vector)
    ax.bar_label(ax.containers[0])
    plt.title(title)
    plt.ylabel('Frecuencia')
    plt.xlabel(xlabel)

def wordcloud_graph(dataframe, col_text, col_sent, sent):
    wordcloud = WordCloud(width=800, height=500, random_state = 10)
    positive_words = " ".join([s for s in dataframe[col_text][dataframe[col_sent]==sent]])

    # Plot del gráfico
    wordcloud = WordCloud(max_words=500, width=1600, height=800).generate(positive_words)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show

def plot_freq_words(df, vector, title, number_words):

    count_vectorizer=CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words=stop_words)

    count_vectorizer_fit = count_vectorizer.fit_transform(df[vector])

    words = count_vectorizer.get_feature_names_out()

    words_freq = count_vectorizer_fit.toarray().sum(axis=0)

    most_freq_words = pd.DataFrame(words_freq, words).sort_values(by=[0], ascending=False)[0:number_words]

    most_freq_words.rename({0:'words'}, axis='columns').plot(kind='bar')

    plt.title(title)

class TweetsPreprocessing(BaseEstimator, TransformerMixin):

    def __init__(self) :
        pass

    def preprocessing_text(self, serie):
        def remove_patter(input_txt, pattern):
            r = re.findall(pattern, input_txt)
            for word in r:
                input_txt = re.sub(word, "", input_txt)

            return input_txt

        def remove_patter_https(text):
            text = re.sub(r'\@w+|\#', '', text)
            text = re.sub(r"https\S+|www\S+https\S+", '', text, flags=re.MULTILINE)
            text = re.sub(r'[^\w\s]', '', text)          
            return text

        serie = serie.str.lower()

        # Remover https
        serie = np.vectorize(remove_patter_https)(serie)
        serie = pd.Series(serie)

        # Remover @user
        serie = np.vectorize(remove_patter)(serie, "@[\w]*")
        serie = pd.Series(serie)        

        # Remover caracteres especiales
        serie = serie.str.replace("[^a-zA-Z#]", " ")   

        # Remover palabras cortas
        serie = serie.apply(lambda x: " ".join([w for w in x.split() if len(w)>3]))

        ## Eliminar palabras comunes
        stop_words = set(stopwords.words('english'))
        #serie = pd.Series([w for w in serie if not w in stop_words])        
        serie = serie.apply(lambda x: " ".join([w for w in x.split() if not w in stop_words]))

        # tokens
        tokenized_tweet = serie.apply(lambda x: x.split())

        # Lemm
        lemmatizer = WordNetLemmatizer()

        tokenized_tweet = tokenized_tweet.apply(lambda sentence: [lemmatizer.lemmatize(word) for word in sentence])

        for i in range(len(tokenized_tweet)):
            tokenized_tweet[i] = " ".join(tokenized_tweet[i])

        serie = tokenized_tweet

        return serie
###### 
    def fit(self, X, y = None):       
        return self
        
    def transform(self, X, y = None):
        X_ = X.copy()
        X_= self.preprocessing_text(X_)

        return X_

def gridsearch_train(dict, X_train, X_test, y_train, y_test):

    counter = 0

    best_score_list = []    
    best_params_list = []
    score_validation = []
    index_names = []

    dict_modelos = {}
    
    for key, item in dict.items():    
        
        counter +=1

        pipe = Pipeline([
                ('tweet_pre', TweetsPreprocessing()),
                ('cv', CountVectorizer(max_df=0.90, min_df=2, max_features=150, stop_words=stop_words)),
                ('model', item[0])
            ])
        
        search = GridSearchCV(pipe, item[1], cv = 5, scoring = 'accuracy', n_jobs = -1, return_train_score=True)
        search.fit(X_train, y_train)

        y_pred = search.best_estimator_.predict(X_test)
        score_validacion = accuracy_score(y_test, y_pred)
                
        best_score_list.append(search.best_score_.round(4))
        best_params_list.append(search.best_params_)
        score_validation.append(score_validacion)
        index_names.append(key)

        dict_modelos[key] = search

    data = pd.DataFrame(data={'best_score':best_score_list,'best_params': best_params_list, 'acc_validacion': score_validation}, index=index_names)    
    return data, dict_modelos