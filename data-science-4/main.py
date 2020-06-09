#!/usr/bin/env python
# coding: utf-8

# # Desafio 6
# 
# Neste desafio, vamos praticar _feature engineering_, um dos processos mais importantes e trabalhosos de ML. Utilizaremos o _data set_ [Countries of the world](https://www.kaggle.com/fernandol/countries-of-the-world), que contém dados sobre os 227 países do mundo com informações sobre tamanho da população, área, imigração e setores de produção.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[29]:


import pandas as pd
import numpy as np
import seaborn as sns
import sklearn as sk
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.compose import make_column_selector as selector
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


# In[2]:


# Algumas configurações para o matplotlib.
# %matplotlib inline

from IPython.core.pylabtools import figsize


figsize(12, 8)

sns.set()


# In[3]:


countries = pd.read_csv("countries.csv")


# In[4]:


new_column_names = [
    "Country", "Region", "Population", "Area", "Pop_density", "Coastline_ratio",
    "Net_migration", "Infant_mortality", "GDP", "Literacy", "Phones_per_1000",
    "Arable", "Crops", "Other", "Climate", "Birthrate", "Deathrate", "Agriculture",
    "Industry", "Service"
]

countries.columns = new_column_names

countries.head(5)


# ## Observações
# 
# Esse _data set_ ainda precisa de alguns ajustes iniciais. Primeiro, note que as variáveis numéricas estão usando vírgula como separador decimal e estão codificadas como strings. Corrija isso antes de continuar: transforme essas variáveis em numéricas adequadamente.
# 
# Além disso, as variáveis `Country` e `Region` possuem espaços a mais no começo e no final da string. Você pode utilizar o método `str.strip()` para remover esses espaços.

# ## Inicia sua análise a partir daqui

# In[5]:


# Sua análise começa aqui.

countries['Country'][0]


# In[6]:


df = countries

# Substituindo as vígulas por pontos
df = df.apply(lambda x: x.replace(',','.', regex=True))

df.head()


# In[7]:


# Transformando strings que contém números e vairáveis numéricas
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='ignore') 
df.head()


# In[8]:


# Removendo os espaços a mais das variáveis Country e Region 
df['Country'] = df['Country'].str.strip()
df['Region'] = df['Region'].str.strip()

df['Country'][0]


# ## Questão 1
# 
# Quais são as regiões (variável `Region`) presentes no _data set_? Retorne uma lista com as regiões únicas do _data set_ com os espaços à frente e atrás da string removidos (mas mantenha pontuação: ponto, hífen etc) e ordenadas em ordem alfabética.

# In[9]:


def q1():
    # Retorne aqui o resultado da questão 1.
    return sorted(df['Region'].unique())


# In[10]:


q1()


# ## Questão 2
# 
# Discretizando a variável `Pop_density` em 10 intervalos com `KBinsDiscretizer`, seguindo o encode `ordinal` e estratégia `quantile`, quantos países se encontram acima do 90º percentil? Responda como um único escalar inteiro.

# In[11]:


def q2():
    # Retorne aqui o resultado da questão 2.
        
    # Discretizando a variável em 10 bins    
    est = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
    df['Pop_density'] = est.fit_transform(df[['Pop_density']])

    # Retornando o número de países no último bin, que são os países acima do 90º percentil
    return int(df['Pop_density'][df['Pop_density'] == 9.0].count())

q2()


# # Questão 3
# 
# Se codificarmos as variáveis `Region` e `Climate` usando _one-hot encoding_, quantos novos atributos seriam criados? Responda como um único escalar.

# In[12]:


def q3():
    # Retorne aqui o resultado da questão 3.
    return df['Region'].nunique(dropna=False) + df['Climate'].nunique(dropna=False)

q3()


# ## Questão 4
# 
# Aplique o seguinte _pipeline_:
# 
# 1. Preencha as variáveis do tipo `int64` e `float64` com suas respectivas medianas.
# 2. Padronize essas variáveis.
# 
# Após aplicado o _pipeline_ descrito acima aos dados (somente nas variáveis dos tipos especificados), aplique o mesmo _pipeline_ (ou `ColumnTransformer`) ao dado abaixo. Qual o valor da variável `Arable` após o _pipeline_? Responda como um único float arredondado para três casas decimais.

# In[13]:


test_country = [
    'Test Country', 'NEAR EAST', -0.19032480757326514,
    -0.3232636124824411, -0.04421734470810142, -0.27528113360605316,
    0.13255850810281325, -0.8054845935643491, 1.0119784924248225,
    0.6189182532646624, 1.0074863283776458, 0.20239896852403538,
    -0.043678728558593366, -0.13929748680369286, 1.3163604645710438,
    -0.3699637766938669, -0.6149300604558857, -0.854369594993175,
    0.263445277972641, 0.5712416961268142
]

df_test = pd.DataFrame (data = [test_country], columns = df.columns.tolist())


# In[16]:


def q4():
    # Retorne aqui o resultado da questão 4.
    X = df.drop(['Region','Country'],axis=1)

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, selector(dtype_exclude="category"))
    ])

    preprocessor.fit(X)
    res = preprocessor.transform(df_test.drop(['Region','Country'],axis=1)).tolist()
    return round(res[0][9],3)

q4()


# ## Questão 5
# 
# Descubra o número de _outliers_ da variável `Net_migration` segundo o método do _boxplot_, ou seja, usando a lógica:
# 
# $$x \notin [Q1 - 1.5 \times \text{IQR}, Q3 + 1.5 \times \text{IQR}] \Rightarrow x \text{ é outlier}$$
# 
# que se encontram no grupo inferior e no grupo superior.
# 
# Você deveria remover da análise as observações consideradas _outliers_ segundo esse método? Responda como uma tupla de três elementos `(outliers_abaixo, outliers_acima, removeria?)` ((int, int, bool)).

# In[17]:


sns.boxplot(x=df['Net_migration'])


# In[18]:


sns.distplot(df['Net_migration'].fillna(0));


# In[19]:


def q5():
    # Retorne aqui o resultado da questão 4.
    
    Q1 = df['Net_migration'].quantile(0.25)
    Q3 = df['Net_migration'].quantile(0.75)
    IQR = Q3 - Q1  
    
    # Quantidade de outliers abaixo
    outlier_abaixo = df['Net_migration'][df['Net_migration'] < (Q1 - 1.5 * IQR)].count()
    
    # Quantidade de outliers abaixo
    outlier_acima = df['Net_migration'][df['Net_migration'] > (Q3 + 1.5 *IQR)].count()    
    
    # Retornando o número de outliers, não acho que os outliers devem ser eliminados pois representam uma fração alta do total de dados (aprox 22%).
    return (outlier_abaixo, outlier_acima, False)

q5()


# ## Questão 6
# Para as questões 6 e 7 utilize a biblioteca `fetch_20newsgroups` de datasets de test do `sklearn`
# 
# Considere carregar as seguintes categorias e o dataset `newsgroups`:
# 
# ```
# categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
# newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)
# ```
# 
# 
# Aplique `CountVectorizer` ao _data set_ `newsgroups` e descubra o número de vezes que a palavra _phone_ aparece no corpus. Responda como um único escalar.

# In[23]:


categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)


# In[25]:


# Obtendo a relação de palavras únicas encontradoas no corpus
count_vect = CountVectorizer()

# Calculando a matriz de ocorrências de cada palavra
newsgroup_counts = count_vect.fit_transform(newsgroup.data)

# Obtendo o indice da palavra desejada "phone" na relação de palavras únicas
word_index = count_vect.vocabulary_.get(u'phone')


# In[27]:


def q6():
    # Retorne aqui o resultado da questão 4.
    
    # Contando o número de ocorrencias de cada palavra a partir da matriz de ocorrências
    count_list = newsgroup_counts.sum(axis=0)
    
    # Selecionando o número de ocorrencias da palavra desejada
    word_count = count_list[0,word_index] 
    
    return int(word_count)

q6()


# ## Questão 7
# 
# Aplique `TfidfVectorizer` ao _data set_ `newsgroups` e descubra o TF-IDF da palavra _phone_. Responda como um único escalar arredondado para três casas decimais.

# In[31]:


def q7():
    # Retorne aqui o resultado da questão 4.
    
    # Calculando TF-IDF de newsgroup
    tfidf_transformer = TfidfVectorizer (use_idf=True)
    newsgroup_counts_tfidf = tfidf_transformer.fit_transform(newsgroup.data)
    
    # Calculando o TF-IDF de cada palavra
    tdidf_list = newsgroup_counts_tfidf.sum(axis=0)
    
    # Selecionando apenas a frequnecia da palavra desejada 
    word_tdidf = tdidf_list[0,word_index] 
    
    return float(word_tdidf.round(3))

q7()

