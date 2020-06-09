#!/usr/bin/env python
# coding: utf-8

# # Desafio 1
# 
# Para esse desafio, vamos trabalhar com o data set [Black Friday](https://www.kaggle.com/mehdidag/black-friday), que reúne dados sobre transações de compras em uma loja de varejo.
# 
# Vamos utilizá-lo para praticar a exploração de data sets utilizando pandas. Você pode fazer toda análise neste mesmo notebook, mas as resposta devem estar nos locais indicados.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Set up_ da análise

# In[1]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# In[2]:


black_friday = pd.read_csv("black_friday.csv")
df = black_friday


# ## Inicie sua análise a partir daqui

# ## Questão 1
# 
# Quantas observações e quantas colunas há no dataset? Responda no formato de uma tuple `(n_observacoes, n_colunas)`.

# In[3]:


def q1():
    # Retorne aqui o resultado da questão 1.
    return df.shape


# ## Questão 2
# 
# Há quantas mulheres com idade entre 26 e 35 anos no dataset? Responda como um único escalar.

# In[4]:


def q2():
    # Retorne aqui o resultado da questão 2.
    women_at_certain_age = df[(df['Gender']=='F') & (df['Age']=='26-35')]
    return int(women_at_certain_age['User_ID'].count())   


# ## Questão 3
# 
# Quantos usuários únicos há no dataset? Responda como um único escalar.

# In[5]:


def q3():
    # Retorne aqui o resultado da questão 3.
    return df['User_ID'].unique().size
    


# ## Questão 4
# 
# Quantos tipos de dados diferentes existem no dataset? Responda como um único escalar.

# In[6]:


def q4():
    # Retorne aqui o resultado da questão 4.
    return df.dtypes.unique().size


# ## Questão 5
# 
# Qual porcentagem dos registros possui ao menos um valor null (`None`, `ǸaN` etc)? Responda como um único escalar entre 0 e 1.

# In[7]:


def q5():
    # Retorne aqui o resultado da questão 5.
    return float(df.isna().any(axis=1).mean())


# ## Questão 6
# 
# Quantos valores null existem na variável (coluna) com o maior número de null? Responda como um único escalar.

# In[8]:


def q6():
    # Retorne aqui o resultado da questão 6.
    # Retorna o valor máximo da soma de valores null por coluna
    return int(df.isna().sum().max())


# ## Questão 7
# 
# Qual o valor mais frequente (sem contar nulls) em `Product_Category_3`? Responda como um único escalar.

# In[9]:


def q7():
    # Retorne aqui o resultado da questão 7.
    return df['Product_Category_3'].value_counts().idxmax()


# ## Questão 8
# 
# Qual a nova média da variável (coluna) `Purchase` após sua normalização? Responda como um único escalar.

# In[10]:


def q8():
    # Retorne aqui o resultado da questão 8.
    
    # Criando o objeto normalizador
    min_max_scaler = MinMaxScaler()
    
    # Aplicando a normalização na coluna Purchase
    x_norm = min_max_scaler.fit_transform(df[['Purchase']])

    # Retornando a média dos valores normalizados
    return float(x_norm.mean())


# ## Questão 9
# 
# Quantas ocorrências entre -1 e 1 inclusive existem da variáel `Purchase` após sua padronização? Responda como um único escalar.

# In[11]:


def q9():
    # Retorne aqui o resultado da questão 9.
    
    # Criando o objeto padronizador
    standard_scaler = StandardScaler()
    
    # Aplicando a padronização na coluna Purchase
    x_std = standard_scaler.fit_transform(df[['Purchase']])
    
    # Criando uma lista que indica se o valor padronizado está ou não no intervalo [-1,1]
    x_in_range = [1 if (item>=-1 and item<=1) else 0 for item in x_std ]

    # Retornando o número de ocorrências no intervalo [-1,1]
    return sum(x_in_range)


# ## Questão 10
# 
# Podemos afirmar que se uma observação é null em `Product_Category_2` ela também o é em `Product_Category_3`? Responda com um bool (`True`, `False`).

# In[12]:


def q10():
    # Retorne aqui o resultado da questão 10.
        
    # Verificando que valores são null em cada categoria
    cat2 = df['Product_Category_2'].isna()
    cat3 = df['Product_Category_3'].isna()
    
    # Se uma observação null em `Product_Category_2` significa uma observação null em `Product_Category_3`,
    # então o conjunto `Product_Category_3` está contido no conjunto `Product_Category_2`. 
    # Assim, a operação lógica (cat2 and cat3) resulta em um conjunto de valores igual a cat2.
    
    cat23 = cat2 & cat3
    
    return bool((cat23 == cat2).all())

