#!/usr/bin/env python
# coding: utf-8

# # Desafio 3
# 
# Neste desafio, iremos praticar nossos conhecimentos sobre distribuições de probabilidade. Para isso,
# dividiremos este desafio em duas partes:
#     
# 1. A primeira parte contará com 3 questões sobre um *data set* artificial com dados de uma amostra normal e
#     uma binomial.
# 2. A segunda parte será sobre a análise da distribuição de uma variável do _data set_ [Pulsar Star](https://archive.ics.uci.edu/ml/datasets/HTRU2), contendo 2 questões.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sct
import seaborn as sns
from statsmodels.distributions.empirical_distribution import ECDF


# In[3]:


# %matplotlib inline

from IPython.core.pylabtools import figsize


figsize(12, 8)

sns.set()


# ## Parte 1

# ### _Setup_ da parte 1

# In[4]:


np.random.seed(42)
    
df = pd.DataFrame({"normal": sct.norm.rvs(20, 4, size=10000),
                     "binomial": sct.binom.rvs(100, 0.2, size=10000)})


# ## Inicie sua análise a partir da parte 1 a partir daqui

# In[5]:


# Sua análise da parte 1 começa aqui.
df.describe()


# In[6]:


df.head(10)


# ## Questão 1
# 
# Qual a diferença entre os quartis (Q1, Q2 e Q3) das variáveis `normal` e `binomial` de `dataframe`? Responda como uma tupla de três elementos arredondados para três casas decimais.
# 
# Em outra palavras, sejam `q1_norm`, `q2_norm` e `q3_norm` os quantis da variável `normal` e `q1_binom`, `q2_binom` e `q3_binom` os quantis da variável `binom`, qual a diferença `(q1_norm - q1 binom, q2_norm - q2_binom, q3_norm - q3_binom)`?

# In[7]:


def q1():
    # Retorne aqui o resultado da questão 1.
    
    # Obtendo os quantis das duas variáveis
    q = df.describe().loc[['25%','50%','75%']]
    
    # Calculando a diferença entre os quantis
    q_diff = q['normal']-q['binomial']

    # Retornando uma tupla arredondada
    return tuple(q_diff.round(3))


# Para refletir:
# 
# * Você esperava valores dessa magnitude?
# 
# * Você é capaz de explicar como distribuições aparentemente tão diferentes (discreta e contínua, por exemplo) conseguem dar esses valores?

# ## Questão 2
# 
# Considere o intervalo $[\bar{x} - s, \bar{x} + s]$, onde $\bar{x}$ é a média amostral e $s$ é o desvio padrão. Qual a probabilidade nesse intervalo, calculada pela função de distribuição acumulada empírica (CDF empírica) da variável `normal`? Responda como uma único escalar arredondado para três casas decimais.

# In[8]:


# x_norm = df['normal']
# x_binom = df['binomial']

# fig, ax = plt.subplots(figsize=(12, 8))
# n_bins = 100

# # plot the cumulative histogram for the 'normal' variable
# n, bins, patches = ax.hist(x_norm, n_bins, density=True, histtype='step',
#                            cumulative=True, label='normal')

# # overlay with the 'binomial' variable
# ax.hist(x_norm, bins=bins, density=True, histtype='step', cumulative=True,
#         label='binomial')


# ax.grid(True)
# ax.legend(loc='right')
# ax.set_title('Cumulative step histograms')


# plt.show()


# In[9]:


# df.plot.kde()


# In[10]:


def q2():
    # Retorne aqui o resultado da questão 2.
    
    # Média e desvio padrão 
    x_mean = df['normal'].mean()
    x_std = df['normal'].std()
    
    # Intervalo
    data_point = (x_mean-x_std,x_mean+x_std)
    
    # CDF empírica da variável normal
    ecdf = ECDF(df['normal'])
    
    # Probabilidade no intervalo é a probabilidade do fim do intervalo 
    # menos a probabilidade do inicio do intervalo
    p1 = ecdf(data_point[0])
    p2 = ecdf(data_point[1])
    
    return round(float(p2-p1),3)


# Para refletir:
# 
# * Esse valor se aproxima do esperado teórico? Sim
# * Experimente também para os intervalos $[\bar{x} - 2s, \bar{x} + 2s]$ e $[\bar{x} - 3s, \bar{x} + 3s]$.

# ## Questão 3
# 
# Qual é a diferença entre as médias e as variâncias das variáveis `binomial` e `normal`? Responda como uma tupla de dois elementos arredondados para três casas decimais.
# 
# Em outras palavras, sejam `m_binom` e `v_binom` a média e a variância da variável `binomial`, e `m_norm` e `v_norm` a média e a variância da variável `normal`. Quais as diferenças `(m_binom - m_norm, v_binom - v_norm)`?

# In[11]:


def q3():
    # Retorne aqui o resultado da questão 3.
    m_diff = round(df['binomial'].mean()-df['normal'].mean(),3)
    v_diff = round(df['binomial'].var()-df['normal'].var(),3)

    return (m_diff,v_diff)
    


# Para refletir:
# 
# * Você esperava valore dessa magnitude?
# * Qual o efeito de aumentar ou diminuir $n$ (atualmente 100) na distribuição da variável `binomial`?

# ## Parte 2

# ### _Setup_ da parte 2

# In[12]:


stars = pd.read_csv("HTRU2/HTRU_2.csv")

stars.rename({old_name: new_name
              for (old_name, new_name)
              in zip(stars.columns,
                     ["mean_profile", "sd_profile", "kurt_profile", "skew_profile", "mean_curve", "sd_curve", "kurt_curve", "skew_curve", "target"])
             },
             axis=1, inplace=True)

stars.loc[:, "target"] = stars.target.astype(bool)


# ## Inicie sua análise da parte 2 a partir daqui

# In[13]:


# Sua análise da parte 2 começa aqui.
stars.describe()


# In[14]:


stars.head()


# ## Questão 4
# 
# Considerando a variável `mean_profile` de `stars`:
# 
# 1. Filtre apenas os valores de `mean_profile` onde `target == 0` (ou seja, onde a estrela não é um pulsar).
# 2. Padronize a variável `mean_profile` filtrada anteriormente para ter média 0 e variância 1.
# 
# Chamaremos a variável resultante de `false_pulsar_mean_profile_standardized`.
# 
# Encontre os quantis teóricos para uma distribuição normal de média 0 e variância 1 para 0.80, 0.90 e 0.95 através da função `norm.ppf()` disponível em `scipy.stats`.
# 
# Quais as probabilidade associadas a esses quantis utilizando a CDF empírica da variável `false_pulsar_mean_profile_standardized`? Responda como uma tupla de três elementos arredondados para três casas decimais.

# In[15]:


# Criando a variável false_pulsar_mean_profile_standardized que será usada nas questões 4 e 5

# Estrelas que não são pulsares
aux = stars.mean_profile[stars['target']==0]

# Padronização das variáveis
false_pulsar_mean_profile_standardized = (aux-aux.mean())/aux.std()


# In[16]:


def q4():
    # Retorne aqui o resultado da questão 4.
    
    # Calculando a CDF empírica
    ecdf = ECDF(false_pulsar_mean_profile_standardized)
    
    # Quantis teóricos
    quantis = sct.norm.ppf([0.8,0.9,0.95])
    
    # Probabilidade associada aos quantis
    return tuple(ecdf(quantis).round(3))
 


# Para refletir:
# 
# * Os valores encontrados fazem sentido? Sim
# * O que isso pode dizer sobre a distribuição da variável `false_pulsar_mean_profile_standardized`? Que ela tem distribuição normal

# ## Questão 5
# 
# Qual a diferença entre os quantis Q1, Q2 e Q3 de `false_pulsar_mean_profile_standardized` e os mesmos quantis teóricos de uma distribuição normal de média 0 e variância 1? Responda como uma tupla de três elementos arredondados para três casas decimais.

# In[18]:


def q5():
    # Retorne aqui o resultado da questão 5.
    
    # Quantis Q1, Q2 e Q3
    quantis_norm = sct.norm.ppf([0.25,0.5,0.75])
    
    
    quantis_false_pulsar = np.quantile(false_pulsar_mean_profile_standardized, [0.25,0.5,0.75])
    
    # Calculando a diferença entre o quantis
    quantis_diff = quantis_false_pulsar-quantis_norm
    
    return tuple(quantis_diff.round(3))


# Para refletir:
# 
# * Os valores encontrados fazem sentido?
# * O que isso pode dizer sobre a distribuição da variável `false_pulsar_mean_profile_standardized`?
# * Curiosidade: alguns testes de hipóteses sobre normalidade dos dados utilizam essa mesma abordagem.
