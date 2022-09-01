# -*- coding: utf-8 -*-
"""





Sparse mutlinomial - prediction distribution
"""
###########################################################
                  #Imports libraries
###########################################################
import numpy as np
import pandas as pd
from scipy.special import gammaln, logsumexp
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from coclust.evaluation.external import accuracy #accuracy metric for clustering
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import f1_score
###########################################################


###########################################################
                   # Functions
###########################################################
            
def scaling_factor(alpha, k_0, L, N):
    
    scale = 0
    lamda = .9
    sum_m = 1e-10
    for k in range(k_0,L-1):
        try:
            fact_k = (np.math.gamma(k+1)/np.math.gamma(np.abs(k-k_0)+1)) \
              * (np.math.gamma(alpha * k)) / (np.math.gamma(alpha *k + N))
        except OverflowError:
            fact_k = 1e20
        prior = lamda ** k  # exponential prior
        #prior = k ** (-lamda) #polynomial prior
        
        m_k = fact_k * prior 
        
        
        sum_m = sum_m + m_k
        scale =  scale +  (k_0 * alpha + N)/(k * alpha + N) *  m_k 
        
        m_k = 0
    scale /= sum_m
    
    return scale

def predictive_Dirichlet(words, d, symbol, alpha, k_0, L):
    
   
   
    predictive_D = 0
    N = np.sum(words)
    
    N_d = np.sum(words[:,d])
    
    scale = scaling_factor(alpha, k_0, L, N)
    epsi = 1/(L- k_0) * (1 - scale)
    for i in range(d):      
        if (symbol[i] == 0).any():
            #predictive_D = 1/(L- k_0) * (1 - scale)   #if the words are unseen
            predictive_D =  epsi     #bayesian sparse multi
        else:                                         # if the words are observed            
            #predictive_D = (alpha + N_d)/(k_0 * alpha + N)  * scale
            predictive_D = (1 - (L -k_0) * epsi) * (alpha + N_d)/(k_0 * alpha + N) #bayesian sparse multi
      
    return predictive_D

def Ristad_approach(words, d, symbol, alpha, k_0, L):
    
    predictive = 0
    N = np.sum(words)
    
    N_d = np.sum(words[:,d])
    
    for i in range(d):
        if (symbol[i] == 0).any():
            
            predictive = k_0 * (k_0 + 1) / ((L - k_0) * (N**2 + N + 2 * k_0))
        
        elif (k_0 < L):
            
            predictive = ((N_d + 1) * (N + 1 - k_0)) / (N ** 2 + N + 2 * k_0)
            
        else:
            
            predictive = (N_d + 1)/(N + L)
        
        
    return predictive
      
def emotion_prediction(sequence, pred_data, pis):

    N, D = sequence.shape
    posterior = np.ones((N,))
    " determining the predicted multinomial parameter for each cluster"
    theta = np.sum(pred_data, axis=0)
    for i in range(N):
        for d in range(D):
            posterior [i] = posterior[i] * (theta[d] ** sequence[i,d])
        posterior[i] = pis * posterior[i]
        
        
    return posterior
############################################################               

"""
###############################################
              Data preprocessing
###############################################"""
data = pd.read_csv("Data\english_all.tsv",sep="\t")
data_samples = data['poem'] [0:1000]
label = data['emotion1'] [0:1000]

#emotion label
labell = []  

for i in range(len(label)):
    if (label[i]=="Sadness"):
        labell.append(0)
    if (label[i]=="Uneasiness"):
        labell.append(1)
    if (label[i]=="Awe / Sublime"):
        labell.append(2)
    if (label[i]=="Vitality"):
        labell.append(3)
    if (label[i]=="Beauty / Joy"):
        labell.append(4)
    if (label[i]=="Humor"):
        labell.append(5)
    if (label[i]=="Suspense"):
        labell.append(6)
    if (label[i]=="Annoyance"):
        labell.append(7)
    if (label[i]=="Nostalgia"): # only for the english dictionary
        labell.append(8) 
# First, we construct the vocabulary,

vectorizer = CountVectorizer(analyzer='word', stop_words='english', max_features=200) #V=size of vocabulary
x_fits = vectorizer.fit_transform(data_samples)

x_counts = vectorizer.transform(data_samples).toarray()

"split the data in 70% training and 30% prediction/testing"
data_train, data_test, label_train, label_test = train_test_split(x_counts, labell, test_size=0.1)
"""
###############################################
             Poem prediction
###############################################
"""

"parameters"
alpha = .2


D, d = data_train.shape
L = d 

T, d = data_test.shape

new_line = np.zeros((T,d))
"""
Sequence level
""""count the number of observed words"
k_0 = np.count_nonzero(data_train)
for i in range(T):
    new_word = []
    for j in range(d):
        #next_word = predictive_Dirichlet(data_train, j,  data_test[i,:], alpha,  k_0, L)
        next_word = Ristad_approach(data_train, j,  data_test[i,:], alpha,  k_0, L)
        new_word = np.append(new_word, next_word)
    
    new_line[i,:] = new_word
        
    "update the training dataset (adding the new word)"
    data_train = np.column_stack((data_train.T, new_word))
    data_train = data_train.T
   
    k_0 = np.count_nonzero(data_train) #update the number of observed words





"""
##################################################
             Emotion prediction
       clustering the new predicted words
#################################################
"""
#Normalized Data
normalized = np.zeros((T,d))
for i in range(T):
    normalized[i,:] = abs(new_line[i,:]/sum(new_line[i,:]))


K = 9  # number of mixture components
"initialize mixing weight"

pis = np.ones (K) / K
"K-means clustering"

kmeans = KMeans(n_clusters=K).fit(normalized)
index  = kmeans.labels_
posterior_cluster = np.zeros((T,K))


for j in range(K):
    posterior_cluster[:,j] = emotion_prediction(data_test, normalized[index==j,:], pis[j])
    
    


"Evaluation metrics"
label_predicted = np.zeros((T,))
for i in range(T):
    (m,label_predicted[i]) = max((v,index) for index,v in enumerate(posterior_cluster[i,:]))



    
accuracy_test = accuracy(label_test, label_predicted)


F_1_mic = f1_score(label_test, label_predicted, average='micro')



