# Projet TAL 
# SOYKOK Aylin 28713301 - CELIK Simay 28713301
# Fonctions d'évaluation

import numpy as np
import matplotlib.pyplot as plt
import codecs
import re
import os.path
import string
import nltk
from nltk.stem.snowball import FrenchStemmer
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from collections import Counter
from utils_donnee import *
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, average_precision_score
from sklearn.model_selection import cross_val_score, train_test_split

def save_pred(pred):
    """ Sauvegarder les prédictions dans un fichier
    Entrée : Array (de M et C)
    Sortie : None + file dans predictions
    """
    directory = "./predictions"
    index = len(os.listdir(directory))+1 #nb de fichiers + 1 = l'indice du nouveau fichier
    filename = os.path.join(directory,f'test_{index}.txt')
    np.savetxt(filename, pred, fmt="%s")

#
#def prediction_evaluation_crossval(vectorizer,vect_params,model,model_params,X,Y,cv=5):
#    """Faire une prédiction sur les données avec la validation croisée.
 #   Entrée : vectorizer et vect_params : vectorizer à utiliser et ses paramètres
  #           model et model_params: l'éstimateur et ses paramètres
   #          X : données train à évaluer
    #         Y : labels
#             cv : nb de folds
 #   Sortie :
  #  Hypothèse : Données sont pre-processed.
   # """
    #vec = vectorizer(**vect_params)
   # X_vec = vec.
   # scores = cross_val_score(model,X,Y,cv=cv)

def eval_test(preprocessor,vectorizer,vect_params,model,model_params):
    """Evaluer une prediction sur le fichier selon preprocessor, vectorizer, model donnés."""
    # chargement des données train 
    alltxts_train,labs_train = load_pres("./datasets/AFDpresidentutf8/corpus.tache1.learn.utf8")
    x_train, x_test, y_train, y_test = train_test_split(alltxts_train, labs_train, test_size=0.2, random_state=42, stratify=labs_train)

    # Vectorization
    vec = vectorizer(preprocessor=preprocessor,**vect_params)
    x_train_trainsformed = vec.fit_transform(x_train)
    x_test_trainsformed = vec.transform(x_test)

    # Modélisation 
    mod = model(**model_params)
    mod.fit(x_train_trainsformed,y_train)

    # Prédiction
    pred =  mod.predict(x_test_trainsformed)
    probabilites = mod.predict_proba(x_test_trainsformed)
    proba_M = probabilites[:,0]

    # Métriques d'évaluation
    accuracy = accuracy_score(y_test, pred)
    f1_weighted = f1_score(y_test, pred, average='weighted')
    f1 = f1_score(y_test, pred)
    f1_minority = f1_score(y_test, pred, pos_label=-1) # pour Mitterrand
    precision = precision_score(y_test, pred, average='weighted')

    # for auc_roc, not sure if I have to use pred or proba?
    auc_m = roc_auc_score(y_test, proba_M)
    roc_auc_weighted = roc_auc_score(y_test, pred, average='weighted') # not very sure about this one
    ap = average_precision_score(y_test, proba_M)

    print("Accuracy:", accuracy)
    print("F1 Score (weighted):", f1_weighted)
    print("F1 Score:", f1)
    print("F1 Score sur Mitterrand (minoritaire):", f1_minority)
    print("Precision (weighted):", precision)
    print("ROC AUC (weighted):", roc_auc_weighted)
    print("ROC AUC sur Mitterrand (minoritaire):", auc_m)
    print("AP sur Mitterrand (minoritaire):", ap)
    #return proba_M

def prediction_generator(preprocessor,vectorizer,vect_params,model,model_params,save=True):
    """Faire une prediction sur le fichier selon preprocessor, vectorizer, model donnés.
    Entrée : file : nom du fichier
             preprocessor : preprocessor pour nettoyer les données
             vectorizer : vectorizer à utiliser
             vect_params : parametres de vectorizer à utiliser
             model : modèle à utiliser
             model_params : paramètres du modèle à utiliser
             save : si True, sauvegarder le résultat dans un fichier
    Sortie : Array de M et C + fichier contenant ce vecteur 
    """

    # chargement des données train 
    alltxts_train,labs_train = load_pres("./datasets/AFDpresidentutf8/corpus.tache1.learn.utf8")
    # chargement des données test
    alltxts_test = load_pres_test("./datasets/AFDpresidentutf8/corpus.tache1.test.utf8")

    # Vectorization
    vec = vectorizer(preprocessor=preprocessor,**vect_params)
    txts_train = vec.fit_transform(alltxts_train)
    txts_test = vec.transform(alltxts_test)

    # Modélisation 
    mod = model(**model_params)
    mod.fit(txts_train,labs_train)

    # Prédiction
    probabilites = mod.predict_proba(txts_test)
    proba_M = probabilites[:,0]
    pred =  mod.predict(txts_test)
    print(pred[0],proba_M[0])
    #pred = np.where(pred == -1, 'M','C')

    if save==True:
        save_pred(proba_M)

    return proba_M


