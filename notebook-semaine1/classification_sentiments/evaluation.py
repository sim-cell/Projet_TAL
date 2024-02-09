# Projet TAL 
# SOYKOK Aylin 28713301 - CELIK Simay 28713301
# Fonctions d'évaluation - Films

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
from collections import Counter
from utils_donnee import *
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

def save_pred(pred):
    """ Sauvegarder les prédictions dans un fichier
    Entrée : Array (de M et C)
    Sortie : None + file dans predictions
    """
    directory = "./predictions"
    index = len(os.listdir(directory))+1 #nb de fichiers + 1 = l'indice du nouveau fichier
    filename = os.path.join(directory,f'test_{index}.txt')
    np.savetxt(filename, pred, fmt="%s")


def prediction_generator(preprocessor,vectorizer,vect_params,model,model_params,save=False):
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
    alltxts_train,labs_train = load_movies("./datasets/movies/movies1000/")
    # chargement des données test
    alltxts_test = load_movies_test("./datasets/movies/testSentiment.txt")

    # Vectorization
    vec = vectorizer(preprocessor=preprocessor,**vect_params)
    txts_train = vec.fit_transform(alltxts_train)

    # Training
    [X_train, X_test, y_train, y_test]  = train_test_split(txts_train, labs_train, test_size=0.2, random_state=10, shuffle=True)

    # Modélisation 
    mod = model(**model_params)
    mod.fit(txts_train,labs_train)

    # Prédiction
    pred_train = mod.predict(X_train)
    pred_test = mod.predict(X_test)

    pred = arr = np.where(pred_test == 0, "N", "P")

    if save==True:
        save_pred(pred_test)

    return pred


