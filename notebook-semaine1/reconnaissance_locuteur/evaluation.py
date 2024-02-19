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
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, average_precision_score, make_scorer
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from itertools import product
from sklearn.pipeline import Pipeline
from nltk.corpus import stopwords


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

def eval_test(preprocessor,vectorizer,vect_params,model,model_params,under_sample=False,over_sample=False):
    """Evaluer une prediction sur le fichier selon preprocessor, vectorizer, model donnés."""
    # chargement des données train 
    alltxts_train,labs_train = load_pres("./datasets/AFDpresidentutf8/corpus.tache1.learn.utf8")
    x_train, x_test, y_train, y_test = train_test_split(alltxts_train, labs_train, test_size=0.2, random_state=42, stratify=labs_train)
    
    # Vectorization
    vec = vectorizer(preprocessor=preprocessor,**vect_params)
    x_train_trainsformed = vec.fit_transform(x_train)
    x_test_trainsformed = vec.transform(x_test)

    # Sampling si nécessaire
    if under_sample:
        sampler = RandomUnderSampler(random_state=42)
        x_train_trainsformed, y_train = sampler.fit_resample(x_train_trainsformed, y_train)
    elif over_sample:
        sampler = RandomOverSampler(random_state=42)
        x_train_trainsformed, y_train = sampler.fit_resample(x_train_trainsformed, y_train)

    # Modélisation 
    mod = model(**model_params)
    mod.fit(x_train_trainsformed,y_train)

    # Prédiction
    pred =  mod.predict(x_test_trainsformed)
    probabilites = mod.predict_proba(x_test_trainsformed)
    proba_M = probabilites[:,0]
    proba_C = probabilites[:,1]

    # Métriques d'évaluation
    accuracy = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    f1_minority = f1_score(y_test, pred, pos_label=-1) # pour Mitterrand
    precision = precision_score(y_test, pred)

    # for auc_roc, not sure if I have to use pred or proba?
    auc_m = roc_auc_score(y_test, proba_M)
    auc_c = roc_auc_score(y_test, proba_C)
    ap = average_precision_score(y_test, proba_M)

    #print("Accuracy:", accuracy)
    #print("F1 Score:", f1)
    #print("Precision:", precision)
    #print("ROC AUC sur Mitterrand (minoritaire):", auc_m)
    print("F1 Score sur Mitterrand (minoritaire):", f1_minority)
    print("ROC AUC sur Chirac:", auc_c)
    print("AP sur Mitterrand (minoritaire):", ap)
    #return proba_M

def prediction_generator(preprocessor,vectorizer,vect_params,model,model_params,over_sample=False,save=True):
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

    # OverSampling
    if over_sample:
        sampler = RandomOverSampler(random_state=42)
        txts_train, labs_train = sampler.fit_resample(txts_train, labs_train)

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


def find_best_params(preprocessor,vectorizer,vect_params,model,model_params):
    """Trouve le meilleur combination des parametres pour le vectorizer"""

    # chargement des données train 
    alltxts_train,labs_train = load_pres("./datasets/AFDpresidentutf8/corpus.tache1.learn.utf8")

    best_score = 0
    best_vect_params = None

    all_vect_param_combinations = product(*[[(key, value) for value in values] for key, values in vect_params.items()])

    for vect_param_combination in all_vect_param_combinations:
        vec_params = dict(vect_param_combination)

        x_train, x_test, y_train, y_test = train_test_split(alltxts_train, labs_train, test_size=0.2, random_state=42, stratify=labs_train)
        
        # Vectorization
        vec = vectorizer(preprocessor=preprocessor,**vec_params)
        x_train_trainsformed = vec.fit_transform(x_train)
        x_test_trainsformed = vec.transform(x_test)

        # Over Sampling
        sampler = RandomOverSampler(random_state=42)
        x_train_trainsformed, y_train = sampler.fit_resample(x_train_trainsformed, y_train)

        # Modélisation 
        mod = model(**model_params)
        mod.fit(x_train_trainsformed,y_train)

        # Prédiction
        pred =  mod.predict(x_test_trainsformed)

        # Métriques d'évaluation
        score = f1_score(y_test, pred, pos_label=-1) # pour Mitterrand

        if score > best_score:
            best_score = score
            best_vect_params = vec_params
    
    return best_vect_params, best_score

def best_params_lr(preprocessor):
   
    # chargement des données train 
    alltxts_train,labs_train = load_pres("./datasets/AFDpresidentutf8/corpus.tache1.learn.utf8")

    tfidf_pipeline = Pipeline([
    ('tfidf_vectorizer', TfidfVectorizer(preprocessor=preprocessor))
    ])

    lr_pipeline = Pipeline([
    ('tfidf_pipeline', tfidf_pipeline),
    ('oversampler', RandomOverSampler(random_state=42)),
    ('lr', LogisticRegression())
    ])

    grid_params = {
    'lr__penalty': ['l1', 'l2'],
    'lr__C': [0.1, 1, 10, 100],
    'lr__solver': ['liblinear'],
    'tfidf_pipeline__tfidf_vectorizer__max_df': [0.5, 0.75, 1.0],
    'tfidf_pipeline__tfidf_vectorizer__binary': [True, False],
    'tfidf_pipeline__tfidf_vectorizer__stop_words': [stopwords.words('french'), None],
    'tfidf_pipeline__tfidf_vectorizer__min_df': [2, 3, 5], 
    'tfidf_pipeline__tfidf_vectorizer__ngram_range': [(1, 2), (1, 3), (2, 3)], 
    'tfidf_pipeline__tfidf_vectorizer__use_idf': [True, False],
    'tfidf_pipeline__tfidf_vectorizer__sublinear_tf': [True, False],
    'tfidf_pipeline__tfidf_vectorizer__max_features': [None, 1000, 5000, 10000]
    }

    custom_scorer = make_scorer(f1_score, greater_is_better=True,  pos_label=-1) # f1 score sur Mitterrand
    grid = GridSearchCV(lr_pipeline, grid_params, scoring=custom_scorer, n_jobs=-1)
    grid.fit(alltxts_train, labs_train)
    print("Best Score: ", grid.best_score_)
    print("Best Params: ", grid.best_params_)
    return grid.best_score_, grid.best_params_