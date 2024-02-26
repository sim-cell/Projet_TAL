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
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV, RandomizedSearchCV
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from itertools import product
from imblearn.pipeline import Pipeline
from nltk.corpus import stopwords
from sklearn.naive_bayes import MultinomialNB
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
import warnings
#warnings.filterwarnings("ignore", message="Your stop_words may be inconsistent with your preprocessing.")
from sklearn.calibration import CalibratedClassifierCV
import lightgbm as lgb

def smoothing(pred,window_size=3):
    step_size = 1
    nb_windows = (len(pred) - window_size) // step_size + 1
    filtered = np.zeros_like(pred)
    for i in range(nb_windows):
        start = i * step_size
        end = start + window_size
        window = pred[start:end]
        mean = np.mean(window)
        filtered[start:end] = mean
    threshold = 0.5
    filtered = np.where(filtered > threshold, 1, -1)
    return filtered

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

def eval_test(preprocessor,vectorizer,vect_params,model,model_params,under_sample=False,over_sample=False,post_processing=False):
    """Evaluer une prediction sur le fichier selon preprocessor, vectorizer, model donnés."""
    # chargement des données train 
    alltxts_train,labs_train = load_pres("./datasets/AFDpresidentutf8/corpus.tache1.learn.utf8")

    if(model==xgb.XGBClassifier):
        # Preprocess des labels car XGBoost prend des labels 0 et 1
        label_encoder = LabelEncoder()
        labs_train = label_encoder.fit_transform(labs_train)
    
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

    if(model==LinearSVC):
            f1_minority = f1_score(y_test, pred, pos_label=-1) # pour Mitterrand
            print("F1 Score sur Mitterrand (minoritaire):", f1_minority)
            return
    
    probabilites = mod.predict_proba(x_test_trainsformed)
    if post_processing:
        probabilites = smoothing(probabilites)
    
    proba_M = probabilites[:,0]
    proba_C = probabilites[:,1]

    # Métriques d'évaluation
    accuracy = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    if(model==xgb.XGBClassifier):
        f1_minority = f1_score(y_test, pred, pos_label=0) # pour Mitterrand
    else:
        f1_minority = f1_score(y_test, pred, pos_label=-1) # pour Mitterrand

    precision = precision_score(y_test, pred)

    auc_m = roc_auc_score(y_test, proba_M)
    auc_c = roc_auc_score(y_test, proba_C)
    ap = average_precision_score(y_test, proba_M)

    

    print("Accuracy:", "%.4f"%accuracy)
    print("F1 Score:", "%.4f"%f1)
    print("Precision:", "%.4f"%precision)
    print("ROC AUC sur Mitterrand (minoritaire):", "%.4f"%auc_m)

    # these 3 metrics are what's used in the server
    print("-----Metrics du serveur--------")
    print("F1 Score sur Mitterrand (minoritaire):", "%.4f"%f1_minority)
    print("ROC AUC sur Chirac:", "%.4f"%auc_c)
    print("AP sur Mitterrand (minoritaire):", "%.4f"%ap)
    #return proba_M
    return accuracy,f1,f1_minority,auc_c,ap

def accuracy_difference(result1,result2):
    """Renvoie quand et le taux des cas où le resultat1 était plus précis / accurate.
    1 si resultat1 > resultat2, 2 si resultat2>resultat1 et 0 s'il y a une égalité.
    """
    res = []
    nb_count = 0
    for i in range(len(result1)) :
        r1,r2 = result1[i],result2[i]
        if r1>r2:
            res.append(1)
            nb_count+=1
        elif r2>r1:
            res.append(2)
        else:
            res.append(0)
    return res,nb_count/len(result1)

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
    if(model==LinearSVC):
        # LinearSVC n'a pas predict_proba
        mod = CalibratedClassifierCV(mod, method='sigmoid')
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

def best_params_lr(preprocessor_f,vect_params,f1=False,auc=False):
    """
    Trouver les meuilleurs paramètres pour la regression logistique avec gridsearch. 
    La métrique est soit f1 sur Mitterand soit roc auc sur Chirac.
    """

    # chargement des données train 
    alltxts_train,labs_train = load_pres("./datasets/AFDpresidentutf8/corpus.tache1.learn.utf8")

    lr_pipeline = Pipeline([
    ('tfidf_vectorizer', TfidfVectorizer(preprocessor=preprocessor_f, **vect_params)),
    ('sampling', RandomOverSampler(random_state=42)),
    ('lr', LogisticRegression())
    ])

    grid_params = {
    'lr__solver': ['liblinear'],
    'lr__penalty': ['l1', 'l2'],
    'lr__C': [0.1, 1, 10, 100]
    }

    if(f1):
        custom_scorer = make_scorer(f1_score, greater_is_better=True,  pos_label=-1) # f1 score sur Mitterrand
    elif(auc):
        custom_scorer = 'roc_auc'
    else:
        raise ValueError("Only one of f1 or auc must be True.")
    
    grid = GridSearchCV(lr_pipeline, grid_params, scoring=custom_scorer, n_jobs=-1)
    grid.fit(alltxts_train, labs_train)
    print("Best Score: ", grid.best_score_)
    print("Best Logistic Regression Params: ", grid.best_params_)
    return grid.best_score_, grid.best_params_

def best_params_nb(preprocessor_f,vect_params,f1=False,auc=False):
    """
    Trouver les meuilleurs paramètres pour Naive Bayes avec gridsearch. 
    La métrique est soit f1 sur Mitterand soit roc auc sur Chirac.
    """
   
    # chargement des données train 
    alltxts_train,labs_train = load_pres("./datasets/AFDpresidentutf8/corpus.tache1.learn.utf8")

    mnb_pipeline = Pipeline([
    ('tfidf_vectorizer', TfidfVectorizer(preprocessor=preprocessor_f, **vect_params)),
    ('sampling', RandomOverSampler(random_state=42)),
    ('mnb', MultinomialNB())
    ])

    grid_params = {
    'mnb__alpha': np.linspace(0.5, 1.5, 6),
    'mnb__fit_prior': [True, False]
    }

    if(f1):
        custom_scorer = make_scorer(f1_score, greater_is_better=True,  pos_label=-1) # f1 score sur Mitterrand
    elif(auc):
        custom_scorer = 'roc_auc'
    else:
        raise ValueError("Only one of f1 or auc must be True.")

    grid = GridSearchCV(mnb_pipeline, grid_params, scoring=custom_scorer, n_jobs=-1)
    grid.fit(alltxts_train, labs_train)
    print("Best Score: ", grid.best_score_)
    print("Best Naive Bayes Params: ", grid.best_params_)
    return grid.best_score_, grid.best_params_

def best_params_xgb(preprocessor_f,vect_params,f1=False,auc=False):
    """
    Trouver les meuilleurs paramètres pour XGBoost avec random search. 
    La métrique est soit f1 sur Mitterand soit roc auc sur Chirac.
    """
   
    # chargement des données train 
    alltxts_train,labs_train = load_pres("./datasets/AFDpresidentutf8/corpus.tache1.learn.utf8")

    # Preprocess des labels car XGBoost prend des labels 0 et 1
    label_encoder = LabelEncoder()
    labs_train = label_encoder.fit_transform(labs_train)

    xgb_pipeline = Pipeline([
    ('tfidf_vectorizer', TfidfVectorizer(preprocessor=preprocessor_f, **vect_params)),
    ('sampling', RandomOverSampler(random_state=42)),
    ('xgb', xgb.XGBClassifier())
    ])

    dico_params = {
    'xgb__subsample': [0.6, 0.8, 1.0],
    'xgb__min_child_weight': [1, 5, 10],
    'xgb__gamma': [0.5, 1, 1.5, 2, 5],
    'xgb__colsample_bytree': [0.6, 0.8, 1.0],
    'xgb__max_depth': [3, 4, 5],
    }

    if(f1):
        custom_scorer = make_scorer(f1_score, greater_is_better=True, pos_label=0) # f1 score sur Mitterrand
    elif(auc):
        custom_scorer = 'roc_auc'
    else:
        raise ValueError("Only one of f1 or auc must be True.")

    random_search = RandomizedSearchCV(xgb_pipeline, param_distributions=dico_params, n_iter=50, scoring=custom_scorer, n_jobs=-1, cv=3, random_state=42, verbose=0)
    random_search.fit(alltxts_train, labs_train)
    print("Best Score: ", random_search.best_score_)
    print("Best XGBoost Params: ", random_search.best_params_)
    return random_search.best_score_, random_search.best_params_

def best_params_lgbm(preprocessor_f, vect_params, f1=False, auc=False):
    """
    Trouver les meuilleurs paramètres pour LightGBM avec random search. 
    La métrique est soit f1 sur Mitterand soit roc auc sur Chirac.
    """

    # chargement des données train 
    alltxts_train, labs_train = load_pres("./datasets/AFDpresidentutf8/corpus.tache1.learn.utf8")

    lgbm_pipeline = Pipeline([
        ('tfidf_vectorizer', TfidfVectorizer(preprocessor=preprocessor_f, **vect_params)),
        ('sampling', RandomOverSampler(random_state=42)),
        ('lgbm', lgb.LGBMClassifier())
    ])

    params = {
        'lgbm__num_leaves': [5, 20, 30, 50],
        'lgbm__learning_rate': [0.05, 0.1, 0.2],
        'lgbm__n_estimators': [50, 100, 150],
        'lgbm__verbose': [-1]
    }

    if f1:
        custom_scorer = make_scorer(f1_score, greater_is_better=True, pos_label=-1)  # f1 score sur Mitterrand
    elif auc:
        custom_scorer = 'roc_auc'
    else:
        raise ValueError("Only one of f1 or auc must be True.")

    random_search = RandomizedSearchCV(lgbm_pipeline, param_distributions=params, n_iter=50, scoring=custom_scorer, n_jobs=-1, cv=3, random_state=42, verbose=0)
    random_search.fit(alltxts_train, labs_train)
    print("Best Score: ", random_search.best_score_)
    print("Best LightGBM Params: ", random_search.best_params_)
    return random_search.best_score_, random_search.best_params_