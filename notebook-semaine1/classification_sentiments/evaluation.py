# Projet TAL 
# SOYKOK Aylin 28713301 - CELIK Simay 28713301
# Fonctions d'évaluation - Films

import numpy as np
import matplotlib.pyplot as plt
import codecs
import re
import os.path
import string
from nltk.stem.snowball import EnglishStemmer
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from collections import Counter
from utils_donnee import *
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, roc_curve, average_precision_score, make_scorer
from sklearn.model_selection import cross_val_score, cross_val_predict, train_test_split, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
import time
from sklearn.model_selection import StratifiedKFold

def save_pred(pred):
    """ Sauvegarder les prédictions dans un fichier
    Entrée : Array (de M et C)
    Sortie : None + file dans predictions
    """
    directory = "./predictions"
    index = len(os.listdir(directory))+1 #nb de fichiers + 1 = l'indice du nouveau fichier
    filename = os.path.join(directory,f'test_{index}.txt')
    with open(filename,"w") as file:
        for i in range(len(pred)):
            if pred[i]==0:
                file.write("N")
            elif pred[i]==1:
                file.write("P")
            else :
                print("Error, value not 0 or 1")
            if i!=len(pred)-1:
                file.write("\n")
    

# Afin de tester le model trainé sur les données de test
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
    txts_test = vec.transform(alltxts_test)

    # Modélisation 
    mod = model(**model_params)
    mod.fit(txts_train,labs_train)

    # Prédiction
    pred_train = mod.predict(txts_train)
    pred_test = mod.predict(txts_test)
    print(pred_test[:4])
    if save==True:
        save_pred(pred_test)

    return pred_test
    

# evaluation avec train-test-split simple
def eval_split(preprocessor,vectorizer,vect_params,model,model_params,graphe=True,over_sample=False,under_sample=False,timer=False):
    """Evaluer une prediction avec train_test_split"""
    # chargement des données train 
    alltxts_train,labs_train = load_movies("./datasets/movies/movies1000/")
    if timer:
        start = time.time()
    # séparer les données en train et test
    x_train, x_test, y_train, y_test = train_test_split(alltxts_train, labs_train, test_size=0.2, random_state=42, stratify=labs_train)

     # Vectorization
    vec = vectorizer(preprocessor=preprocessor,**vect_params)
    
    txts_train = vec.fit_transform(x_train)
    txts_test = vec.transform(x_test)

    # Sampling si nécessaire
    if under_sample:
        sampler = RandomUnderSampler(random_state=42)
        txts_train, y_train = sampler.fit_resample(txts_train, y_train)
    elif over_sample:
        sampler = RandomOverSampler(random_state=42)
        txts_train, y_train = sampler.fit_resample(txts_train, y_train)

    # Modélisation 
    mod = model(**model_params)
    mod.fit(txts_train,y_train)

    #Prédiction
    pred_train = mod.predict(txts_train)
    pred_test = mod.predict(txts_test)

    #Proba qui marche pour LR et SVC 
    if isinstance(mod, LinearSVC):
        decision_scores_train = mod.decision_function(txts_train)
        decision_scores_test = mod.decision_function(txts_test)
        
        accuracy_train = accuracy_score(y_train, pred_train)
        f1_train = f1_score(y_train, pred_train)
        roc_auc_train = roc_auc_score(y_train, decision_scores_train)
        avg_precision_train = average_precision_score(y_train, decision_scores_train)

        accuracy_test = accuracy_score(y_test, pred_test)
        f1_test = f1_score(y_test, pred_test)
        roc_auc_test = roc_auc_score(y_test, decision_scores_test)
        avg_precision_test = average_precision_score(y_test, decision_scores_test)
        if timer:
            end = time.time()
            print("Durée d'exécution :",end-start)
        print("Résultats train")
        print("Acc\tF1\tROC-AUC\tAP:")
        print("%.4f" % accuracy_train, "\t%.4f" % f1_train, "\t%.4f" % roc_auc_train, "\t%.4f" % avg_precision_train)
        print("Résultats test")
        print("Acc\tF1\tROC-AUC\tAP:")
        print("%.4f" % accuracy_test, "\t%.4f" % f1_test, "\t%.4f" % roc_auc_test, "\t%.4f" % avg_precision_test)

        if graphe:
            # Afficher la courbe ROC
            fpr_train, tpr_train, _ = roc_curve(y_train, decision_scores_train)
            fpr_test, tpr_test, _ = roc_curve(y_test, decision_scores_test)

            plt.figure()
            plt.plot(fpr_train, tpr_train, label='Courbe ROC Train (AUC = %0.2f)' % roc_auc_train)
            plt.plot(fpr_test, tpr_test, label='Courbe ROC Test (AUC = %0.2f)' % roc_auc_test)
            plt.plot([0, 1], [0, 1], linestyle='--')
            plt.xlabel('Taux de faux positifs')
            plt.ylabel('Taux de vrais positifs')
            plt.title('Courbe ROC')
            plt.legend(loc="lower right")
            plt.show()

        return accuracy_test, f1_test, roc_auc_test, avg_precision_test

    probabilites = mod.predict_proba(txts_test)
    proba_neg = probabilites[:,0]
    proba_pos = probabilites[:,1]

    # Métriques d'évaluation
    accuracy_test = accuracy_score(y_test, pred_test)
    f1_test = f1_score(y_test, pred_test)
    roc_auc_test = roc_auc_score(y_test, proba_pos)
    avg_precision_test = average_precision_score(y_test, proba_pos)

    accuracy_train = accuracy_score(y_train, pred_train)
    f1_train = f1_score(y_train, pred_train)
    roc_auc_train = roc_auc_score(y_train,  mod.predict_proba(txts_train)[:, 1])
    avg_precision_train = average_precision_score(y_train,  mod.predict_proba(txts_train)[:, 1])
    if timer:
        end = time.time()
        print("Durée d'exécution :",end-start)
    print("Resultats train")
    print("Acc\tF1\tROC-AUC\tAP:")
    print("%.4f"%accuracy_train,"\t%.4f"%f1_train,"\t%.4f"%roc_auc_train,"\t%.4f"%avg_precision_train)
    print("Resultats test")
    print("Acc\tF1\tROC-AUC\tAP:")
    print("%.4f"%accuracy_test,"\t%.4f"%f1_test,"\t%.4f"%roc_auc_test,"\t%.4f"%avg_precision_test)  
    if graphe:
        #afficher roc courbe
        fpr, tpr, thresholds = roc_curve(y_test, proba_pos)
        fpr2, tpr2, thresholds = roc_curve(y_train,  mod.predict_proba(txts_train)[:, 1])
        plt.figure()
        plt.plot(fpr2, tpr2, label='courbe ROC train' % roc_auc_train)
        plt.plot(fpr, tpr, label='courbe ROC test' % roc_auc_test)
        plt.xlabel('FP')
        plt.ylabel('TP')
        plt.legend(loc="lower right")
        plt.title('Courbe ROC')
        plt.show()

    return accuracy_test,f1_test,roc_auc_test,avg_precision_test



# evaluation avec crossval
def eval_crossval(preprocessor,vectorizer,vect_params,model,model_params,graphe=True,cv=5,over_sample=False,under_sample=False,timer=False):
    """Evaluer une prediction avec la validation croisée"""
    # chargement des données train 
    alltxts_train,labs_train = load_movies("./datasets/movies/movies1000/")
    if timer:
        start = time.time()
     # Vectorization
    vec = vectorizer(preprocessor=preprocessor,**vect_params)
    
    txts_train = vec.fit_transform(alltxts_train)
    # Sampling si nécessaire
    if under_sample:
        sampler = RandomUnderSampler(random_state=42)
        txts_train, labs_train = sampler.fit_resample(txts_train, labs_train)
    elif over_sample:
        sampler = RandomOverSampler(random_state=42)
        txts_train, labs_train = sampler.fit_resample(txts_train, labs_train)

    # Modélisation 
    mod = model(**model_params)

    # Cross val
    #Proba qui marche pour LR et SVC 
    if isinstance(mod, LinearSVC):
        proba_pos = cross_val_predict(mod, txts_train, labs_train, cv=cv, method='decision_function')
    else:
        probabilites = cross_val_predict(mod, txts_train, labs_train, cv=cv, method='predict_proba')
        proba_neg = probabilites[:,0]
        proba_pos = probabilites[:,1]
    pred =  cross_val_predict(mod, txts_train, labs_train, cv=cv)
    
    # Métriques d'évaluation
    accuracy = accuracy_score(labs_train, pred)
    f1 = f1_score(labs_train, pred)
    roc_auc = roc_auc_score(labs_train, proba_pos)
    avg_precision = average_precision_score(labs_train, proba_pos)

    print("Resultats cross validation")
    if timer:
        end = time.time()
        print("Durée d'exécution :",end-start)

    print("Acc\tF1\tROC-AUC\tAP:")
    print("%.4f"%accuracy,"\t%.4f"%f1,"\t%.4f"%roc_auc,"\t%.4f"%avg_precision)

    if graphe:
        #afficher roc courbe
        fpr, tpr, thresholds = roc_curve(labs_train, proba_pos)
        plt.figure()
        plt.plot(fpr, tpr, label='courbe ROC' % roc_auc)
        plt.xlabel('FP')
        plt.ylabel('TP')
        plt.legend(loc="lower right")
        plt.title('Courbe ROC')
        plt.show()

    return accuracy,f1,roc_auc,avg_precision

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
            nb_count+=1
    return res,nb_count/len(result1)

def comparaison_evaluation(preprocessor,vect_params_tf,vect_params_c,model_params_LR,model_params_SVM,eval_func=eval_split,result_type='lr',timer=False,**args):
    print("LOGISTIC REGRESSION")
    print("Résultats Tfidf")
    lr_tfidf_sans = eval_func(preprocessor=preprocessor, vectorizer=TfidfVectorizer, vect_params=vect_params_tf,
    model=LogisticRegression, model_params=model_params_LR,graphe=False,timer=timer,**args)
    print("_________________________")
    print("Résultats CountVectorizer")
    lr_countv_sans = eval_func(preprocessor=preprocessor, vectorizer=CountVectorizer, vect_params=vect_params_c,
    model=LogisticRegression, model_params=model_params_LR,graphe=False,timer=timer,**args)
    print("_________________________")
    res,nb_fois = accuracy_difference(lr_tfidf_sans,lr_countv_sans)
    print(f'Taux d\'accuracy de Tfidf contre Count : {nb_fois}')
    print("_____________________________________________")
    print("\nSVM ")
    print("Résultats Tfidf")
    svm_tfidf_sans = eval_func(preprocessor=preprocessor, vectorizer=TfidfVectorizer, vect_params=vect_params_tf,
    model=LinearSVC, model_params=model_params_SVM,graphe=False,timer=timer,**args)
    print("_________________________")
    print("Résultats CountVectorizer")
    svm_countv_sans = eval_func(preprocessor=preprocessor, vectorizer=CountVectorizer, vect_params=vect_params_c,
    model=LinearSVC, model_params=model_params_SVM,graphe=False,timer=timer,**args)
    print("_________________________")
    res,nb_fois = accuracy_difference(svm_tfidf_sans,svm_countv_sans)
    print(f'Taux d\'accuracy de Tfidf contre Count : {nb_fois}')
    print("_____________________________________________")
    print("\n MultinomialNB ")
    print("Résultats Tfidf")
    m_tfidf_sans = eval_func(preprocessor=preprocessor, vectorizer=TfidfVectorizer, vect_params=vect_params_tf,
    model=MultinomialNB, model_params={},graphe=False,timer=timer,**args)
    print("_________________________")
    print("Résultats CountVectorizer")
    m_countv_sans = eval_func(preprocessor=preprocessor, vectorizer=CountVectorizer, vect_params=vect_params_c,
    model=MultinomialNB, model_params={},graphe=False,timer=timer,**args)
    print("_________________________")
    res,nb_fois = accuracy_difference(m_tfidf_sans,m_countv_sans)
    print(f'Taux d\'accuracy de Tfidf contre Count : {nb_fois}')
    print("______________________________________________________")
    res,nb_fois1 = accuracy_difference(lr_tfidf_sans,svm_tfidf_sans)
    print(f'Taux d\'accuracy de LinReg contre LinSVM : {nb_fois1}')
    res,nb_fois2 = accuracy_difference(lr_tfidf_sans,m_tfidf_sans)
    print(f'Taux d\'accuracy de LinReg contre Multinom : {nb_fois2}')
    if result_type=='m':
        return m_tfidf_sans
    if result_type=='svm':
        return svm_tfidf_sans
    return lr_tfidf_sans
    

def comparaison_evaluation_single(preprocessor,vectorizer,vect_params,model_params_LR,model_params_SVM,eval_func=eval_split,result_type='lr',timer=False,**args):
    print("Comparaison des modèles :",vectorizer.__class__.__name__)
    print("LOGISTIC REGRESSION")
    print("Résultats ")
    lr = eval_func(preprocessor=preprocessor, vectorizer=vectorizer, vect_params=vect_params,
    model=LogisticRegression, model_params=model_params_LR,graphe=False,timer=timer,**args)
    print("_____________________________________________")
    print("\nSVM ")
    print("Résultats")
    svm = eval_func(preprocessor=preprocessor, vectorizer=vectorizer, vect_params=vect_params,
    model=LinearSVC, model_params=model_params_SVM,graphe=False,timer=timer,**args)
    print("_____________________________________________")
    print("\n MultinomialNB ")
    print("Résultats")
    m= eval_func(preprocessor=preprocessor, vectorizer=vectorizer, vect_params=vect_params,
    model=MultinomialNB, model_params={},graphe=False,timer=timer,**args)
    print("______________________________________________________")
    res,nb_fois1 = accuracy_difference(lr,svm)
    print(f'Taux d\'accuracy de LinReg contre LinSVM : {nb_fois1}')
    res,nb_fois2 = accuracy_difference(lr,m)
    print(f'Taux d\'accuracy de LinReg contre Multinom : {nb_fois2}')
    return [lr,svm,m]


#comparaison des k pour crossval
def comparaison_crossval(preprocessor,vectorizer,vect_params,model,model_params,graphe=True,cvs=[5]):
    # chargement des données train 
    alltxts_train,labs_train = load_movies("./datasets/movies/movies1000/")

     # Vectorization
    vec = vectorizer(preprocessor=preprocessor,**vect_params)
    txts_train = vec.fit_transform(alltxts_train)

    # Modélisation 
    mod = model(**model_params)

    # Cross val
    best_k = -1
    best_accuracy = -1
    best_f1 = -1
    best_roc_auc = -1
    best_avg_precision = -1
    best_proba_pos = None
    best_pred = None   
    tous_res = []
    for cv in cvs: 
        if isinstance(mod, LinearSVC):
            proba_pos = cross_val_predict(mod, txts_train, labs_train, cv=cv, method='decision_function') #pas vraiment proba_pos mais je veux pas changer ts les variables
        else:
            probabilites = cross_val_predict(mod, txts_train, labs_train, cv=cv, method='predict_proba')
            proba_neg = probabilites[:,0]
            proba_pos = probabilites[:,1]
        pred =  cross_val_predict(mod, txts_train, labs_train, cv=cv)
        
        # Métriques d'évaluation
        accuracy = accuracy_score(labs_train, pred)
        f1 = f1_score(labs_train, pred)
        roc_auc = roc_auc_score(labs_train, proba_pos)
        avg_precision = average_precision_score(labs_train, proba_pos)
        tous_res.append([accuracy,f1,roc_auc,avg_precision])
        if accuracy>best_accuracy and f1>best_f1:
            best_k = cv
            best_accuracy = accuracy
            best_f1 = f1
            best_roc_auc = roc_auc
            best_avg_precision = avg_precision
            best_pred = pred
            best_proba_pos = proba_pos

        
    print("Resultats cross validation")
    print("Best k = ",best_k)
    print("Acc\tF1\tROC-AUC\tAP:")
    print("%.4f"%best_accuracy,"\t%.4f"%best_f1,"\t%.4f"%best_roc_auc,"\t%.4f"%best_avg_precision)

    if graphe:
        #afficher roc courbe
        fpr, tpr, thresholds = roc_curve(labs_train, best_proba_pos)
        plt.figure()
        plt.plot(fpr, tpr, label='courbe ROC' % best_roc_auc)
        plt.xlabel('FP')
        plt.ylabel('TP')
        plt.legend(loc="lower right")
        plt.title('Courbe ROC')
        plt.show()

    return best_k,[best_accuracy,best_f1,best_roc_auc,best_avg_precision],tous_res


#comparaison des k pour crossval avec KFold
def comparaison_crossval_grain(preprocessor,vectorizer,vect_params,model,model_params,graphe=True,cvs=[5],random_state=None):
    # chargement des données train 
    alltxts_train,labs_train = load_movies("./datasets/movies/movies1000/")

     # Vectorization
    vec = vectorizer(preprocessor=preprocessor,**vect_params)
    txts_train = vec.fit_transform(alltxts_train)

    # Modélisation 
    mod = model(**model_params)

    # Cross val
    best_k = -1
    best_accuracy = -1
    best_f1 = -1
    best_roc_auc = -1
    best_avg_precision = -1
    best_proba_pos = None
    best_pred = None   
    for cv in cvs: 
        fold = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
        for train_index, _ in fold.split(txts_train, labs_train):
            txts_train_cv, labs_train_cv = txts_train[train_index], np.array(labs_train)[train_index]
            if isinstance(mod, LinearSVC):
                proba_pos = cross_val_predict(mod, txts_train_cv, labs_train_cv, cv=cv, method='decision_function') #pas vraiment proba_pos mais je veux pas changer ts les variables
            else:
                probabilites = cross_val_predict(mod, txts_train_cv, labs_train_cv, cv=cv, method='predict_proba')
                proba_neg = probabilites[:,0]
                proba_pos = probabilites[:,1]
            pred =  cross_val_predict(mod, txts_train_cv, labs_train_cv, cv=cv)
            
            # Métriques d'évaluation
            accuracy = accuracy_score(labs_train_cv, pred)
            f1 = f1_score(labs_train_cv, pred)
            roc_auc = roc_auc_score(labs_train_cv, proba_pos)
            avg_precision = average_precision_score(labs_train_cv, proba_pos)
            if accuracy>best_accuracy and f1>best_f1:
                best_k = cv
                best_accuracy = accuracy
                best_f1 = f1
                best_roc_auc = roc_auc
                best_avg_precision = avg_precision
                best_pred = pred
                best_proba_pos = proba_pos

            
    print("Resultats cross validation")
    print("Best k = ",best_k)
    print("Acc\tF1\tROC-AUC\tAP:")
    print("%.4f"%best_accuracy,"\t%.4f"%best_f1,"\t%.4f"%best_roc_auc,"\t%.4f"%best_avg_precision)


    return best_k,[best_accuracy,best_f1,best_roc_auc,best_avg_precision]

def plot_evaluation_metrics(results_data, k_list):
    colors = ['b', 'g', 'r', 'c']
    for i,metric in enumerate(results_data.columns):
        x_values = k_list
        y_values = results_data[metric]
        plt.plot(x_values, y_values, label=f'{metric}', color=colors[i])

    plt.xlabel('Value of k in Cross-Validation')
    plt.ylabel('k')
    plt.title('Evaluation Metrics for Different Values of k in Cross-Validation')
    plt.legend()
    plt.grid(True)
    plt.show()
