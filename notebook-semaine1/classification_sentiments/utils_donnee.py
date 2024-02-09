# Projet TAL 
# SOYKOK Aylin 28713301 - CELIK Simay 28713301
# Fonctions de données : chargement, cleanup etc.

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

# Chargement des données:
def load_movies(path2data): # 1 classe par répertoire
    alltxts = [] # init vide
    labs = []
    cpt = 0
    for cl in os.listdir(path2data): # parcours des fichiers d'un répertoire
        for f in os.listdir(path2data+cl):
            txt = open(path2data+cl+'/'+f).read()
            alltxts.append(txt)
            labs.append(cpt)
        cpt+=1 # chg répertoire = cht classe
        
    return alltxts,labs

# Chargement des données tests:
def load_movies_test(path2data):
    with open(path2data, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    return lines

# Suppression de la ponctuation
def ponc_suppression(text):
    punc = string.punctuation 
    punc += '\n\r\t'
    return text.translate(str.maketrans(punc, ' ' * len(punc)))

# Suppression des chiffres
def chiffre_suppression(text):
    return re.sub('[0-9]+', '', text)

# Transformation des mots entièrement en majuscule en marqueurs spécifiques
def transform_uppercase(text, marker="<UPPER>"):
    words = text.split()
    new_text = []
    for word in words:
        if word.isupper(): # si tout en majuscule
            new_text.append(marker)
        else:
            new_text.append(word) 
    return ' '.join(new_text)

# Supression des balises
def remove_tags(text):
    t = re.compile(r'<[^>]+>')
    return t.sub('',text)

# Stemming
nltk.download('punkt') #décommenter ça si vous n'avez pas encore téléchargé
def stem(text):
    french_stemmer = FrenchStemmer()
    words = nltk.word_tokenize(text, language='french')
    stemmed_words = [french_stemmer.stem(word) for word in words]
    stemmed_text = ' '.join(stemmed_words)
    return stemmed_text