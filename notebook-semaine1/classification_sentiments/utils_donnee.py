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
from nltk.stem.snowball import EnglishStemmer
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
    print(len(lines))
    return lines

# Suppression de la ponctuation
def ponc_suppression(text):
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    return regex.sub('', text)

# Suppression des chiffres
def chiffre_suppression(text):
    return re.sub('[0-9]+', '', text)

# Comptage des ratings 
def count_ratings(text):
    pattern = r'\b[0-9]/10\b'
    ratings = re.findall(pattern,text)
    if len(ratings)>0:
        return len(ratings) #, ratings
    return None

# Suppression des majuscules 
def uppercase_suppression(text):
    return text.lower()

# Ne prend en compte que la première ligne
def premier_ligne(text):
    return text.split('\n')[0]

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

# Trouver les lignes contenant des balises
def find_lines_with_tags(texts):
    lines_with_tags = []
    for i, text in enumerate(texts):
        if re.search(r'<[^>]+>', text):
            lines_with_tags.append(i)
    return lines_with_tags

# Supression des balises
def remove_tags(text):
    return re.sub(r'<[^>]+>', '', text)

# Stemming
#nltk.download('punkt') #décommenter ça si vous n'avez pas encore téléchargé
def stem(text):
    english_stemmer = EnglishStemmer()
    words = nltk.word_tokenize(text)
    stemmed_words = [english_stemmer.stem(word) for word in words]
    stemmed_text = ' '.join(stemmed_words)
    return stemmed_text