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
import unicodedata
from nltk.stem.snowball import FrenchStemmer
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from collections import Counter

# Chargement des données:
def load_pres(fname):
    alltxts = []
    alllabs = []
    s=codecs.open(fname, 'r','utf-8') # pour régler le codage
    while True:
        txt = s.readline()
        if(len(txt))<5:
            break
        #
        lab = re.sub(r"<[0-9]*:[0-9]*:(.)>.*","\\1",txt)
        txt = re.sub(r"<[0-9]*:[0-9]*:.>(.*)","\\1",txt)
        if lab.count('M') >0:
            alllabs.append(-1)
        else: 
            alllabs.append(1)
        alltxts.append(txt)
    return alltxts,alllabs

# Chargement des données tests:
def load_pres_test(fname):
    alltxts = []
    s=codecs.open(fname, 'r','utf-8') # pour régler le codage
    while True:
        txt = s.readline()
        if(len(txt))<5:
            break
        txt = re.sub(r"<[0-9]*:[0-9]*:.>(.*)","\\1",txt)
        alltxts.append(txt)
    return alltxts

# Suppression de la ponctuation
def ponc_suppression(text):
    punc = string.punctuation 
    punc += '\n\r\t'
    return text.translate(str.maketrans(punc, ' ' * len(punc)))

# Suppression des chiffres
def chiffre_suppression(text):
    return re.sub('[0-9]+', '', text)

# Suppression des majuscules 
# non appliquée
def uppercase_suppression(text):
    return text.lower()

# Suppression d'accent et char non normalisée
def accent_suppression(text):
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode("utf-8")

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
    t = re.compile(r'<[^>]+>')
    return t.sub('',text)

# Stemming
#nltk.download('punkt') 
def stem(text):
    french_stemmer = FrenchStemmer()
    words = nltk.word_tokenize(text, language='french')
    stemmed_words = [french_stemmer.stem(word) for word in words]
    stemmed_text = ' '.join(stemmed_words)
    return stemmed_text
