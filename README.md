# Projet_TAL

TO DO/ HOW TO DO

For both
- Analyse des données biraz kimde ne var
-Prétraitement
 transformation en minuscule ou pas
 suppression de la ponctuation
 transformation des mots entièrement en majuscule en marqueurs spécifiques
 suppression des chiffres ou pas
 conservation d'une partie du texte seulement (seulement la première ligne = titre, seulement la dernière ligne = résumé, ...)
 stemming
Extraction du vocabulaire (BoW)
- Exploration préliminaire des jeux de données
  Quelle est la taille d'origine du vocabulaire?
  Que reste-t-il si on ne garde que les 100 mots les plus fréquents? [word cloud]
  Quels sont les 100 mots dont la fréquence documentaire est la plus grande? [word cloud]
  Quels sont les 100 mots les plus discriminants au sens de odds ratio? [word cloud]
  Quelle est la distribution d'apparition des mots (Zipf)
  Quels sont les 100 bigrammes/trigrammes les plus fréquents?
  Variantes de BoW
  
  TF-IDF
  Réduire la taille du vocabulaire (min_df, max_df, max_features)
  BoW binaire
  Bi-grams, tri-grams
  Quelles performances attendre ? Quels sont les avantages et les inconvénients des ces variantes?
-Métriques d'évaluation
  - Accuracy + courbe ROC + AUC + F1-score
- Sur-apprentissage
   - optimiser au sens de la métrique qui nous semble la plus appropriée whatever that means
...
- Optimisationdan once test edilecek questionlar:
 -Combien de temps ça prend d'apprendre un classifieur NB/SVM/RegLog sur ces données en fonction de la taille du vocabulaire?
 -La validation croisée est-elle nécessaire? Est ce qu'on obtient les mêmes résultats avec un simple split?
 -La validation croisée est-elle stable? A partir de combien de fold (travailler avec différentes graines aléatoires et faire des statistiques basiques)?
  
Reconnaissance locuteurs
- Equilibrage des données:
 - ré-équilibrer en supprimant des données dans la classe majoritaire et/ou sur-échantiloner la classe minoritaire => bunu biz test ederek bulacagiz
 - ya da changer la formulation de la fonction de coût pour pénaliser plus les erreurs dans la classe minoritaire => SVM ve sklearnlerde var bu minoritaire e verilen poids
 - courbe ROC + modification du biais : apprentissage yapilip ychap bulunduktan sonra eger butun prédictionlar ayni classe ise ychapi biais ekleyerek hesapliyoruz (en az 1 tane point classe degistirene kadar)
- Post processing :
   - phrase successiveler genelde ayni locuteurden cikiyor. les données ne sont pas IID o yuzden gaussian filter koyunce outlierlar siliniyor. yani bir noktanin voisinage i tamamen diger classe ise, bu noktayi ignorelamis oluyoruz. gaussian smoothing i np.convolve ile yap demis tme1 de def gaussian_smoothing(pred, size):




Classification sentiments
- Bunda pek bir sey yok ek olarak
