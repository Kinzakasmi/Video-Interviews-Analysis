# import speech_feats_extract
# import spacy
# import pandas as pd
# import speech_feats_extract as sf
from speech_utils import *
import time

print('start')
t = time.time()

# nlp = spacy.load("fr_core_news_sm")         #efficiency
# nlp2 = spacy.load("fr_dep_news_trf")        #accuracy
# nlp3 = spacy.load('fr_core_news_md')        #lemma
#
# test = "J'ai mangé des pommes hier pour cette après-midi de belle journée en compagnie d'un chat et d'un chien avant d'aller faire du sport"
# test2 = "ceci est une phrase longue ceci est une autre phrase encore plus longue enfin ceci est la dernière phrase"
#
# info = nlp2(test2)
#
# print([[token.lemma_ for token in info]])
# print([token.pos_ for token in info])
# for sent in info.sents:
#     print(sent.text)
# #
# countvect = sf.CountVectorizer()
# text_fts = countvect.fit_transform([test2])
# count_array = text_fts.toarray()
#
# _words = list(countvect.get_feature_names_out())
# word_count = count_array.sum()

# lis = ['ceci', 'être', 'un', 'phrase', 'long', 'ceci', 'être', 'un', 'autre',
#         'phrase', 'encore', 'plus', 'long', 'enfin', 'ceci', 'être', 'le', 'dernier', 'phrase']
# data = getDictionary()
# boolean_list = data.ortho.isin(lis)
# lex = data[boolean_list]
#
# syll = lex['nbsyll'].to_list()

# speech = "Le gouvernement britannique a annoncé jeudi qu’il sanctionnait le milliardaire russe né en " \
#          "Ouzbékistan Alicher Ousmanov déjà mis à l’écart par le club de football d’Everton et l’ancien" \
#          " vice-premier ministre russe Igor Chouvalov après l’invasion de l’Ukraine par la Russie " \
#          "Sanctionner Ousmanov et Chouvalov envoie le message clair que nous frapperons les oligarques et" \
#          " les individus étroitement associés au régime de Poutine et à sa guerre barbare Nous ne nous" \
#          " arrêterons pas là Notre objectif est de paralyser l’économie russe et d’affamer la machine de " \
#          "guerre de Poutine a déclaré la ministre des affaires étrangères britannique Liz Truss dans un communiqué"


# 
# lexic = sf.Lexic(speech, time=10)
# lexic.preprocessing()

# filename = 'chunk_1.wav'
# a = speech_recognition(filename)
#
# print(f'file : {filename} -- {a}')

chunk_name = "chunk_1.wav"
recognizer = sr.Recognizer()

try:
    with sr.AudioFile(chunk_name) as chunk_audio:
        chunk_listened = recognizer.listen(chunk_audio)

        content = recognizer.recognize_google(chunk_listened,language = 'fr-FR')
        texte = " " + content
    # if not recognized

except:
    print('Audio not recognized, retrying.')



print(f'end in {time.time()-t} seconds')
