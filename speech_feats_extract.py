from matplotlib.pyplot import text
from nltk.stem.snowball import FrenchStemmer
from nltk import wordpunct_tokenize          
from nltk.corpus import stopwords
from nltk.corpus import words
from sklearn.feature_extraction.text import CountVectorizer
from string import punctuation
from speech_utils import *
import audio_feats_extract
import pandas as pd
import spacy

class FrenchStemTokenizer(object):

    def __init__(self, remove_non_words=True):
        self.st = FrenchStemmer()
        self.stopwords = set(stopwords.words('french'))
        self.words = set(words.words())
        self.remove_non_words = remove_non_words

    def __call__(self, doc):
        # tokenize words and punctuation
        word_list = wordpunct_tokenize(doc)
        # remove stopwords
        word_list = [word for word in word_list if word not in self.stopwords]
        # remove non words
        if(self.remove_non_words):
            word_list = [word for word in word_list if word in self.words]
        # remove 1-character words
        word_list = [word for word in word_list if len(word)>1]
        # remove non alpha
        word_list = [word for word in word_list if word.isalpha()]
        return [self.st.stem(t) for t in word_list]


class Lexic:

    def __init__(self, audio):

        ## List of raw information not usable for IA models ##
        self._speech = speech_recognition(audio)    ##Done      Raw speech from audio
        self._sentences = []                        ##Done      Sentences detected in speech
        self._words = []                            ##Done      List of words from speech as a set (appearring only once if reppeated)
        self._vec = []                              ##Done      List of words with meaning on itself (vector)
        self._lem = []                              ##Done      List of lemmes
        self._ntm = []                              ##Done      Grammatical type of words as a list of tuples (lemme, gram)
        self._words_Dataset = {}                    ##Done      Pandas dataframe regrouping features from a dictionary (open lexicon as default)

        ## List of values usable by IA models and stat ##
        self.words_per_sentence = []                ##Done      List of sentences lengths in number of words
        self.letters_per_word = []                  ##Done      List of word length in number of letters
        self.gram_count = []                        ##Todo      Number of each gram type in self._ntm

        ## Numbers of different variables ##
        self.word_count = 0                         ##Done      Number of words in all the speech
        self.nb_vec = 0                             ##Done      Number of vec in the speech
        self.nb_lem = 0                             ##Done      Number of lem in the speech
        self.diff_word_count = 0                    ##Done      Number of words in the speech (declinaison considered as a new word)
        self.word_rate = 0                          ##Done      Number of words per second in the speech

        ## Statistics on variables ##
        self.stats_word = {}                        ##Done      Python dict of stats on words length
        self.stats_vec = {}                         ##Done      python dict of stats on vec length
        self.stats_sentence = {}                    ##Done      Python dict of stats on sentences length

        self.lexical_features = {}                  ##Todo      Dataframe of all lexical features

    def __call__(self):
        return vars(self)



    ### Set variable related to words
    def set_words(self):
        '''
            Set the private variable words as a list of all words and word count as the sum of the
            number of words used times their occurence.
        '''

        countvect = CountVectorizer()
        text_fts = countvect.fit_transform([self._speech])
        count_array = text_fts.toarray()

        self._words = list(countvect.get_feature_names_out())
        self.word_count = count_array.sum()


    def set_vec(self):
        '''
            Set vectorized words list as a private argument
        '''

        countvect = CountVectorizer(tokenizer=FrenchStemTokenizer(remove_non_words=True))
        text_fts = countvect.fit_transform([self._speech])

        self._vec = list(countvect.get_feature_names_out())
        self.nb_vec = len(self._vec)

    def set_spacy_feats(self):
        nlp = spacy.load("fr_dep_news_trf")
        info = nlp(self._speech)
        self._lem = list(set([token.lemma_ for token in info]))
        self._sentences = [sentence for sentence in info.sents]
        self._ntm = [(token.lemma_, token.pos_) for token in info]

    def set_dictionnary(self, dictionary=getDictionnary()):
        boolean_list = dictionary.ortho.isin(self._lem)
        self._words_Dataset = dictionary[boolean_list]

    def set_letters_per_word(self):
        '''
            Set the number of letters per word for each word as a list of numbers
        '''
        self.letters_per_word = [len(word) for word in self._words]

    def set_diff_word_count(self):
        self.diff_word_count = len(set(self._words))

    def set_nb_lem(self):
        self.nb_lem = len(set(self._lem))

    def set_speech_rate(self, time):
        self.word_rate = (self.word_count / time)

    def set_words_per_sentence(self):
        '''
            Set number of words per sentences for each sentence as a list of numbers
        '''
        self.words_per_sentence = [len(sentence.split()) for sentence in self._sentences]


    ## Set statistics
    def set_stats_word(self):
        """
            Set stats (mean, median, std, 95c, max) from words.
        """
        self.stats_word = stats([len(word) for word in self._words])

    def set_stats_vec(self):
        """
            Set stats (mean, median, std, 95c, max) from vec.
        """
        self.stats_vec = stats([len(word) for word in self._vec])

    def set_stats_sentence(self):
        """
            Set stats (mean, median, std, 95c, max) from sentences.
        """
        self.stats_sentence = stats([len(sentence) for sentence in self._sentences])


    def preprocessing(self, time):
        pass