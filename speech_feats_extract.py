from nltk.stem.snowball import FrenchStemmer
from nltk import wordpunct_tokenize          
from nltk.corpus import stopwords
from nltk.corpus import words
from string import punctuation
from speech_utils import *
import audio_feats_extract

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


class LexicalFeatures:

    def __init__(self, filename):

        self.__speech = speech_recognition(filename)
        self.word_count = 0
        self.words_per_sentence = 0
        self.letters_per_word = 0
        self.dic_words = 0
        self.func_words = 0
        self.i = {}
        self.article = {}
        self.past = {}
        self.present = {}
        self.adverb = {}
        self.social = {}
        self.negemo = {}
        self.sad = {}
        self.work = {}
        self.period = {}
        self.average_word_len = 0
        self.longest_sentence = 0
        self.average_sentence_len = 0
        self.type_token_ratio = 0
        self.tot_vocab = 0
        self.unique_word_count = 0
        self.diff_word_count = 0
        self.complexity = 0
        self.rate_of_speech = 0

    def __call__(self):
        return vars(self)

