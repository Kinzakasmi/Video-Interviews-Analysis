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

    def __init__(self, audio):

        self.__speech = speech_recognition(audio)   ##Kinda Done
        self.__sentences = []                       ##Done
        self.__words = []                           ##Done
        self.word_count = 0                         ##Done
        self.words_per_sentence = []                ##Done
        self.letters_per_word = []                  ##Done
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
        self.average_word_len = 0                   ##Done
        self.longest_sentence = 0
        self.average_sentence_len = 0               ##Done
        self.type_token_ratio = 0
        self.tot_vocab = 0
        self.unique_word_count = 0
        self.diff_word_count = 0
        self.complexity = 0
        self.rate_of_speech = 0

    def __call__(self):
        return vars(self)


    ### Set variable related to words
    def set_words(self):
        '''
        Set the private variable words as a list of all words
        '''
        self.__words = self.__speech.split()

    def set_word_count(self) :
        '''
        Set number of words in the speech
        '''
        self.word_count = len(self.__words)

    def set_average_word_len(self):
        '''
        Set the average word length
        '''
        nb_letters = [len(word) for word in self.__speech.split()]
        self.average_word_len = sum(nb_letters)/len(nb_letters)

    def set_letters_per_word(self):
        '''
        Set the number of letters per word for each word as a list of numbers
        '''
        self.letters_per_word = [len(word) for word in self.__speech.split()]

    ### Set variables related to sentences
    def set_sentences(self):
        '''
        Split speech as sentences using . ! ? ... as delimiter in a private variable
        '''
        sentences = [sent.split('?') for sent in self.__speech.split('!')]
        sentences = [sent.split('...') for sent in sentences]

        self.__sentences = [sent.split('.') for sent in sentences]

    def set_words_per_sentence(self):
        '''
        Set number of words per sentences for each sentence as a list of numbers
        '''
        self.words_per_sentence = [len(sentence.split()) for sentence in self.__sentences]

    def set_average_sentence_len(self):
        self.average_sentence_len = sum(self.words_per_sentence)/len(self.words_per_sentence)

    def set_longest_sentence(self):
        '''
        Set highest number for word per in a sentence of the speech
        '''
        self.longest_sentence = max([len(sentence.split()) for sentence in self.__sentences])
