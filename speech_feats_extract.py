from nltk.stem.snowball import FrenchStemmer
from nltk import wordpunct_tokenize          
from nltk.corpus import stopwords
from nltk.corpus import words
from sklearn.feature_extraction.text import CountVectorizer
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

        self._speech = speech_recognition(audio)    ##Kinda Done
        self._sentences = []                        ##Done
        self._words = []                            ##Done
        self._vec = []                              ##Done
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
        self.longest_sentence = 0                   ##Done
        self.average_sentence_len = 0               ##Done
        self.type_token_ratio = 0
        self.tot_vocab = 0                          ##MaybeDone
        self.unique_word_count = 0
        self.diff_word_count = 0                    ##Done
        self.complexity = 0
        self.rate_of_speech = 0

    def __call__(self):
        return vars(self)



    ### Set variable related to words
    def set_words(self):
        '''
            Set the private variable words as a list of all words and word count as the sum of the
            number of words used times their occurence.
        '''

        countvect = CountVectorizer()
        text_fts = countvect.fit_transform(self._speech)
        count_array = text_fts.toarray()

        self._words = list(countvect.get_feature_names_out())
        self.word_count = count_array.sum()


    def set_vec(self):
        '''
            Set vectorized words list as a private argument
        '''

        countvect = CountVectorizer(tokenizer=FrenchStemTokenizer(remove_non_words=True))
        text_fts = countvect.fit_transform([self._speech])

        self._vect = list(countvect.get_feature_names_out())
        self.tot_vocab = len(self._vect)

    def set_average_word_len(self):
        '''
            Set the average word length
        '''
        self.average_word_len = sum(self.letters_per_word)/len(self.letters_per_word)

    def set_letters_per_word(self):
        '''
            Set the number of letters per word for each word as a list of numbers
        '''
        self.letters_per_word = [len(word) for word in self._words]

    def set_diff_word_count(self):
        self.diff_word_count = len(self._words)



    ### Set variables related to sentences
    def set_sentences(self):
        '''
            Split speech as sentences using . ! ? ... as delimiter in a private variable
        '''
        sentences = [sent.split('?') for sent in self._speech.split('!')]
        sentences = [sent.split('...') for sent in sentences]

        self._sentences = [sent.split('.') for sent in sentences]

    def set_words_per_sentence(self):
        '''
            Set number of words per sentences for each sentence as a list of numbers
        '''
        self.words_per_sentence = [len(sentence.split()) for sentence in self._sentences]

    def set_average_sentence_len(self):
        '''
            Set average length of sentences in analysed speech
        '''
        self.average_sentence_len = sum(self.words_per_sentence)/len(self.words_per_sentence)

    def set_longest_sentence(self):
        '''
            Set highest number for words in a sentence of the speech
        '''
        self.longest_sentence = max([len(sentence.split()) for sentence in self._sentences])

