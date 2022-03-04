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
        if (self.remove_non_words):
            word_list = [word for word in word_list if word in self.words]
        # remove 1-character words
        word_list = [word for word in word_list if len(word) > 1]
        # remove non alpha
        word_list = [word for word in word_list if word.isalpha()]
        return [self.st.stem(t) for t in word_list]


class Lexic:

    def __init__(self, audio, time):

        ## List of raw information not usable for IA models ##
        self._speech = speech_recognition(audio)    ##Done      Raw speech from audio
        self._sentences = []            ##Done      Sentences detected in speech
        self._words = []                ##Done      List of words from speech as a set (appearring only once if reppeated)
        self._vec = []                  ##Done      List of words with meaning on itself (vector)
        self._lem = []                  ##Done      List of lemmes
        self._ntm = []                  ##Done      Grammatical type of words as a list of tuples (lemme, gram)
        self._words_Dataset = {}        ##Done      Pandas dataframe regrouping features from a dictionary (open lexicon as default)

        ## List of values usable by IA models and stat ##
        self.words_per_sentence = []    ##Done      List of sentences lengths in number of words
        self.letters_per_word = []      ##Done      List of word length in number of letters
        self.syll = []                  ##Done      List of syllabs per lemme of words used
        self.phon = []                  ##Done      list of phonems per lemme of words used
        self.gram = {}                  ##Done      Number of each gram type in self._ntm
        self.film_occ = []              ##Done      List of occurrence of the word in a film on a given corpus (see Open Lexicon)
        self.book_occ = []              ##Done      List of occurrence of the word in a book on a given corpus (see Open Lexicon)

        ## Numbers of different variables ##
        self.word_count = 0         ##Done      Number of words in all the speech
        self.diff_word_count = 0    ##Done      Number of words in the speech (declinaison considered as a new word)
        self.nb_vec = 0             ##Done      Number of vec in the speech
        self.nb_lem = 0             ##Done      Number of lem in the speech
        self.word_rate = 0          ##Done      Number of words per second in the speech

        ## Statistics on variables ##
        self.stats_word = {}        ##Done      Python dict of stats on words length
        self.stats_vec = {}         ##Done      python dict of stats on vec length
        self.stats_sentence = {}    ##Done      Python dict of stats on sentences length
        self.stats_syll = {}        ##Done      Python dict of stats on sentences length
        self.stats_film_occ = {}    ##Done      Python dict of stats on sentences length
        self.stats_book_occ = {}    ##Done      Python dict of stats on sentences length
        self.stats_phon = {}
        ## Assembling all usable data in a pandas dataframe ##
        self.lexical_features = pd.DataFrame()  ##Done      Dataframe of all lexical features

        self.set_ALL(time)

    def __call__(self):
        return {
            'wrd_cnt': self.word_count,
            'dwrd_cnt': self.diff_word_count,
            'nb_vec': self.nb_vec,
            'nb_lem': self.nb_lem,
            'w_rate': self.word_rate,

            'ADJ': self.gram['ADJ'],
            'ADP': self.gram['ADP'],
            'ADV': self.gram['ADV'],
            'AUX': self.gram['AUX'],
            'CONJ': self.gram['CONJ'],
            'CCONJ': self.gram['CCONJ'],
            'DET': self.gram['DET'],
            'INTJ': self.gram['INTJ'],
            'NOUN': self.gram['NOUN'],
            'NUM': self.gram['NUM'],
            'PART': self.gram['PART'],
            'PRON': self.gram['PRON'],
            'PROPN': self.gram['PROPN'],
            'PUNCT': self.gram['PUNCT'],
            'SCONJ': self.gram['SCONJ'],
            'SYM': self.gram['SYM'],
            'VERB': self.gram['VERB'],
            'X': self.gram['X'],
            'EOL': self.gram['EOL'],
            'SPACE': self.gram['SPACE'],

            'wrd_min': self.stats_word['min'],
            'wrd_mean': self.stats_word['mean'],
            'wrd_med': self.stats_word['median'],
            'wrd_std': self.stats_word['std'],
            'wrd_95c': self.stats_word['95c'],
            'wrd_max': self.stats_word['max'],

            'vec_min': self.stats_vec['min'],
            'vec_mean': self.stats_vec['mean'],
            'vec_med': self.stats_vec['median'],
            'vec_std': self.stats_vec['std'],
            'vec_95c': self.stats_vec['95c'],
            'vec_max': self.stats_vec['max'],

            'stc_min': self.stats_sentence['min'],
            'stc_mean': self.stats_sentence['mean'],
            'stc_med': self.stats_sentence['median'],
            'stc_std': self.stats_sentence['std'],
            'stc_95c': self.stats_sentence['95c'],
            'stc_max': self.stats_sentence['max'],

            'syll_min': self.stats_syll['min'],
            'syll_mean': self.stats_syll['mean'],
            'syll_med': self.stats_syll['median'],
            'syll_std': self.stats_syll['std'],
            'syll_95c': self.stats_syll['95c'],
            'syll_max': self.stats_syll['max'],

            'film_min': self.stats_film_occ['min'],
            'film_mean': self.stats_film_occ['mean'],
            'film_med': self.stats_film_occ['median'],
            'film_std': self.stats_film_occ['std'],
            'film_95c': self.stats_film_occ['95c'],
            'film_max': self.stats_film_occ['max'],

            'book_min': self.stats_book_occ['min'],
            'book_mean': self.stats_book_occ['mean'],
            'book_med': self.stats_book_occ['median'],
            'book_std': self.stats_book_occ['std'],
            'book_95c': self.stats_book_occ['95c'],
            'book_max': self.stats_book_occ['max'],

            'phon_min': self.stats_phon['min'],
            'phon_mean': self.stats_phon['mean'],
            'phon_med': self.stats_phon['median'],
            'phon_std': self.stats_phon['std'],
            'phon_95c': self.stats_phon['95c'],
            'phon_max': self.stats_phon['max']
        }

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
        self._sentences = [sentence.text for sentence in info.sents]
        self._ntm = [(token.lemma_, token.pos_) for token in info]

    def set_dictionary(self, dictionary=getDictionary()):
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

    def set_syll(self):
        self.syll = [syll for syll in self._words_Dataset['nbsyll'].to_list()]

    def set_film_occ(self):
        self.film_occ = [freq for freq in self._words_Dataset['freqfilms2'].to_list()]

    def set_book_occ(self):
        self.book_occ = [freq for freq in self._words_Dataset['freqlivres'].to_list()]

    def set_phon(self):
        self.phon = [phon for phon in self._words_Dataset['nbphons'].to_list()]

    def set_gram(self):
        gr = [c for (_, c) in self._ntm]
        self.gram = {
            "ADJ": gr.count("ADJ"),
            "ADP": gr.count("ADP"),
            "ADV": gr.count("ADV"),
            "AUX": gr.count("AUX"),
            "CONJ": gr.count("CONJ"),
            "CCONJ": gr.count("CCONJ"),
            "DET": gr.count("DET"),
            "INTJ": gr.count("INTJ"),
            "NOUN": gr.count("NOUN"),
            "NUM": gr.count("NUM"),
            "PART": gr.count("PART"),
            "PRON": gr.count("PRON"),
            "PROPN": gr.count("PROPN"),
            "PUNCT": gr.count("PUNCT"),
            "SCONJ": gr.count("SCONJ"),
            "SYM": gr.count("SYM"),
            "VERB": gr.count("VERB"),
            "X": gr.count("X"),
            "EOL": gr.count("EOL"),
            "SPACE": gr.count("SPACE")
        }

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

    def set_stats_syll(self):
        """
            Set stats (mean, median, std, 95c, max) from syllables in words.
        """
        self.stats_syll = stats(self.syll)

    def set_stats_film_occ(self):
        """
            Set stats (mean, median, std, 95c, max) from occurrence of words in films.
        """
        self.stats_film_occ = stats(self.film_occ)

    def set_stats_book_occ(self):
        """
            Set stats (mean, median, std, 95c, max) from occurrence of words in books.
        """
        self.stats_book_occ = stats(self.book_occ)

    def set_stats_phon(self):
        """
            Set stats (mean, median, std, 95c, max) from phonems in words.
        """
        self.stats_phon = stats(self.phon)

    def set_ALL(self, time):
        self.set_words()
        self.set_vec()
        self.set_spacy_feats()
        self.set_dictionary()
        self.set_letters_per_word()
        self.set_diff_word_count()
        self.set_nb_lem()
        self.set_speech_rate(time)
        self.set_words_per_sentence()
        self.set_syll()
        self.set_film_occ()
        self.set_book_occ()
        self.set_phon()
        self.set_gram()
        self.set_stats_word()
        self.set_stats_vec()
        self.set_stats_sentence()
        self.set_stats_syll()
        self.set_stats_film_occ()
        self.set_stats_book_occ()
        self.set_stats_phon()

    def preprocessing(self):
        return pd.DataFrame(self())
