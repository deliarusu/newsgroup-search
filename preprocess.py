import nltk
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()


def stem_tokens(tokens):

    '''
    Stem tokens using the Porter Stemmer

    :param tokens: list of tokens to stem
    :return: stemmed tokens
    '''

    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed


def tokenize(text):

    '''
    Tokenize and stem text received as input

    :param text: text to tokenize
    :return: tokenized and stemmed text
    '''

    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens)
    return stems
