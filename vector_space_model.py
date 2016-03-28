from sklearn.feature_extraction.text import TfidfVectorizer
from preprocess import tokenize
import numpy


def create_vectors(newsgroups):

    '''
    Create tfidf vectors for the 20 newsgroup dataset

    :param newsgroups:
    :return: the vectorizer object and tfidf vectors
    '''

    # the tfidf vectorizer using a custom tokenizer and a list of English
    # stop-words
    vectorizer = TfidfVectorizer(tokenizer=tokenize, stop_words='english')

    # fit the vectorizer using the 20 newsgroup corpus
    tfidf_vectors = vectorizer.fit_transform(newsgroups.data)

    # return the tfidf vectorizer object and the vectors obtained after fitting
    # the corpus
    return vectorizer, tfidf_vectors


def rank_documents(query, k, vectorizer, tfidf_vectors):

    '''
    Rank documents based on a query, using the tfidf vectors

    :param query: the query
    :param k: the number of documents to return
    :param vectorizer: the vectorizer object
    :param tfidf_vectors: the tfidf vectors fitted for the 20 newsgroups corpus
    :return: top k ranked document according to the query
    '''

    # transform the query
    result = vectorizer.transform([query])

    # get the terms of the vectorizer
    terms = vectorizer.get_feature_names()

    # if the query consists of multiple terms, aggregate the result in this
    # accumulator
    accumulator = numpy.zeros(tfidf_vectors.shape[0])

    assert k < tfidf_vectors.shape[0], 'k should be smaller than the number ' \
                                       'of documents'

    for col in result.nonzero()[1]:

        # a column corresponding to a term in the query, for now a sparse
        # representation
        column = tfidf_vectors[:, col]

        # convert the sparse respresentation to a dense matrix
        dense_column = column.todense()

        # reshape to a numpy array
        reshaped_dense_column = numpy.asarray(dense_column).reshape(-1)

        # add the column to the accumulator
        accumulator += reshaped_dense_column

    # sort the documents
    sorted_column_index = numpy.argsort(accumulator)

    # create a slice only with the top-k results
    result = sorted_column_index[-k:].tolist()

    return result[::-1]
