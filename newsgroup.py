import inverted_index as ii
import vector_space_model as vsm

from sklearn.datasets import fetch_20newsgroups


def query_iindex(query, k, newsgroups):

    '''
    Helper function to query the inverted index

    :param query: the query string
    :param k: number of results to display
    :param newsgroups: corpus
    :return: -
    '''

    result = ii.boolean_search(query, iindex)

    print 'The result of the boolean query: {0}'.format(query)

    if result:
        print 'Obtained {0} results'.format(len(result))

        print 'A subset of {0} results'.format(k)
        for r in result[:k]:
            print newsgroups.filenames[r]
    else:
        print 'No result for this query: {0}'.format(query)


def query_tfidf_vectors(query, k, vectorizer, tfidf_vectors, newsgroups):

    '''
    Helper function to query using tfidf vectors

    :param query: query string
    :param k: number of top ranked results
    :param vectorizer: vectorizer object
    :param tfidf_vectors: the tfidf weighted vectors
    :param newsgroups: corpus
    :return: -
    '''

    result = vsm.rank_documents(query, k, vectorizer, tfidf_vectors)

    if result:
        print 'Top {0} documents as result of the query {0}'.format(query)

        for r in result:
            print newsgroups.filenames[r]
    else:
        print 'No result for this query: {0}'.format(query)


if __name__ == '__main__':

    # fetch the 20newsgroups dataset
    newsgroups = fetch_20newsgroups(subset='all')

    # obtain the documents and the words in the 20 newsgroups corpus
    docs, words = ii.read_newsgroups(newsgroups.data)

    print 'Statistics about the corpus'
    print 'Number of documents {0} and number of words {1}'.format(len(docs),
                                                                   len(words))

    print 'Creating the inverted index...'
    iindex = ii.inverted_index(docs, words)

    # perform boolean queries
    query_iindex('science', 5, newsgroups)
    query_iindex('science and religion', 5, newsgroups)
    query_iindex('science or religion', 5, newsgroups)

    print 'Creating the tfidf vectors...'

    vectorizer, tfidf_vectors = vsm.create_vectors(newsgroups)

    # perform queries using tfidf vectors
    query_tfidf_vectors('science religion', 5, vectorizer, tfidf_vectors,
                        newsgroups)

