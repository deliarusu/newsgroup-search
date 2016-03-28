import inverted_index as ii
import vector_space_model as vsm

from sklearn.datasets import fetch_20newsgroups

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

    query = 'science and (religion or technology)'

    result = ii.boolean_search(query, iindex)

    print 'The result of the boolean query: {0}'.format(query)

    print 'Obtained {0} results'.format(len(result))

    for r in result:
        print newsgroups.filenames[r]

    print 'Creating the tfidf vectors...'

    vectorizer, tfidf_vectors = vsm.create_vectors(newsgroups)

    query = 'science religion'

    result = vsm.rank_documents(query, 5, vectorizer, tfidf_vectors)

    print 'Top {0} documents as result of the query {0}'.format(query)

    for r in result:
        print newsgroups.filenames[r]