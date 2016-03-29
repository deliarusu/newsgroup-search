import ast
import _ast


def read_newsgroups(newsgroups_data):

    '''
    Read newsgroups and return the documents in the newsgroups corpus and
    all the words

    :param newsgroups_data: the newsgroups corpus
    :return: documents in the corpus and all words
    '''

    docs, words = {}, set()

    # use only lowercasing as a pre-processing step
    # this can be further extended to use tokenization, stemming, stop-words
    # e.g. the output of the TfidfVectorizer
    for idx, item in enumerate(newsgroups_data):
        txt = item.split()
        txt = [t.lower() for t in txt]
        words |= set(txt)
        docs[idx] = txt
    return docs, words


def inverted_index(docs, words):

    '''
    Generate the inverted index based on the documents and the words

    :param docs: the documents in the collection as a dictionary with
    document id keys and words as values
    :param words: all words in the collection
    :return: inverted index
    '''

    # initialize the inverted index for faster index generation
    iindex = {word: [] for word in words}

    for did, txt in docs.items():
        for word in txt:
            iindex[word] += [did]

    return iindex


def traverse_syntax_tree(tree, iindex):

    '''
    Traverse the syntax tree representing a boolean query. By traversing the
    tree we can evaluate the query.

    :param tree: syntax tree
    :param iindex: inverted index
    :return: list of document ids for the query
    '''

    assert isinstance(tree, _ast.BoolOp), 'Only boolean operators are ' \
                                               'allowed'

    doc_ids = [None, None]
    for i in [0, 1]:
        if isinstance(tree.values[i], _ast.Name): # recursion ends
            term = tree.values[i].id
            doc_ids[i] = iindex.get(term) or []
        else:                                     # recursive call
            doc_ids[i] = traverse_syntax_tree(tree.values[i], iindex)

    # if it is an OR, then perform set union
    if isinstance(tree.op, _ast.Or):
        return set(doc_ids[0]) | set(doc_ids[1])
    # if it is an AND, perform set intersection
    elif isinstance(tree.op, _ast.And):
        return set(doc_ids[0]) & set(doc_ids[1])
    else:
        print 'Not a supported boolean operator'
        raise Exception


def boolean_search(query, iindex):

    '''
    Perform boolean search using arbitrary AND or OR operators

    :param query: the query to perform
    :param iindex: the inverted index
    :return: a list of document ids matching the query
    '''

    if not query:
        print 'No query received as input'
        return

    # convert query is lowercase
    query = query.lower()

    # special case if the query only contains one term
    query_elems = query.split()
    if len(query_elems) <= 1:
        return list(iindex.get(query))

    # use a syntax tree to parse the query
    tree = ast.parse(query, mode='eval')

    # traverse the syntax tree to obtain a result
    return list(traverse_syntax_tree(tree.body, iindex))



