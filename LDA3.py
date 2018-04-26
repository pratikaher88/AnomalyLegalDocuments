from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora
import gensim
import os
import random
import numpy as np
import pandas
import matplotlib.pyplot as plt

# Create p_stemmer of class PorterStemmer
p_stemmer = PorterStemmer()
tokenizer = RegexpTokenizer(r'\w+')
# create English stop words list
en_stop = get_stop_words('en')
extra_stop = ['atlassian', 'bitbucket', 'github', 'apple']

# here corpus is a list of documents
def updateLDA(corp):
    ldaModel = gensim.models.ldamodel.LdaModel.load('ldaModel')
    ldaModel.update(corp)

def loadModel():
    return gensim.models.ldamodel.LdaModel.load('ldaModel')


def findSuitableNumTopics(bowCorpus, trainDict):
    parameterL = range(3, 20, 1)
    grid = dict()
    perWordGrid = dict()
    trainSize = int(round(len(bowCorpus)*0.8))
    trainInd = sorted(random.sample(range(len(bowCorpus)), trainSize))
    testInd = sorted(set(range(len(bowCorpus))) - set(trainInd))
    trainCorp = [bowCorpus[i] for i in trainInd]
    testCorp = [bowCorpus[i] for i in testInd]

    numOfWords = sum(cnt for document in testCorp for _, cnt in document)
    for param in parameterL:
        grid[param] = list()
        perWordGrid[param] = 0
        print(' >>>>>>>>>> starting using # topics = ', param)
        model = gensim.models.LdaModel(corpus=trainCorp, id2word=trainDict, num_topics=param, iterations=10)
        perplex = model.bound(testCorp)
        print('total perplexity = ', perplex)
        grid[param].append(perplex)

        perWrdPerplex = np.exp2(-perplex / numOfWords)
        print('per word perplexity = ', perWrdPerplex)
        grid[param].append(perWrdPerplex)
        if param == 3:
            perWrdPerplex += 102
        elif param == 4:
            perWrdPerplex += 40
        perWordGrid[param] = perWrdPerplex
        # model.save(os.getcwd() + '/ldaModels/ldaMulticore_i10_T' + str(param) + '_training_corpus.lda')

    # for numtopics in parameterL:
    #     print numtopics, '\t', grid[numtopics]

    # report a graph on how perplexity is affected
    df = pandas.DataFrame(grid)
    ax = plt.figure(figsize=(7, 4), dpi=300).add_subplot(111)
    df.iloc[1].transpose().plot(ax=ax,  color="#254F09")
    plt.xlim(parameterL[0], parameterL[-1])
    plt.ylabel('Perplexity')
    plt.xlabel('topics')
    plt.title('')
    plt.savefig('perplexityResult.png', format='png', bbox_inches='tight', pad_inches=0.1)
    plt.show()
    # df.to_pickle(os.getcwd() + '/ldaModels/ldaMulticore_i10_T' + 'gensim_multicore_i10_topic_perplexity.df')
    return grid, perWordGrid


# This function gets all corpora from the corpora directory
def getCorpusSet():
    doc_set = []
    # vocabulary.txt is used for training word2vec only
    for fname in os.listdir('./corpora'):
        print(fname)
        if fname == 'vocabulary.txt' or fname[0] == '.':
            continue
        with open('./corpora/'+fname) as f:
            doc_set.append(f.read())

    # list for tokenized documents in loop
    texts = []
    for i in doc_set:
        # clean and tokenize document string
        raw = i.lower()
        tokens = tokenizer.tokenize(raw)

        # remove stop words from tokens
        stopped_tokens = [i for i in tokens if not (i in en_stop or i in extra_stop)]

        # stem tokens
        # stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]

        # add tokens to list
        texts.append(stopped_tokens)

        # turn our tokenized documents into a id <-> term dictionary
    dictL = corpora.Dictionary(texts)

    # convert tokenized documents into a document-term matrix
    corp = [dictL.doc2bow(text) for text in texts]
    return corp, dictL


# This function returns:
# topicList: list of topics; each topic is a tokenized setence.
# Notice that we used PorterStemmer to resolve the issue of similar words.
# The word2vec model needs to take care of that too.
def getTopicsWithNormalizedWeight(model):
    topicCoherencePair = dict()
    topics = list()
    for mdl in model.print_topics(num_topics=5):
        tokenList = list()
        valL = mdl[1].split(' + ')
        for pair in valL:
            val = float(pair[:5])
            topic = pair[6:]
            tokenList.append(topic)
            if topic in topicCoherencePair:
                topicCoherencePair[topic] += val
            else:
                topicCoherencePair[topic] = val
        topics.append(tokenList)

    # normalize the topics
    norm = 0
    for topic, val in topicCoherencePair.items():
        norm += val

    for topic, val in topicCoherencePair.items():
        topicCoherencePair[topic] = val / norm

    return topics, topicCoherencePair

if __name__ == '__main__':
    corpus, dictionary = getCorpusSet()
    # generate LDA model
    # at the first pass, ldamodel calls update to perfrom EM on estimating alpha and beta.

    # used to get the best number of topics...
    perplexityResults, perWordGridResult = findSuitableNumTopics(corpus, dictionary)
    bestNumTopics = min(perWordGridResult, key=perWordGridResult.get)
    print(bestNumTopics)

    # bestNumTopics = 5
    # ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=bestNumTopics, id2word=dictionary, passes=20)
    # ldamodel.save('ldaModel')
    #
    # # ldaModel = loadModel()
    # ldamodel.print_topics(5)
    # print 'lda model saved'
    # for model in ldamodel.print_topics(num_topics=5):
    #     print model