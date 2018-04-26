'''
K-means gives a base-line model.
Also K-means is a run-time training. We call k-means on each incoming new model to test the
document-specific centroids.

LDA preprocess the documents and generates topics.

It seems that LDA can target at more accurate clustering by estimating topics of each section
Reference = http://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf

And performance can be evaluated by:
Reference = http://dirichlet.net/pdf/wallach09evaluation.pdf
'''

# wordcloud, can use this to generate word graphs

import sys
import nltk
from numpy import dot, mean, array
import Kmeans
import vocabBuilder
import htmlParser
import LDA3
import LOF
from nltk.tokenize import sent_tokenize
from difflib import SequenceMatcher


def getFinalKmeans(text):
    word2vecModel = vocabBuilder.loadModel()
    # print topics
    # print topics
    # example of analyzing EULA of atlassian.
    # inputSrc = './atlassianLegalDocDir/EULA.html'
    # inputDir = './atlassianLegalDocDir'
    #
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    # emphList, h1List, h2List, textList = htmlParser.extractTextInfo(inputDir, inputSrc)
    # compile a topic list
    def mergeEmphLists(emphLists):
        resultTopics = list()
        for emphL in emphLists:
            for emph in emphL:
                if len(emph) != 0:
                    resultTopics.append(emph)
        return resultTopics

    # topics = mergeEmphLists([emphList, h1List, h2List])
    #
    # minTokenNum = 4
    # tokenList = list()
    # sentenceList = list()
    # # this is atlassian specific
    # for text in textList[0]:
    #     sentences = vocabBuilder.tokenize(text)
    #     for sentence in sentences:
    #         # print sentence
    #         tokens = list()
    #         tmpTokens = sentence.split()
    #         for token in tmpTokens:
    #             if token not in topics:
    #                 tokens.append(token)
    #
    #         # tokens = sentence.split()
    #         if len(tokens) >= minTokenNum:
    #             tokenList.append(sentence.split())
    #             sentenceList.append(sentence)
    # task1: how many centroids shall we have?
    # The only thing we can control right now is really the number of centroids...
    # use Silhouette Coefficent to evaluate how robust the clustering is
    # # Manual Test Part
    # inputSrc = './appleLegalDocDir/apple-osx-test.txt'
    inputSrc = './sampleTest.txt'
    minTokenNum = 4
    tokenList = list()
    sentenceList = list()
    # with open(inputSrc) as f:
    #     text = f.read()

    # task 1.1 use LDA to filter out common topics. This way the main popular topic words won't affect
    # calculation of similarities.
    ldaModel = LDA3.loadModel()
    topics, topicCoherencePair = LDA3.getTopicsWithNormalizedWeight(ldaModel)
    sentences = vocabBuilder.tokenize(text)
    for sentence in sentences:
        tokens = list()
        tmpTokens = sentence.split()
        for token in tmpTokens:
            if token not in topics:
                tokens.append(token)

        if len(tokens) >= minTokenNum:
            tokenList.append(tokens)
            sentenceList.append(sentence)
    print(len(sentenceList))
    # debug test
    # size = 1
    # musR, assignmentsIndexListR, assignmentsR, vectorListR = Kmeans.kmeans(tokenList, word2vecModel, 1000, size)
    # silhouetteScoreList = Kmeans.silhouetteScore(musR, assignmentsIndexListR, vectorListR)
    # meanList.append(mean(array(silhouetteScoreList)))
    #
    # meanList = list()
    # for size in range(1, 5):
    #     print size
    #     musR, assignmentsIndexListR, assignmentsR, vectorListR = Kmeans.kmeans(tokenList, word2vecModel, 1000, size)
    #     silhouetteScoreList = Kmeans.silhouetteScore(musR, assignmentsIndexListR, vectorListR)
    #     if len(silhouetteScoreList) == 0:
    #         break
    #     meanList.append(mean(array(silhouetteScoreList)))
    #     print meanList
    #
    # clusterNum = [x for x in range(1, 30)][meanList.index(max(meanList))]
    # print 'best number of clusters is: ', clusterNum
    # Rerun kmeans with the best cluster structure:
    musR, assignmentsIndexListR, assignmentsR, vectorListR = Kmeans.kmeans(tokenList, word2vecModel, 1000, 1)
    # task 2.1: anormaly detection:
    # Use Local Outlier Factor to detect if a point is likely to be an anormaly.
    lof = LOF.LOF(musR, assignmentsIndexListR, vectorListR, 6)
    lofList = list()
    for ptIndex in range(len(assignmentsIndexListR)):
        lofPt = lof.calcLOF(ptIndex)
        lofList.append(lofPt)
        print(ptIndex, lofPt)
    lofIdList = list()
    anormalySentences = list()
    for lofId in range(len(lofList)):
        if lofList[lofId] > 1.0:  # now try to find inliners
            # print sentenceList[lofId]
            lofIdList.append(lofId)
            anormalySentences.append((sentenceList[lofId], lofList[lofId], vectorListR[lofId], lofId))
    anormalySentencesReversed = sorted(anormalySentences, key=lambda x: x[1], reverse=True)
    anormalySentencesIds = [sentence[3] for sentence in anormalySentencesReversed]
    # for sen in anormalySentencesReversed:
    #     print sen
    print(len(anormalySentences))
    anomaloussentences = []
    for sentence in anormalySentencesReversed:
        anomaloussentences.append(sentence[0])
        print(sentence[0], sentence[1], sentence[3])

    # # task 3: Visualization:
    # t-SNE
    import numpy as np
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    tsneModel = TSNE(n_components=3, random_state=0, metric='precomputed')
    np.set_printoptions(suppress=True)
    r = np.zeros(shape=(len(vectorListR), len(vectorListR)))
    for xInd in range(len(vectorListR)):
        x = vectorListR[xInd]
        yList = list()
        for y in vectorListR:
            precomputeResult = 1 - dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
            if precomputeResult < 0:
                print(precomputeResult)
                precomputeResult = 0
            yList.append(precomputeResult)
        r[xInd] = yList
    result = tsneModel.fit_transform(r)
    print(result)
    x = list()
    y = list()
    z = list()
    xa = list()
    ya = list()
    za = list()
    for prInd in range(len(result)):
        pr = result[prInd]
        if prInd not in anormalySentencesIds:
            if abs(pr[0]) <= 50 and abs(pr[1]) <= 50:
                x.append(pr[0])
                y.append(pr[1])
                z.append(pr[2])
        else:
            xa.append(pr[0])
            ya.append(pr[1])
            za.append(pr[2])
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(array(x), array(y), array(z), c='y', s=20, marker='o')
    ax.scatter(array(xa), array(ya), array(za), c='b', s=30, marker='^')
    plt.xlim(-50, 50)
    plt.ylim(-50, 50)
    plt.show()

    #
    # x = list()
    # y = list()
    # for xRL in result:
    #     x.append(xRL[0])
    #     y.append(xRL[1])
    #
    # plt.scatter(x, y)
    #
    # tsneModelWithAnormaly = TSNE(n_components=2, random_state=0)
    # np.set_printoptions(suppress=True)
    #
    # anormalyVectorList = list()
    # for anomId in range(len(anormalySentencesReversed)):
    #     anormalyVectorList.append(1.0 / anormalySentencesReversed[anomId][2])
    #
    # print len(anormalyVectorList)
    #
    # nResult = tsneModelWithAnormaly.fit_transform(anormalyVectorList)
    #
    # x1 = list()
    # y1 = list()
    # for xRL in nResult:
    #     x1.append(xRL[0])
    #     y1.append(xRL[1])
    #
    #
    # plt.scatter(x1, y1, c='g')
    #
    # plt.show()
    #
    def getDim(dim, mat):
        dimArr = list()
        for row in mat:
            dimArr.append(row[dim])

        return dimArr

    print(lofIdList)
    # # Result using PCA:
    from sklearn.decomposition import PCA
    pca = PCA(n_components=3)
    pca.fit(vectorListR)
    result = pca.transform(vectorListR)
    print(len(result))
    pt1 = getDim(0, result)
    pt2 = getDim(1, result)
    pt3 = getDim(2, result)
    px = list()
    py = list()
    pz = list()
    pxa = list()
    pya = list()
    pza = list()
    for senId in range(len(result)):
        pr = result[senId]
        if senId not in anormalySentencesIds:
            px.append(pt1[senId])
            py.append(pt2[senId])
            pz.append(pt3[senId])
        else:
            pxa.append(pt1[senId])
            pya.append(pt2[senId])
            pza.append(pt3[senId])
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, projection='3d')
    ax1.scatter(array(px), array(py), array(pz), c='y', s=20, marker='o')
    ax1.scatter(array(pxa), array(pya), array(pza), c='b', s=30, marker='^')
    plt.xlim(-0.2, 0.2)
    plt.ylim(-0.2, 0.2)

    for sentence in anomaloussentences:
        print(sentence)

    # plt.show()
    newdict={}

    id=0
    newdict2={}
    for sentence in sent_tokenize(text):
        newdict2[id]=sentence
        print("Original wala Value",id,newdict2[id])
        id+=1
    #
    # # print(len(newdict))
    #
    # # for id,value in newdict2.items():
    # #     print(id,value)
    #
    id_print=[]
    for sent in anomaloussentences:
        compare={}
        for id,prevsnet in newdict2.items():
            seq = SequenceMatcher(None, sent, prevsnet)
            d = seq.ratio() * 100
            compare[id]=d
        id_print.append(max(compare, key=compare.get))
    #
    print(id_print)
    counter = 0
    for item in newdict2:
        if item in id_print:
            # newdict2[item]= colored(newdict2[item],'blue')
            newdict2[item] = '<span style="color:#ff0000">'+newdict2[item]+'</span>'
            # newdict2[item] = '\033[1;31m;40m' + newdict2[item] + '\033[1;m'
        counter += 1

    output_string = ''

    for id, value in newdict2.items():
        output_string = output_string + value
        print(value)

    return output_string
    #
    # # for id,value in newdict2.items():
    # #     print(id,value)
    #
    # f = open('outputfile.txt', 'w')
    #
    # for id,value in newdict2.items():
    #     print(value)
    #
    # f.close()
    # plt.scatter(np.array(getDim(0, result)), np.array(getDim(1, result)))
    # # task 4: Do we still want to have SVM? Seems not quite necessary though...
    #
