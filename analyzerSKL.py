import sys
from termcolor import colored
from numpy import array
import vocabBuilder
import gensim
import LOF
import LDA3
from colorama import Fore
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from difflib import SequenceMatcher
from nltk.tokenize import sent_tokenize

# reload(sys)
# sys.setdefaultencoding('utf8')

def getFinalText(text):

    # inputSrc = './sampleTest.txt'
    # inputSrc = './appleLegalDocDir/apple-osx-test.txt'
    minTokenNum = 4
    tokenList = list()
    sentenceList = list()
    # with open(inputSrc) as f:
    #     text = f.read()

    # Task 0: use LDA to filter out common topics. This way the main popular topic words won't affect
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
    print('There are ', len(sentenceList), ' sentences in the input sample.')
    word2vecModel = vocabBuilder.loadModel()
    sentenceVecModelList = list()
    for tokenSen in tokenList:
        v1 = []
        for word in tokenSen:
            if word not in word2vecModel.wv.vocab:
                v1.append([-1.0] * 100)
            else:
                v1.append(word2vecModel[word])
        # normalize v1 with mean 0.
        sentenceVecModelList.append(gensim.matutils.unitvec(array(v1).mean(axis=0)))

    ## Task 1: Try AgglomerativeClustering:
    aggModelList = list()
    silhouette_avg_list = list()
    for n_cluster in range(2, 40):
        aggModel = AgglomerativeClustering(n_clusters=n_cluster,
                                           linkage='average', affinity='cosine')
        aggModel.fit(sentenceVecModelList)
        aggModelList.append(aggModel)
        silhouette_avg = silhouette_score(array(sentenceVecModelList), aggModel.labels_)
        silhouette_avg_list.append(silhouette_avg)
    print(silhouette_avg_list)
    n_clusters = 3

    # Task 1 reports that three clusters give the best division...
    testAggModel = AgglomerativeClustering(n_clusters=n_clusters,
                                           linkage='average', affinity='cosine')
    testAggModel.fit(sentenceVecModelList)
    labels = testAggModel.labels_.tolist()
    print(labels)
    # find labels with the farthest distances...
    clusterList = [list() for _ in range(n_clusters)]
    for senVecId in range(len(sentenceVecModelList)):
        clusterList[labels[senVecId]].append(senVecId)

    # filter out sentences that reside in one cluster. If such a sentence is found, then this sentence is an anormaly.
    clusterAnormalySentences = list()
    removeInds = list()
    for ptsInd in range(len(clusterList)):
        print(ptsInd)
        pts = clusterList[ptsInd]
        if len(pts) == 1:
            print(sentences[pts[0]])
            clusterAnormalySentences.append(sentences[pts[0]])
            removeInds.append(ptsInd)

    # for i in removeInds:
    #     pts = clusterList[i]
    #     clusterList.remove(pts)
    #     labels.remove(i)
    # use LOF on the rest of the cluster
    mus = list()
    for cluster in clusterList:
        # get the centroid value
        numOfPts = len(cluster)
        mu = sum([sentenceVecModelList[pt] for pt in cluster]) / numOfPts
        mus.append(mu)
    lof = LOF.LOF(mus, labels, sentenceVecModelList, 6)
    lofList = list()
    for ptIndex in range(len(labels)):
        lofPt = lof.calcLOF(ptIndex)
        lofList.append(lofPt)
        print(ptIndex, lofPt)
    lofIdList = list()
    anormalySentences = list()
    for lofId in range(len(lofList)):
        if lofList[lofId] > 1.0:  # now try to find inliners
            # print sentenceList[lofId]
            lofIdList.append(lofId)
            anormalySentences.append((sentenceList[lofId], lofList[lofId], sentenceVecModelList[lofId], lofId))

    # def similar(a, b):
    #     return SequenceMatcher(None, a, b).ratio()
    print(len(anormalySentences))
    anormalySentencesReversed = sorted(anormalySentences, key=lambda x: x[1], reverse=True)
    anomaloussentences = []
    for sentence in anormalySentencesReversed:
        anomaloussentences.append(sentence[0])
        print(colored("Anomaly", 'red'), sentence[0], sentence[1])
        # for value in sentences:
        #     if similar(value,sentence[0])==1.0:
        #         new_val=Fore.RED+sentence[0]
        #         value= new_val
        #         print(value)
    newdict = {}
    id = 0
    newdict2 = {}
    for sentence in sent_tokenize(text):
        newdict2[id] = sentence
        id += 1

    # print(len(newdict))
    # for id,value in newdict2.items():
    #     print(id,value)
    id_print = []
    for sent in anomaloussentences:
        compare = {}
        for id, prevsnet in newdict2.items():
            seq = SequenceMatcher(None, sent, prevsnet)
            d = seq.ratio() * 100
            compare[id] = d
        id_print.append(max(compare, key=compare.get))
    print(id_print)
    counter = 0
    for item in newdict2:
        if item in id_print:
            # newdict2[item]= colored(newdict2[item],'blue')
            newdict2[item] = '<span style="color:#ff0000">'+newdict2[item]+'</span>'
            # newdict2[item] = '\033[1;31m;40m' + newdict2[item] + '\033[1;m'
        counter += 1

    # for id,value in newdict2.items():
    #     print(id,value)
    # f = open('outputfile.txt', 'w')

    output_string=''

    for id, value in newdict2.items():
        output_string= output_string+value
        print(value)

    return output_string
        # f.close()


        # nlp=spacy.load('en')
        #
        # for sent in anomaloussentences:
        #     doc1 = nlp(sent)
        #     for prevsnet in sentenceList:
        #         doc2=nlp(prevsnet)
        #         print(doc1.similarity(doc2))
        #         counter = 0
        #         org_tokens= word_tokenize(prevsnet)
        #         for token in ano_tokens:
        #             if token in org_tokens:
        #                 counter+=1
        #         newdict[prevsnet] = counter


        # for sent in anomaloussentences:
        #     ano_tokens = word_tokenize(sent)
        #     for prevsnet in sentenceList:
        #         org_tokens= word_tokenize(prevsnet)
        #         print(ano_tokens)
        #         print(org_tokens)
        #         ratio = float(len(set(ano_tokens).intersection(org_tokens))) / float(len(set(ano_tokens).union(org_tokens)))
        #         if ratio>0.5:
        #             newdict[prevsnet]=ratio
        #
        # for word,value in newdict.items():
        #     print(word,value)
        #
        # tokenList = list()
        # sentenceList = list()
        #
        # sentences = vocabBuilder.tokenize(text)
        # for sentence in sentences:
        #     tokens = list()
        #     tmpTokens = sentence.split()
        #     for token in tmpTokens:
        #         if token not in topics:
        #             tokens.append(token)
        #
        #     if len(tokens) >= minTokenNum:
        #         tokenList.append(tokens)
        #         sentenceList.append(sentence)
        #
        # id=0
        # for sentence in sentenceList:
        #     newdict[id]=sentence
        #     id+=1
        #
        # id_print=[]
        #
        # for id,word in newdict.items():
        #     print(id,word)
        #
        # for id,word in newdict.items():
        #     if word in anomaloussentences:
        #         id_print.append(id)
        #
        # print(id_print)
        #
        # id=0
        # newdict2={}
        # for sentence in sent_tokenize(text):
        #     newdict2[id]=sentence
        #     id+=1
        #
        # print(len(newdict))
        #
        # for id,value in newdict2.items():
        #     print(id,value)



        # with open('sampleTest.txt','w') as f:
        #     f.write(text)
