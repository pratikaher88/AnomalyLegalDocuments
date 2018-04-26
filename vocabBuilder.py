import logging
import htmlParser
# import txtParser
import os.path
import sys
import gensim
import re
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
import nltk
from gensim import corpora, models

tokenizer_old = nltk.data.load('tokenizers/punkt/english.pickle')
tokenizer = RegexpTokenizer(r'\w+')

# def updateModel(model, fileNameList):
#     for trainingTxt in fileNameList:
#         newSentence = gensim.models.word2vec.LineSentence(trainingTxt)
#         model.build_vocab(newSentence, update=True)
#         model.train(newSentence)

extraStopWords = ['bitbucket', 'atlassian', 'github']

en_stop = get_stop_words('en')

p_stemmer = PorterStemmer()

def cleanupDoc(fileString):
    raw = fileString.lower()
    tokens = tokenizer.tokenize(raw)

    stoppedTokens = [i for i in tokens if i not in en_stop]
    # stemmedTokens = [p_stemmer.stem(i) for i in stoppedTokens]
    return ' '.join(stoppedTokens)


def generateVocabulary(fname):
    f = open(fname, 'w+')
    f.write(' ')
    corporaFiles = os.listdir('./corpora')
    for filename in corporaFiles:
        with open('./corpora/'+filename, 'rb') as infile:
            infileContent = infile.read().decode('utf-8')
            f.write(cleanupDoc(infileContent))
    f.close()


def initVocabModel():
    generateVocabulary('./corpora/vocabulary.txt')
    model = gensim.models.Word2Vec()
    trainSentences = gensim.models.word2vec.LineSentence('./corpora/vocabulary.txt')
    model.build_vocab(trainSentences)
    model.train(trainSentences,total_examples=trainSentences.max_sentence_length, epochs=model.iter)
    model.save('word2vecModel')

r = re.compile('[A-Za-z]+')
deleteSet = set(['a', 'an', 'and', 'or', 'the'])

def tokenize(ct):
    resultSentences = list()
    sentences = tokenizer_old.tokenize(ct)
    for sentence in sentences:
        tokens = r.findall(sentence.lower())
        resultSentences.append(" ".join([token for token in tokens if token not in deleteSet]))
    return resultSentences

def file2CorpusTxt(txtPrefix):
    fileDirs = [t for t in os.listdir('.') if txtPrefix in t and 'Parsed' not in t]
    for fileDir in fileDirs:
        if 'atlassian' in fileDir or 'github' in fileDir :
            for fname in os.listdir('./'+fileDir):
                print(fname)
                if 'html' not in fname:
                    continue
                textList = htmlParser.extractTextInfo(fileDir, './'+fileDir+'/'+fname)[3]
                print(">>>> translated file name = ", './corpora/vocab-'+txtPrefix+"-"+fname[:-5] + '.txt')
                with open('./corpora/vocab-'+txtPrefix+"-"+fname[:-5] + '.txt', 'w+') as fVocab:
                    for div in textList:
                        for content in div:
                            sentences = tokenize(content)
                            for sentence in sentences:
                                fVocab.write(sentence + '\n')
        elif 'apple' in fileDir:
            for fname in os.listdir('./'+fileDir):
                with open('./'+fileDir+'/'+fname) as f:
                    with open('./corpora/vocab-'+txtPrefix+'-'+fname[:-4], 'w+') as fVocab:
                        fVocab.write(f.read())


def loadModel():
    return gensim.models.Word2Vec.load('word2vecModel')

if __name__ == '__main__':
    # file2CorpusTxt('atlassian')
    # file2CorpusTxt('github')
    file2CorpusTxt('apple')
    initVocabModel()
