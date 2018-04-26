

import os
import urllib
import re
from bs4 import BeautifulSoup, SoupStrainer

targetAtlassianAddr = 'https://www.atlassian.com/legal/'
targetGithubAddr = 'https://help.github.com/categories/site-policy/'

targetAtlassianDirHTML = 'atlassianLegalDocDir'
targetGithubDirHTML = 'githubLegalDocDir'

targetAtlassianDirParsed = 'atlassianLegalDocDirParsed'
targetGithubDirParsed = 'githubLegalDirParsed'

re_subsSecNum = re.compile('.*([0-9][0-9]\.[0-9]+)', re.UNICODE)
re_secNum = re.compile(r'\d+', re.UNICODE)

def extractTextInfo(targetDir, htmlDocName=None, htmlObject=None, agreementSource='atlassian'):
    def getMeaningfulDivLists(divSp):
        divList = divSp.findAll('div')
        if len(divList) == 0: return []
        maxLen = max(len(divBlk) for divBlk in divList)
        return [divBlk for divBlk in divList if len(divBlk) >= 0.2 * maxLen]

    def extractInfo(sp):
        divLists = getMeaningfulDivLists(sp)
        emphResult = list()
        h1Result = list()
        h2Result = list()
        textResult = list()
        for div in divLists:

            def getTagList():
                tagNames = ['strong', 'h1', 'h2', 'p']
                result = list()
                for name in tagNames:
                    result.append([info.text for info in div.findAll(name)])
                return result
            emphList, h1List, h2List, textList = getTagList()
            emphResult.append(emphList)
            h1Result.append(h1List)
            h2Result.append(h2List)
            textResult.append(textList)
        return emphResult, h1Result, h2Result, textResult

    if htmlDocName:
        with open(htmlDocName, 'r') as f:
            sp = BeautifulSoup(f, 'html.parser')
    else:
        sp = BeautifulSoup(htmlObject, 'html.parser')
    if agreementSource == 'atlassian' or 'github':

        return extractInfo(sp)


    else:
        raise Exception('unknown source!')


def getTrainingText(targetDirHTML, targetAddr, sourceType='atlassian'):
    if not os.path.exists(targetDirHTML):
        os.makedirs(targetDirHTML)
    hname = urllib.request.urlopen(targetAddr).read()
    if sourceType == 'atlassian':
        with open(targetAtlassianDirHTML + '/Eula.html', 'wb') as fname:
            fname.write(hname)
        for lname in BeautifulSoup(hname, 'html.parser', parse_only=SoupStrainer('a')):
            if lname.has_attr('href'):
                if '/legal/' in lname['href']:
                    loadAddrName = targetAddr + lname['href'][len('/legal/'):]
                    loadName = targetDirHTML + '/' + loadAddrName.split('/')[-1] + '.html'
                    with open(loadName, 'wb') as fname:
                        tmpHtmlname = urllib.request.urlopen(loadAddrName).read()
                        fname.write(tmpHtmlname)
    elif sourceType == 'github':
        with open(targetGithubDirHTML + '/Eula.html', 'wb') as fname:
            fname.write(hname)
            for lname in BeautifulSoup(hname, 'html.parser', parse_only=SoupStrainer('a')):
                if lname.has_attr('href'):
                    if '/articles/' in lname['href']:
                        loadAddrName = 'https://help.github.com/articles/' + lname['href'][len('/articles/'):]
                        print(loadAddrName)
                        loadName = targetDirHTML + '/' + loadAddrName.split('/')[-1] + '.html'
                        with open(loadName, 'wb') as fname:
                            tmpHtmlname = urllib.request.urlopen(loadAddrName).read()
                            fname.write(tmpHtmlname)
    else:
        raise Exception('unknown source!')



if __name__ == '__main__':
    getTrainingText(targetAtlassianDirHTML, targetAtlassianAddr)
    emph, h1, h2, text = extractTextInfo(targetAtlassianDirParsed, htmlDocName=targetAtlassianDirHTML + '/Eula.html')

    getTrainingText(targetGithubDirHTML, targetGithubAddr, sourceType='github')
    emph, h1, h2, text = extractTextInfo(targetAtlassianDirParsed, agreementSource='github', htmlDocName=targetGithubDirHTML + '/github-registered-developer-agreement.html')