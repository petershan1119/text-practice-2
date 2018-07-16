import lucem_illud

import requests
import nltk
import pandas
import matplotlib.pyplot as plt
import wordcloud
import numpy as np
import scipy
import seaborn as sns
import sklearn.manifold
from nltk.corpus import stopwords
import json
import urllib.parse


def wordCounter(wordLst):
    wordCounts = {}
    for word in wordLst:
        wLower = word.lower()
        if wLower in wordCounts:
            wordCounts[wLower] += 1
        else:
            wordCounts[wLower] = 1

    countsForFrame = {'word': [], 'count': []}
    for w, c in wordCounts.items():
        countsForFrame['word'].append(w)
        countsForFrame['count'].append(c)
    return pandas.DataFrame(countsForFrame)

countedWords = wordCounter(nltk.corpus.gutenberg.words('shakespeare-macbeth.txt'))


words = [word.lower() for word in nltk.corpus.gutenberg.words('shakespeare-macbeth.txt')]
freq = nltk.FreqDist(words)

countedWords.sort_values('count', ascending=False, inplace=True)

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(range(len(countedWords)), countedWords['count'])
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(range(len(countedWords)), countedWords['count'])
ax.set_yscale('log')
ax.set_xscale('log')
plt.show()


macbethindex = nltk.text.ConcordanceIndex(nltk.corpus.gutenberg.words('shakespeare-macbeth.txt'))

print(countedWords[countedWords['word'] == 'donalbaine'])
macbethindex.print_concordance('Donalbaine')


r = requests.get('https://api.github.com/repos/lintool/GrimmerSenatePressReleases')
senateReleaseData = json.loads(r.text)
print(senateReleaseData.keys())
print(senateReleaseData['description'])
print(senateReleaseData['contents_url'])

r = requests.get('https://api.github.com/repos/lintool/GrimmerSenatePressReleases/contents/raw/Whitehouse')

whitehouseLinks = json.loads(r.text)
whitehouseLinks[0]

r = requests.get(whitehouseLinks[0]['download_url'])
whitehouseRelease = r.text
print(whitehouseRelease[:1000])

whTokens = nltk.word_tokenize(whitehouseRelease)

whText = nltk.Text(whTokens)
whitehouseIndex = nltk.text.ConcordanceIndex(whText)
whitehouseIndex.print_concordance('Whitehouse')

whText.collocations()


def getGithubFiles(target, maxFiles=100):
    # We are setting a max so our examples don't take too long to run
    # For converting to a DataFrame
    releaseDict = {
        'name': [],
        'text': [],
        'path': [],
        'html_url': [],
        'download_url': [],
    }

    # Get the directory information from Github
    r = requests.get(target)
    filesLst = json.loads(r.text)

    for fileDict in filesLst[:maxFiles]:
        # These are provided by the directory
        releaseDict['name'].append(fileDict['name'])
        releaseDict['path'].append(fileDict['path'])
        releaseDict['html_url'].append(fileDict['html_url'])
        releaseDict['download_url'].append(fileDict['download_url'])

        # We need to download the text
        text = requests.get(fileDict['download_url']).text
        releaseDict['text'].append(text)
    return pandas.DataFrame(releaseDict)

