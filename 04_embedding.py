import os
import sklearn

import gensim
import lucem_illud
import nltk
import pandas

kennedyDF = lucem_illud.loadTextDirectory('./data/grimmerPressReleases/kennedy/')

kennedyDF['category'] = 'Kennedy'
kennedyDF[:5]

dataDir = '/Users/sangwonhan/practices/python/text_practice_2/data/grimmerPressReleases/'

senReleasesDF = pandas.DataFrame()

def loadTextDirectory(targetDir, encoding = 'utf-8'):
    kbs = []
    text = []
    fileName = []

    for file in (file for file in os.scandir(targetDir) if file.is_file() and not file.name.startswith('.')):
        try:
            with open(file.path, encoding = encoding) as f:
                text.append(f.read())
            fileName.append(file.name)
        except UnicodeDecodeError:
            with open(file.path, encoding = 'latin-1') as f:
                text.append(f.read())
            fileName.append(file.name)
            kbs.append(file.path)
    print(kbs)
    return pandas.DataFrame({'text' : text}, index = fileName)


for senatorDir in (file for file in os.scandir(dataDir) if not file.name.startswith('.') and file.is_dir()):
    senDF = loadTextDirectory(senatorDir.path)
    senDF['category'] = senatorDir.name
    senReleasesDF = senReleasesDF.append(senDF, ignore_index=False)


# Apply our functions, notice each row is a list of lists now
senReleasesDF['tokenized_sents'] = senReleasesDF['text'].apply(lambda x: [nltk.word_tokenize(s) for s in nltk.sent_tokenize(x)])


stop_words_basic = nltk.corpus.stopwords.words('english')

def normalizeTokens(tokenLst, stopwordLst = None, stemmer = None, lemmer = None):
    # We can use a generator here as we just need to iterate over it

    # Lowering the case and removing non-words
    workingIter = (w.lower() for w in tokenLst if w.isalpha())

    # Now we can use the stemmer, if provided
    if stemmer is not None:
        workingIter = (stemmer.stem(w) for w in workingIter)

    # And the lemmer
    if lemmer is not None:
        workingIter = (lemmer.lemmatize(w) for w in workingIter)

    # And remove the stopwords
    if stopwordLst is not None:
        workingIter = (w for w in workingIter if w not in stopwordLst)


    print("iter: {}".format(tokenLst))
    return list(workingIter)

senReleasesDF['normalized_sents'] = \
    senReleasesDF['tokenized_sents'].apply(
        lambda x: [normalizeTokens(s, stopwordLst=stop_words_basic, stemmer=None) for s in x])

senReleasesDF.columns.values

senReleasesW2V = gensim.models.word2vec.Word2Vec(senReleasesDF['normalized_sents'].sum())

senReleasesW2V.most_similar('president')
senReleasesW2V.most_similar('war')

def cos_difference(embedding, word1, word2):
    return sklearn.metrics.pairwise.cosine_similarity(embedding[word1].reshape(1, -1), embedding[word2].reshape(1, -1))

cos_difference(senReleasesW2V, 'war', 'chaos')

senReleasesW2V.doesnt_match(['administration', 'administrations', 'presidents', 'president', 'washington'])
senReleasesW2V.most_similar(positive=['clinton', 'republican'], negative=['democrat'])



# Doc2Vec
apsDF = pandas.read_csv('/Users/sangwonhan/practices/python/text_practice_2/data/APSabstracts1950s.csv', index_col=0)
keywords = ['photomagnetoelectric', 'quantum', 'boltzman', 'proton', 'positron', 'feynman', 'classical', 'relativity']

apsDF['tokenized_words'] = apsDF['abstract'].apply(lambda x: nltk.word_tokenize(x))
apsDF['normalized_words'] = apsDF['tokenized_words'].apply(lambda x: normalizeTokens(x, stopwordLst=stop_words_basic, stemmer = None))


taggedDocs  = []
for index, row in apsDF.iterrows():
    #Just doing a simple keyword assignment
    docKeywords = [s for s in keywords if s in row['normalized_words']]
    docKeywords.append(row['copyrightYear'])
    docKeywords.append(row['doi']) #This lets us extract individual documnets since doi's are unique
    taggedDocs.append(gensim.models.doc2vec.LabeledSentence(words = row['normalized_words'], tags = docKeywords))
apsDF['TaggedAbstracts'] = taggedDocs

apsD2V = gensim.models.doc2vec.Doc2Vec(apsDF['TaggedAbstracts'], size=100)

