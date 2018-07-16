import lucem_illud

import pandas
import numpy

import sklearn

newsgroups = sklearn.datasets.fetch_20newsgroups(subset='train', data_home='/Users/sangwonhan/practices/python/text_practice_2/data')
print(dir(newsgroups))

print(newsgroups.target_names)
print(len(newsgroups.data))

newsgroupsCategories = ['comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos']

newsgroupsDF = pandas.DataFrame(columns=['text', 'category', 'source_file'])

for category in newsgroupsCategories:
    print("Fetching data for: {}".format(category))
    ng = sklearn.datasets.fetch_20newsgroups(subset='train', categories=[category], remove=['headers', 'footers', 'quotes'], data_home='/Users/sangwonhan/practices/python/text_practice_2/data/')

    newsgroupsDF = newsgroupsDF.append(pandas.DataFrame({'text': ng.data, 'category': [category] * len(ng.data), 'source_file': ng.filenames}), ignore_index=True)


# First it needs to be initialized
ngCountVectorizer = sklearn.feature_extraction.text.CountVectorizer()
# Then trained
newsgroupsVects = ngCountVectorizer.fit_transform(newsgroupsDF['text'])

newsgroupsVects[:10, :20].toarray()

vector_vec = ngCountVectorizer.vocabulary_.get('vector')

ng_vect = newsgroupsVects[:,vector_vec].toarray()

numpy.where(ng_vect >= 1)

newsgroupsTFTransformer = sklearn.feature_extraction.text.TfidfTransformer().fit(newsgroupsVects)
newsgroupsTF = newsgroupsTFTransformer.transform(newsgroupsVects)

list(zip(ngCountVectorizer.vocabulary_.keys(), newsgroupsTF.data))[:20]

ngTFVectorizer = sklearn.feature_extraction.text.TfidfVectorizer(max_df=0.5, max_features=1000, min_df=3, stop_words='english', norm='l2')
newsgroupsTFVects = ngTFVectorizer.fit_transform(newsgroupsDF['text'])


print(ngTFVectorizer.vocabulary_['apple'])