import unicodedata
import codecs
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

########################

#Converts text to ascii from unicode.
def load(text):
    #open the text file
    #converting the file from unicode to ascii
    with codecs.open(str(text),
                 "r",encoding='utf-8', errors='replace') as speech:
        raw=speech.read()
        raw_asc = unicodedata.normalize('NFKD', raw).encode('ascii','ignore')
    return raw_asc

#all speeches are in list. 
speech = []

for i in range(1,59):
    sp = 'InAd/in' + str(i) + '.txt'
    inaug = load(str(sp))
    speech.append(inaug)

print 'Loaded all speeches'

#converts all speeches into tfidf matrix
tfidf = TfidfVectorizer(stop_words='english').fit_transform(speech)

print 'Converted to TF-IDF Matrix.'


#create a cosine similarity matrix for each text file
from sklearn.metrics.pairwise import cosine_similarity
cos = cosine_similarity(tfidf, tfidf)
print cos
print 'Cosine Similarity Matrix Complete'

#show a heat map of how similar speeches are to each other
plt.title('Cosine Similarity of Inaugural Addresses')
plt.xlabel('Inaugural Address')
plt.ylabel('Inaugural Address')
heatmap = plt.pcolor(cos,cmap=plt.cm.Blues)
plt.colorbar(heatmap)
plt.savefig('CosSimInAd')
plt.show()
