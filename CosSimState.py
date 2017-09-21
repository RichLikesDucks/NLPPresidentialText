import unicodedata
import codecs
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

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

for i in range(1789,2018):
    sp = 'StateU/state' + str(i) + '.txt'
    inaug = load(str(sp))
    speech.append(inaug)

    
print 'Loaded all speeches'

#converts all speeches into tfidf matrix
tfidf = TfidfVectorizer(speech,stop_words='english').fit_transform(speech)

print 'Converted to TF-IDF Matrix.'


#create a cosine similarity matrix for each text file
from sklearn.metrics.pairwise import cosine_similarity
cos = cosine_similarity(tfidf, tfidf)
print cos
print 'Cosine Similarity Matrix Complete'

#for i in range(len(speech)):
#    print('State Speech %d: year %d' %(i, i+1789))


#show a heat map of how similar speeches are to each other
plt.title('Cosine Similarity of State of The Union Addresses by year')
#plt.xlabel('State of the Union Address')
#plt.ylabel('State of the Union Address')
#plt.xticks(range(0,228,75),range(1789,2017,75))
#plt.yticks(range(0,228,75),range(1789,2017,75))
cmap = plt.pcolor(cos, cmap=plt.cm.Blues)

#Calvin Coolidge
#plt.axhline(134, color = 'black')
#plt.axvline(134, color = 'black')

#Calvin Coolidge
#plt.axhline(141, color = 'black')
#plt.axvline(141, color = 'black')
#plt.text(141,-10,'Calvin',rotation=30)


#Start of WWI
#plt.axhline(125, color = 'black')
#plt.axvline(125, color = 'black')
#plt.text(125,-10,'WWI',rotation=30)

#Great Depression
#plt.axvline(140, color = 'black')
#plt.axhline(140, color = 'black')
#plt.text(140,-10,'Great Depression',rotation=40)

#End of WWII
#plt.axhline(156, color = 'black')
#plt.axvline(156, color = 'black')
#plt.text(156,-10,'WWII End',rotation=30)

#First speech after 9/11
#plt.axvline(213, color = 'black')
#plt.text(213,-10,'9/11',rotation=30)

#plt.axhline(213, color = 'black')

#Obama 
#plt.axvline(220, color = 'black')
#plt.axhline(220, color = 'black')
#plt.text(220,-10,'Obama',rotation=30)



ax1 = plt.colorbar(cmap)
#plt.savefig('CosSimStateWar')
plt.show()



