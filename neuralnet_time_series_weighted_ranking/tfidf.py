import math
from operator import itemgetter

def freq(word, review): #here we will be providing feature vector of the review, in our case feature vector is a list 
 return review.count(word)

def wordCount(review):
 return len(review)

def numDocsContaining(word,reviewsList):
 count = 0
 for review in reviewsList:
  if freq(word,review) > 0:
   count += 1
 return count

def tf(word, review):
 return (freq(word,review) / float(wordCount(review)))

def idf(word, reviewsList):
  return math.log(len(reviewsList) / numDocsContaining(word,reviewsList))

def tfidf(word, review, reviewList):
  return (tf(word,review) * idf(word,reviewList))

def getWords(documentList):
 #documentList = []
 words = {}
 documentNumber = 0
 document_tfidf={}
 for review in documentList:
  document_tfidf[documentNumber]={}
  for word in review:
   val=tfidf(word,review,documentList)
   document_tfidf[documentNumber][word]=val
   if word not in words:
    words[word] = val
   else:
    if val > words[word]:
     words[word]= val
  documentNumber+=1   
 #for item in sorted(words.items(), key=itemgetter(1), reverse=True):
 # print "%f <= %s" % (item[1], item[0])
 return document_tfidf,sorted(words.items(),key=itemgetter(1), reverse=True)
