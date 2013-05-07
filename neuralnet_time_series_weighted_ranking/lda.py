from gensim import corpora, models, similarities
from gensim.models import hdpmodel, ldamodel
from itertools import izip
from collections import defaultdict
import codecs, os, glob, math

class lda(object):
 def generateCorpusAndDic(self,data):                                                            #   doc 1      doc2
  #this function aceeppts list of list, each sublist consisting words beloning to a document [['a','b'],['c','d']]
  dictionary = corpora.Dictionary(data)
  corpus = [dictionary.doc2bow(doc) for doc in data]
  self.dic=dictionary
  self.corpus=corpus
 
 def generateLDAModel(self,numTopic): # 3 in our case
  model = ldamodel.LdaModel(self.corpus, id2word=self.dic, num_topics=numTopic)
  self.model=model

 def getDocTopicProb(self):
  li=[]
  for i,j in enumerate(self.model[self.corpus]):
   li.append(j)
  return li
 
 
