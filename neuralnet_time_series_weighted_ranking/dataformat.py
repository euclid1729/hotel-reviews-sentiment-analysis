from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import WordPunctTokenizer
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.classify.util import accuracy
import numpy as np
import sys
import re
from matplotlib.mlab import PCA
import os

def process_review(review): 
 review=review.lower()
 #Convert www.* or https?://* to ''
 review = re.sub('((www\.[\S]+)|(https?://[^\S]+))','',review)
  #Convert @username to ''
 review = re.sub('@[^\S]+','',review)
  #Remove additional white spaces
 review = re.sub('[\s]+', ' ', review)
  #Replace #word with word
 review = re.sub(r'#([^\S]+)', r'\1', review)
 review = review.strip('\'"')
 review = re.sub('(([\@]\S*))','',review)
 review = re.sub('(([\?]))','',review)
 review = re.sub('(([\!]))','',review)
 review = re.sub('(([\:]))','',review)
 review = re.sub('(([\)|\(]))',' ',review)
 review=re.sub(r'([\d]+)','',review)
 review=re.sub(r'([\-])','',review)
 review=re.sub(r'([\<]\S*\>)','',review) 
 pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
 return pattern.sub(r"\1\1", review)

def extract_words(text):
 stemmer = PorterStemmer()
 tokenizer = WordPunctTokenizer()
 tokens = tokenizer.tokenize(text)
 bigram_finder = BigramCollocationFinder.from_words(tokens)
 bigrams = bigram_finder.nbest(BigramAssocMeasures.chi_sq, 500)
 for bigram_tuple in bigrams:
  x = "%s %s" % bigram_tuple
  tokens.append(x)
 result =  [stemmer.stem(x.lower()) for x in tokens if x not in stopwords.words('english') and len(x) > 1]
 return result 
    
def getfeaturevector(review):
 featureVector=[]
 words=extract_words(review)
 for w in words:
  w = w.strip('\'"?,.')
  val = re.search(r"^[a-zA-Z][a-zA-Z0-9]*$", w)
  if (val is None):
   continue
  else:
   featureVector.append(w.lower())
 return featureVector
 
def formatted_dataset(data):
 data=data[1:]
 reviews=[]
 for line in data:
  rating=int(line[2])
  if rating <3: sentiment=0 #'negative'
  elif rating >=3 and rating <=5: sentiment=1 #'normal'and 'postive'
#  elif rating >3.5 and rating <=5 : sentiment=2 #'positive'
  review=line[3]
  processedReview=process_review(review)
  fvector=getfeaturevector(processedReview)
  reviews.append((fvector,sentiment))
 return reviews
 
def trimmed_data(data): #extracting useful features 
 trim_data=[]
 for line in data:
  li=[]
  li.append(line[4])
  li.append(line[5])
  li.append(line[6])
  li.append(line[8])
  trim_data.append(li)
 
 f=file('trimmed_data.csv','w')
 for line in trim_data:
  f.write("%s\t"%line[0])
  f.write("%s\t"%line[1])
  f.write("%s\t"%line[2])
  f.write("%s\t"%line[3])
  f.write("\n")
 return trim_data
 
#lodaing data set 
def getTimeline(trim_data):
 
 data=[line.rstrip().split('\t') for line in file('trimmed_data.csv')]
 data=data[1:len(data)]
 dates=[row[0] for row in data]
 return dates
 #newDates=[]
 #for line in dates:
 # val=line.split('-')
 # str=val[0]+val[1]+val[2]
 # newDates.append(str)
 #newDates=[int(val) for val in newDates]
 #i=0
 #for i in range(len(data)-1):
 # data[i][0]=newDates[i]


def balance_dataset(reviews):
 li_pos=[]
 li_neg=[]
 li_nor=[]
 for i in range(len(reviews)-1):
  if reviews[i][1]=='positive':li_pos.append(i)
  if reviews[i][1]=='negative':li_neg.append(i)
  if reviews[i][1]=='normal':li_nor.append(i)
  
 balanced_dataset=[]
 for i in li_neg:
  balanced_dataset.append(reviews[i])
 for i in li_pos[:len(li_neg)]:
  balanced_dataset.append(reviews[i])
 for i in li_nor[:len(li_neg)]:
  balanced_dataset.append(reviews[i])
 return balanced_dataset

def get_feature(word):
    return dict([(word, True)])

def bag_of_words(words):
    return dict([(word, True) for word in words])
    
def globalFeatureVector(reviews):
 #this function will make a list of global features from the formatted_dataset
 dic_feature={}
 #reviews=formatted_dataset()
 #reviews=balance_dataset(reviews)
 temp=[li for line in reviews for li in line[0]]
 for key in temp: dic_feature[key]='TRUE'
 return dic_feature.keys()
 
def transformToGFV(reviewFeatureVector,gfeaturelist):
 unique_words=set(reviewFeatureVector)
 g_reviewfeaturedic=[]
 val=0
 for word in gfeaturelist:
  if word in unique_words: val=1
  else: val=0
  g_reviewfeaturedic.append(val)
 return g_reviewfeaturedic

 
if __name__ == '__main__':
 # getting all files in current directory 
 
 data=[line.rstrip().split('\t') for infile in os.listdir('./data') for line in file('./data/'+infile)]
 data=trimmed_data(data)
 data=formatted_dataset(data)
 #using tf-idf to reduce dimensionality 
 
 gfeaturelist=globalFeatureVector(data)
 print "length of global dic vector"
 print len(gfeaturelist)
 gv=[]
 for line in data:
  gv.append(transformToGFV(line[0],gfeaturelist))
 
 ar=np.array(gv)
 results = PCA(ar) 
 print results.fracs
 
 #f_data=balance_dataset(data)
