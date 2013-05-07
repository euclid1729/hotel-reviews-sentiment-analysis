from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import WordPunctTokenizer
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy
import sys
import re
import nltk

def trim_dataset(filename):
 data=[line.rstrip().split('\t') for line in file(filename)]
 trim_data=[]
 for line in data:
  li=[]
  li.append(line[4])
  li.append(line[5])
  li.append(line[6])
  li.append(line[8])
  trim_data.append(li)
 
 #f=file('trimmed_data.csv','w')
 return trim_data

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
 #data=[line.rstrip().split('\t') for line in file('trimmed_data.csv')]
 data=data[1:]
 reviews=[]
 for line in data:
  rating=int(line[2])
  if rating <3: sentiment='negative'
  #elif rating >=3 and rating <=3.5: sentiment='normal'
  elif rating >=3 and rating <=5 : sentiment='positive'
  review=line[3]
  processedReview=process_review(review)
  fvector=getfeaturevector(processedReview)
  reviews.append((fvector,sentiment))
 return reviews

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

def transformToGFV(reviewFeatureVector):
 unique_words=set(reviewFeatureVector)
 g_reviewfeaturedic={}
 for word in gfeaturelist:
  g_reviewfeaturedic['contains(%s)' %word]= (word in unique_words)
 return g_reviewfeaturedic

print "enter file name"
filename=raw_input()
trim_data=trim_dataset(filename)
print "data set loaded"
reviews=formatted_dataset(trim_data)
print "data formatted"
gfeaturelist=globalFeatureVector(reviews)
train_size=1200
train_set=reviews[:train_size]
test_data=reviews[train_size:]
training_set = nltk.classify.apply_features(transformToGFV, train_set)

NBClassifier = nltk.NaiveBayesClassifier.train(training_set)

print "classifiers trained"
count_nb=0

for line in test_data:
  sentiment=line[1]
  review=line[0]
  gvector_review = transformToGFV(review)
  decision1 = NBClassifier.classify( gvector_review)

  if decision1==sentiment:
   count_nb+=1
  
accuracy_nb=float(count_nb)/len(test_data)
print "accuracy Naive Bayes= ",accuracy_nb
