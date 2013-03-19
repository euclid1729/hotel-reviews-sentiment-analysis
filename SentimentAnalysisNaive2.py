from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import WordPunctTokenizer
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy
import sys
import re

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
 
def formatted_dataset():
 data=[line.rstrip().split('\t') for line in file('trimmed_data.csv')]
 data=data[1:]
 reviews=[]
 for line in data:
  rating=int(line[2])
  if rating <3: sentiment='negative'
  elif rating >=3 and rating <=3.5: sentiment='normal'
  elif rating >3.5 and rating <=5 : sentiment='positive'
  review=line[3]
  processedReview=process_review(review)
  fvector=getfeaturevector(processedReview)
  reviews.append((fvector,sentiment))
 return reviews

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

if __name__ == '__main__':
 data=formatted_dataset() 
 f_data=balance_dataset(data)
 train_set = []
 for features,sentiment in f_data:
  train_set = train_set + [(get_feature(word), sentiment) for word in features]

 classifier = NaiveBayesClassifier.train(train_set)
 #testing classifier
 line="toilets were clean. service was good price was afordable overall nice experience"
 tokens = bag_of_words(extract_words(line))
 decision = classifier.classify(tokens)
