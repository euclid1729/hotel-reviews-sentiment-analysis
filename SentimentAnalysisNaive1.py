data=[line.rstrip().split('\t') for line in file('treasureisland.csv')]
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
 f.write("\n"])
 
#lodaing data set 
data=[line.rstrip().split('\t') for line in file('trimmed_data.csv')]
data=data[1:len(data)]
dates=[row[0] for row in data]
newDates=[]
for line in dates:
 val=line.split('-')
 str=val[0]+val[1]+val[2]
 newDates.append(str)

newDates=[int(val) for val in newDates]

i=0
for i in range(len(data)-1):
 data[i][0]=newDates[i]
 
 
import re

#this function will filter review sentences to remove special characters and will format the string 
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
 

def processed_data(data):
 pdata=[]
 for row in data:
  new_row=[]
  processedReview = process_review(row[3])
  new_row=row[:3]
  new_row.append(processedReview)
  pdata.append(new_row)
 return pdata

stopwords=[word.strip() for word in file('stopwords.txt')]

def getfeaturevectorwithoutstopwords(review):
 featureVector=[]
 words=review.split()
 for w in words:
  w = w.strip('\'"?,.')
  val = re.search(r"^[a-zA-Z][a-zA-Z0-9]*$", w)
  if (val is None):
   continue
  else:
   featureVector.append(w.lower())
 return featureVector
 
def getfeaturevector(review):
 featureVector=[]
 words=review.split()
 for w in words:
  w = w.strip('\'"?,.')
  val = re.search(r"^[a-zA-Z][a-zA-Z0-9]*$", w)
  if (w in stopwords or val is None):
   continue
  else:
   featureVector.append(w.lower())
 return featureVector

#forming new data set with two attributes, feature vector of review, sentiment 
#for training purposes we will say negative sentiment if rating <2, positive sentiment if rating >=3, normal if 2<rating<3 

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

def globalFeatureVector():
 #this function will make a list of global features from the formatted_dataset
 dic_feature={}
 reviews=formatted_dataset()
 reviews=balance_dataset(reviews)
 temp=[li for line in reviews for li in line[0]]
 for key in temp: dic_feature[key]='TRUE'
 return dic_feature.keys()

gfeaturelist=globalFeatureVector()

def transformToGFV(reviewFeatureVector):
 unique_words=set(reviewFeatureVector)
 g_reviewfeaturedic={}
 #for word in gfeaturelist:
 # g_reviewfeaturedic['contains(%s)' %word]= (word in unique_words)
 #return g_reviewfeaturedic
 for word in unique_words:
   g_reviewfeaturedic['contains(%s)' %word]='TRUE'
 return  g_reviewfeaturedic
 
 
#converting all reviews local feature vector to dictionary vector
import nltk
data_set = nltk.classify.util.apply_features(transformToGFV, reviews)
#training_set=data_set[: 1200]
#doing bayesian classification 
NBClassifier = nltk.NaiveBayesClassifier.train(data_set)

testReview="loved the hotel. Service was pretty good. Casinos were awesome. Will surely visit again. A nice hotel. The view was great."
testreview=process_review(testReview)
print NBClassifier.classify(transformToGFV(getfeaturevector(testreview)))

##############
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
 
training_set= nltk.classify.util.apply_features(transformToGFV, balanced_dataset)
NBClassifier = nltk.NaiveBayesClassifier.train(training_set)

 ##########maximum entropy classifier 
MaxEntClassifier = nltk.classify.maxent.MaxentClassifier.train(training_set, 'GIS', trace=3, encoding=None, labels=None, sparse=True, gaussian_prior_sigma=0, max_iter = 10)




 
 


