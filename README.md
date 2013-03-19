hotel-reviews-sentiment-analysis
================================
Topic sentiment change analysis : Sentiment flow analysis, Sentiment Labeling 

Algorithms/Keywords : Sentiment Analysis,Latent Dirichlet Allocation, Hidden Markov Models, Time Series Analysis, Conditional Random Fields, Bayesian Classification, Maximum Entropy Classifier, Time Series based Neural Network, SVM

Problem Statement:
The Trip Advisor data set consists of comments/reviews by tourists on different Las Vegas hotels. During the intial phase
we will use basic classification techniques like Naive Bayes, SVM and Maximum Entropy classifier to classify each user
comment into "positive", "negative" and "normal". The training data set contians user's comment, date, and rating. The 
rating is use to provide label to training data. 

In second phase we will try to model the sentiment flow as a time series. Since the reviews are time stamped, we will try
to make a time window and analyse the sentiment flow in that time period e.g lets say we take average of ratings provided
in a day's interval as the sentiment of reviews on that day if out of 5 stars, the mean rating of a day <3 then its "negative",
3<=rating <3.5 is normal sentiment and rest is positive sentiment. We will try to train a Neural network providing 10
days consecutive mean sentiments and will try to predict the 11th day's average sentiment. We believe that sentiments about
a hotel will only change over time when the hotel has taken some drastic steps e.g. to improve its service, food quality etc.
So over some perticular period the time series of sentiments can be stationary and we will try to exploit this feature via
Neural network based prediction. The prediced value can be used for better labelling of sentiments of future reviews by 
using ensemble learning technique. 


The project can be broken down into following phases :

1. Sentiment labelling using Naive Bayes, Maximum Entropy Classifier, SVM
2. Time line formation, Sentiment window formation, Training Neural Network to predict sentiment flow, Sentiment flow time series analysis
3. Performance comparision, Time series sentiment flow prediction vs standard classification techniques
4. Making ensemble sentiment labeling technique 
