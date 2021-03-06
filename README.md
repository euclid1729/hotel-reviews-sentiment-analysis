hotel-reviews-sentiment-analysis
================================
Topic sentiment change analysis : Sentiment flow analysis, Sentiment Labeling 

Algorithms/Keywords : Sentiment Analysis,Latent Dirichlet Allocation, Time Series Analysis, Bayesian Classification, Maximum Entropy Classifier, Time Series based Neural Network, Time Weighted Ranking, Bags of Words

Problem Statement:
The Trip Advisor data set consists of comments/reviews by tourists on different Las Vegas hotels. During the intial phase
we will use basic classification techniques like Naive Bayes and Maximum Entropy classifier to classify each user
comment into "positive", "negative" and "normal". The training data set contians user's comment, date, and rating. The 
rating is use to provide label to training data. 

In second phase we will try to model the sentiment flow as a time series. Since the reviews are time stamped, we will try
to make time window and analyse the sentiment flow in that time period e.g lets say we take average of ratings provided
in a day's interval as the sentiment of reviews on that day if out of 5 stars, the mean rating of a day < '3' then its "negative",
3<=rating <'3.5' is normal sentiment and rest is positive sentiment. We will try to train a Neural network providing 10
days consecutive mean sentiments and will try to predict the 11th day's average sentiment. 

Time Series Modelling:

1. The time window based recurrent neural network is provided input as sentiment rating of last 3 or last 5 time window's comments and LDA's output of the current comment in the form of two input nodes ( one representing positive sentiment and the other negative sentiment). This approach can be extended to any number of classes (e.g. to do classification of sentiments in 3 classes positive, negative, and normal, provide neural net with LDA's 3 output).

2. In this approach we tries to model Kinetic model of time series as a recurrent memory bases neural network. An extra set of neurons are added to the neural network that will keep historical time weighted ranking. The positive and negative ranking are decaying exponentially. 
