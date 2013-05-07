from numpy import *
import os
import dataformat
import lda
import re
class FeedForwardNetwork:
     
    def __init__(self, nIn, nHidden, nOut):
        # learning rate
        self.alpha = 0.1
                                                 
        # number of neurons in each layer
        self.nIn = nIn
        self.nHidden = nHidden
        self.nOut = nOut
          
        #rank update parameters 
        self.sum_day_rank=zeros(2).reshape(2,1)
        self.day_count=0  
        # initialize weights randomly (+1 for bias)
        self.hWeights = random.random((self.nHidden, self.nIn+1))
        self.oWeights = random.random((self.nOut, self.nHidden+1))
        #initializing 2 weights from memory neurons ( 1 from positive ranking, 1 from negative ranking)
        self.pWeight=random.random(1) #positive weight
        self.nWeight=random.random(1) #negative weight
        self.mWeights=array([self.pWeight,self.nWeight]).reshape(2,1)
        # activations of neurons (sum of inputs)
        self.hActivation = zeros((self.nHidden, 1), dtype=float)
        self.oActivation = zeros((self.nOut, 1), dtype=float)
        
        #initialize activation (i.e. weighted ranks ) for the memory neurons, initially we have no postive,negative or normal rank
        self.pActivation=zeros(1)
        self.nActivation=zeros(1)
        self.mActivation=array([self.pActivation,self.nActivation]).reshape(2,1) 
        # outputs of neurons (after sigmoid function)
        self.iOutput = zeros((self.nIn+1, 1), dtype=float)      # +1 for bias
        self.hOutput = zeros((self.nHidden+1, 1), dtype=float)  # +1 for bias
        self.oOutput = zeros((self.nOut), dtype=float)
        self.mOutput = zeros(2, dtype=float)
        # deltas for hidden and output layer
        self.hDelta = zeros((self.nHidden), dtype=float)
        self.oDelta = zeros((self.nOut), dtype=float)  
        self.mDelta=zeros(2)        
     
    def forward(self, input):
        # set input as output of first layer (bias neuron = 1.0)
        self.iOutput[:-1, 0] = input
        self.iOutput[-1:, 0] = 1.0
         
        # hidden layer
        self.hActivation = dot(self.hWeights, self.iOutput)
        self.hOutput[:-1, :] = tanh(self.hActivation)
         
        # set bias neuron in hidden layer to 1.0
        self.hOutput[-1:, :] = 1.0
         
        # output layer
        self.oActivation = dot(self.oWeights, self.hOutput)
        #adding contribution from memory (rank) layer 
        self.mOutput=tanh(self.mActivation)
        self.mContribute=self.mWeights * self.mOutput
        self.oOutput = tanh(self.oActivation + self.mContribute)
     
    def backward(self, teach):
        error = self.oOutput - array(teach, dtype=float)
        # deltas of output neurons
        self.oDelta = (1 - tanh(self.oActivation)) * tanh(self.oActivation) * error
                 
        # deltas of hidden neurons
        self.hDelta = (1 - tanh(self.hActivation)) * tanh(self.hActivation) * dot(self.oWeights[:,:-1].transpose(), self.oDelta)
        
        self.mDelta= (1 - tanh(self.mActivation)) * tanh(self.mActivation) * (self.mWeights * self.oDelta)        
        # apply weight changes
        self.hWeights = self.hWeights - self.alpha * dot(self.hDelta, self.iOutput.transpose())
        self.oWeights = self.oWeights - self.alpha * dot(self.oDelta, self.hOutput.transpose())
        self.mWeights = self.mWeights - self.alpha * (self.mDelta * self.mOutput)
    def getOutput(self):
        return self.oOutput
     
    def updateRank(self,teach,flag):
        if flag=="no":
         #ranking belongs to same day
         self.sum_day_rank+=teach
         self.day_count+=1
        elif flag=="first":
         self.sum_day_rank=teach
         self.day_count=1
        else: 
         self.mActivation+=(self.sum_day_rank / self.day_count)
         self.mActivation*=.5
         self.sum_day_rank=teach
         self.day_count=1
        
if __name__ == '__main__': 
    # define training set
    data=[line.rstrip().split('\t') for infile in os.listdir('./data') for line in file('./data/'+infile)]
    tdata=dataformat.trimmed_data(data)
    timeline=dataformat.getTimeline(tdata)
    fdata=dataformat.formatted_dataset(tdata)
    print "\n data formatted"
    neurons=2
    # create network
    ffn = FeedForwardNetwork(neurons, 3, 2) #using one per class coding 
    #using lda 

    d=[]
    for line in fdata:
     d.append(line[0])
    obj=lda.lda()
    obj.generateCorpusAndDic(d)
    obj.generateLDAModel(2)
    inputs=obj.getDocTopicProb()
    
    print "\n lda done"
    size=int (len(inputs) * .8)
    train_inputs=inputs[:size]
    train_outputs=[]
    counter=0
    while(counter<size):
     train_outputs.append(fdata[counter][1])
     counter+=1
    
    test_inputs=inputs[size:]
    test_outputs=[]
    counter=size
    while(counter<len(fdata)):
     test_outputs.append(fdata[counter][1])
     counter+=1
    print "\n train test datasets prepared"
    count = 0
    print "\n starting neural net training"
 
    while(count <1000): #epochs 
      i=0
      epoch_accuracy=0      
      for i in range(len(train_inputs)):
       #if i+5 < len(sentiments):  
        # forward and backward pass
        feed_input=[0,0]
        li=train_inputs[i]
        # li is of form [(0, 0.55527820230757163), (1, 0.43314313521528525)] but can vary in number of elements 
        for j in range(len(li)):
         feed_input[li[j][0]]=li[j][1]
        ffn.forward(feed_input)
        #now in feedbackwards we will have to feed a 3 valued vector 
        expected_output=[[0],[0]]
        expected_output[int(train_outputs[i])][0]=1
        ffn.backward(array(expected_output,dtype=int))
        #if time stamp is same dont decay or update rank
        flag="yes"
        if i!=0:
         if timeline[i-1]==timeline[i]:
          flag="no"
         else: 
          flag="yes" 
        else: 
          flag="first"
        ffn.updateRank(array(expected_output,dtype=int),flag)
        # output for verification
        #print count,expected_output, ffn.getOutput()
        neuron_index=list(ffn.getOutput()).index(max(ffn.getOutput()))
        if neuron_index == train_outputs[i]:
         epoch_accuracy+=1
     # print 'epoch= ',count,'  accuracy=',float(epoch_accuracy)/len(train_inputs)  
      count += 1
    print "train data accuracy :",float(epoch_accuracy)/len(train_inputs)  
      #preedicting  outputs on test data set
    correct_answers=0
    for i in range(len(test_inputs)):
     feed_input=[0,0]
     li=test_inputs[i]
     for j in range(len(li)):
       feed_input[li[j][0]]=li[j][1]
     ffn.forward(feed_input)
  #  ffn.updateRank(ffn.getOutput())
     #getting index of neuron with max value 
     neuron_index=list(ffn.getOutput()).index(max(ffn.getOutput()))
     if neuron_index == test_outputs[i]:
      correct_answers+=1
    accuracy=float (correct_answers)/len(test_inputs)
    print "accuracy test data = ",accuracy
