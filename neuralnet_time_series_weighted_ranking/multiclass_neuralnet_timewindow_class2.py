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
         
        # initialize weights randomly (+1 for bias)
        self.hWeights = random.random((self.nHidden, self.nIn+1))
        self.oWeights = random.random((self.nOut, self.nHidden+1))
         
        # activations of neurons (sum of inputs)
        self.hActivation = zeros((self.nHidden, 1), dtype=float)
        self.oActivation = zeros((self.nOut, 1), dtype=float)
         
        # outputs of neurons (after sigmoid function)
        self.iOutput = zeros((self.nIn+1, 1), dtype=float)      # +1 for bias
        self.hOutput = zeros((self.nHidden+1, 1), dtype=float)  # +1 for bias
        self.oOutput = zeros((self.nOut), dtype=float)
         
        # deltas for hidden and output layer
        self.hDelta = zeros((self.nHidden), dtype=float)
        self.oDelta = zeros((self.nOut), dtype=float)  
     
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
        self.oOutput = tanh(self.oActivation)
     
    def backward(self, teach):
        error = self.oOutput - array(teach, dtype=float)
        # deltas of output neurons
        self.oDelta = (1 - tanh(self.oActivation)) * tanh(self.oActivation) * error
                 
        # deltas of hidden neurons
        self.hDelta = (1 - tanh(self.hActivation)) * tanh(self.hActivation) * dot(self.oWeights[:,:-1].transpose(), self.oDelta)
                 
        # apply weight changes
        self.hWeights = self.hWeights - self.alpha * dot(self.hDelta, self.iOutput.transpose())
        self.oWeights = self.oWeights - self.alpha * dot(self.oDelta, self.hOutput.transpose())
     
    def getOutput(self):
        return self.oOutput


     
if __name__ == '__main__': 
    # define training set
    data=[line.rstrip().split('\t') for infile in os.listdir('./data') for line in file('./data/'+infile)]
    tdata=dataformat.trimmed_data(data)
    fdata=dataformat.formatted_dataset(tdata)
    print "\n data formatted"
    window_size=3
    input_neurons=window_size+2
    # create network
    ffn = FeedForwardNetwork(input_neurons, 3, 2) #using one per class coding 
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
       if (i+ ((input_neurons-2)+1)) < len(train_inputs):  
        # forward and backward pass
        feed_input=[0]*input_neurons # 2 lda values and 3 earlier sentiment values
        li=train_inputs[i+window_size]
        # li is of form [(0, 0.55527820230757163), (1, 0.43314313521528525)] but can vary in number of elements 
        for j in range(len(li)):
         feed_input[li[j][0]]=li[j][1]
        #populating the input array with last 3 sentiment values 
        k=2
        l=i
        while k<input_neurons:
         feed_input[k]= train_outputs[l]
         l+=1
         k+=1
        ffn.forward(feed_input)
        #now in feedbackwards we will have to feed a 2 valued vector 
        expected_output=[[0],[0]]
        expected_output[int(train_outputs[i+window_size])][0]=1
        ffn.backward(array(expected_output,dtype=int))
        # output for verification
        #print count,expected_output, ffn.getOutput()
        neuron_index=list(ffn.getOutput()).index(max(ffn.getOutput()))
        if neuron_index == train_outputs[i+window_size]:
         epoch_accuracy+=1
     # print 'epoch= ',count,'  accuracy=',float(epoch_accuracy)/len(train_inputs)  
      count += 1
    print "train data accuracy :",float(epoch_accuracy)/len(train_inputs)  
      #preedicting  outputs on test data set
    correct_answers=0
    for i in range(len(test_inputs)):
     if (i+ ((input_neurons-2)+1)) < len(test_inputs): 
      feed_input=[0] *input_neurons
      li=test_inputs[i+window_size]
      for j in range(len(li)):
       feed_input[li[j][0]]=li[j][1]
      k=2
      l=i
      while k<input_neurons:
       feed_input[k]= test_outputs[l]
       l+=1
       k+=1
      ffn.forward(feed_input)
      expected_output=[[0],[0]]
      expected_output[int(test_outputs[i+window_size])][0]=1
      neuron_index=list(ffn.getOutput()).index(max(ffn.getOutput()))
      if neuron_index == test_outputs[i+window_size]:
       correct_answers+=1
    accuracy=float (correct_answers)/len(test_inputs)
    print "accuracy test data = ",accuracy  
