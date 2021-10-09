#Benjamin Stanelle    1001534907
import sys
from sys import exit
import os
import numpy as np

def neural_network():
#===============Command Line Arguments======================================
    training_file=sys.argv[1]
    test_file= sys.argv[2]
    layers = int(sys.argv[3])
    if(layers<2):
        print("Layers must be 2 or larger., please enter a larger value.")
        exit(1)
    elif(layers==2):
        units_per_layer = int(sys.argv[4])
        rounds = int(sys.argv[5])
    else:
        rounds = int(sys.argv[4])

#===============Training STAGE======================================
    if os.path.exists(training_file):
        file_train= open(training_file, "r")
        
        if os.path.isfile(training_file):
            #command line arguments
            input_arr = np.genfromtxt(file_train, dtype= 'str')
            columns_train= len(input_arr[0])  #total number of columns in training file
            input_arr= np.hsplit(input_arr,([columns_train-1, columns_train-1])) #the input array split into 2 differen numpy sub arrs.
            train_arr=input_arr[0]  #input values array from text file
            features_arr=input_arr[2] #features array from text file
            
            train_arr= train_arr.astype(float) #converts train_arr from strings to floats
            #Convert class labels to new class labels Sn between 1-k
            #Convert input file to one hot vector, where the dimensionality of every vector is the number of classes (1,0,0)^T
            
            #train separate perceptrons as many as the number of classes, using only one of the dimensions of the hot vector
            #a single perceptron only recognizes elements from a specific class, determined by the one hot vector
            
            #input layers are not perceptrons they just input to the network
            
            #slide 7 calculate a array weighted sum for each layer starting from layer 2
            #activation function of that a array which results in z array(also starts from layer 2)
            
            
            
            #at the end calculate the sum of squared errors  for the output layer slide 9.
        else:
            print(training_file," is not a file.")
            exit(1)
            
    else:
        print("'", training_file, "' Path does not exist")
        exit(1)
        
        
#===============TESTING STAGE======================================
    #most of this is same as the training stage 
    if os.path.exists(test_file):
        file_test= open(test_file, "r")
        
        if os.path.isfile(test_file):
            input_arr_t = np.genfromtxt(file_test, dtype= 'str')
            columns_train_t= len(input_arr_t[0])  
            input_arr_t= np.hsplit(input_arr_t,([columns_train_t-1, columns_train_t-1]))
            test_arr_t=input_arr_t[0]  
            features_arr_t=input_arr_t[2] 

            test_arr_t= test_arr_t.astype(float)
            #09/27/2021  16:30
            
        else:
            print(test_file," is not a file.")
            exit(1)
            
    else:
        print("'", test_file, "' Path does not exist")
        exit(1)
neural_network()

