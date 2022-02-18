#Benjamin Stanelle    1001534907
import sys
from sys import exit
import os
import numpy as np
import math
import random

def neural_network():
#===============Command Line Arguments======================================
    training_file=sys.argv[1]
    test_file= sys.argv[2]
    layers = int(sys.argv[3])
    if(layers<2):
        print("Layers must be 2 or larger, please enter a larger value.")
        exit(1)
    elif(layers==2):
        rounds = int(sys.argv[4])
    else:
        units_per_layer = int(sys.argv[4])
        rounds = int(sys.argv[5])

#===============Training STAGE======================================
    #Passing on command line "C:\Users\benji\.spyder-py3\pendigits_string_training.txt C:\Users\benji\.spyder-py3\pendigits_string_test.txt 3 20 20"
    if os.path.exists(training_file):
        file_train= open(training_file, "r")
        
        if os.path.isfile(training_file):
            #command line arguments and intializations
            input_arr = np.genfromtxt(file_train, dtype= 'str')
            columns_train= len(input_arr[0])  #total number of columns in training file
            rows_train= len(input_arr)
            input_arr= np.hsplit(input_arr,([columns_train-1, columns_train-1])) #the input array split into 2 differen numpy sub arrs.
            train_arr=input_arr[0]  #input values array from text file
            features_arr=input_arr[2] #features array from text file
            
            train_arr= train_arr.astype(float) #converts train_arr from strings to floats
            
            #normalizing the numpy array by dividing ever value in the dataset by the highest value
            max_value= abs(np.amax(train_arr)) #finds max value in 2d array
            for i in range(rows_train): 
                for j  in range(columns_train-1): #columns input data excluding last column which are features
                    train_arr[i][j]= train_arr[i][j]/max_value
    
            #--------------One Hot Vector-----------------------
            #Convert class labels to new class labels Sn between 1-k
            #Convert input file to one hot vector, where the dimensionality of every vector is the number of classes (1,0,0)^T
            Qn_arr=np.array(features_arr[0]) #manually putting in the first class
            for i in range(rows_train):
                if not (features_arr[i] in Qn_arr):
                    Qn_arr= np.append(Qn_arr, features_arr[i], axis = 0) #adding class labels to Qn_arr that don't exist in it
            index=0
            Sn_arr= np.empty([0], dtype = int)

            for i in range(len(features_arr)):
                index = list(Qn_arr).index(features_arr[i])
                Sn_arr= np.append(Sn_arr, [index], axis=0)
            one_hot_vector = np.zeros([ len(features_arr),len(Qn_arr)], dtype=int)
            for i in range(len(one_hot_vector)):
                one_hot_vector[i][Sn_arr[i]] = 1
            
            #+++++++++++++++Initializing the network weights++++++++++++

            if(layers==2):
                #intialize weights and bias
                weights, bias= INIT_WEIGHTS_BIAS_2LAYER(layers, Qn_arr, columns_train)
                
                #going through the network
                learn_rate = 1
                for t_rounds in range(rounds):
                    for t in range(rows_train): #for each training example
            
                        #intializing Z and A array at the first layer
                        Z= []
                        a= []
                        for l in range(layers):
                            Z.append([])
                            a.append([])
                        # takes dimensionality of input (AKA # of columns)
                        for d in range(columns_train-1):
                            Z[0].append([])
                            
                        for d in range(columns_train-1): #put each dimension of a specific training input t, into the first layer of z
                            Z[0][d]= train_arr[t][d]
        
                        units_p_layer = len(Qn_arr)
                            #initializing our Z and a arrays for all layers after the first
                        for l in range(1, layers):

                                
                                #intializing a and z for all layers
                            for i in range(units_p_layer): #adding the units per layer
                                a[l].append([])
                                Z[l].append([])
                                
                                a_val=0.0
                                dimensionality= columns_train-1
                                    
                                for j in range(dimensionality):
                                    a_val+= (weights[l-1][i][j][0]) * (Z[l-1][j]) #current weight times previous layer activation function e.g. weight at layer 2 z at layer 1
                                
                                a[l][i]= bias[l-1][i][0] + a_val
                                Z[l][i]= 1/(1+ (math.pow(math.e, (-a[l][i])))) #Activation function: Simoid of a
                        #----------------------Back Propagation---------------------
                        phi= []
                        for l in range(layers): #declaring layers in phi
                            phi.append([])
                        for i in range(len(Qn_arr)):#declaring the last layers unit size
                            phi[layers-1].append([])
                            
                        for i in range(len(Qn_arr)): #calculating the last layers value
                            phi[layers-1][i].append( (Z[layers-1][i]- one_hot_vector[t][i])*(Z[layers-1][i])*(1- Z[layers-1][i]) )

                        for l in range(layers-2, 0, -1):
                            for i in range(units_per_layer):
                                phi[l].append([])
                        
                                phi_temp=0

                                units_p_layer= len(Qn_arr)

                                for k in range(units_p_layer):
                                    phi_temp+=(phi[l+1][k][0]* weights[l][k][i][0])
                                phi[l][i].append((phi_temp)*(Z[l][i])*(1-Z[l][i]))
                        for l in range(1, layers):
                            for i in range(units_p_layer):
                                bias[l-1][i][0]= (bias[l-1][i][0]) - (learn_rate*phi[l][i][0])

                                #units_p= len(Qn_arr)
                                for j in range(columns_train-1):           
                                    weights[l-1][i][j][0]= (weights[l-1][i][j][0] - (learn_rate*phi[l][i][0]*Z[l-1][j]))
                    learn_rate =learn_rate*0.98
                TEST(weights, bias, test_file, layers, len(Qn_arr), rounds, Qn_arr)
                    
            
            elif(layers > 2):
                #intialize weights and bias
                weights, bias= INIT_WEIGHTS_BIAS(layers, units_per_layer, Qn_arr, columns_train)
                
                learn_rate = 1
                for t_rounds in range(rounds):
                    for t in range(rows_train): #for each training example
            
                        #intializing Z and A array at the first layer
                        Z= []
                        a= []
                        for l in range(layers):
                            Z.append([])
                            a.append([])
                        #very first layer takes dimensionality of input (AKA # of columns)
                        for d in range(columns_train-1):
                            Z[0].append([])
                            
                        for d in range(columns_train-1): #put each dimension of a specific training input t, into the first layer of z
                            Z[0][d]= train_arr[t][d]
        
                            #initializing our Z and a arrays for all layers after the first
                        for l in range(1, layers):
                            if not(l == layers-1): #checking to see if its not the final layer
                                units_p_layer= units_per_layer 
                
                            else:
                                units_p_layer= len(Qn_arr) #checking to see if its the final layer
                                
                                #intializing a and z for all layers
                            for i in range(units_p_layer): #adding the units per layer
                                a[l].append([])
                                Z[l].append([])
                                
                                a_val=0.0
                                if l==1: #if we are at the 2nd layer then the dimensionality is the dimensionality of the input
                                    dimensionality= columns_train-1
                                elif l==layers-1: #if we are at the last layer the dimensionality is the number of classes
                                    dimensionality= len(Qn_arr)
                                    
                                else: #if we at a hidden layer between (not including) the 2nd and last layer
                                    dimensionality= units_per_layer
                                for j in range(dimensionality):
                                    a_val+= (weights[l-1][i][j][0]) * (Z[l-1][j]) #current weight times previous layer activation function e.g. weight at layer 2 z at layer 1
                                
                                a[l][i]= bias[l-1][i][0] + a_val
                                Z[l][i]= 1/(1+ (math.pow(math.e, (-a[l][i])))) #Activation function: Simoid of a
                        #----------------------Back Propagation---------------------
                        phi= []
                        for l in range(layers): #declaring layers in phi
                            phi.append([])
                        for i in range(len(Qn_arr)):#declaring the last layers unit size
                            phi[layers-1].append([])
                            
                        for i in range(len(Qn_arr)): #calculating the last layers value
                            phi[layers-1][i].append( (Z[layers-1][i]- one_hot_vector[t][i])*(Z[layers-1][i])*(1- Z[layers-1][i]) )

                        for l in range(layers-2, 0, -1):
                            for i in range(units_per_layer):
                                phi[l].append([])
                        
                                phi_temp=0
                                if (l+1 == layers-1): 
                                    units_p_layer= len(Qn_arr)
                
                                else:
                                    units_p_layer= units_per_layer
                                for k in range(units_p_layer):
                                    phi_temp+=(phi[l+1][k][0]* weights[l][k][i][0])
                                phi[l][i].append((phi_temp)*(Z[l][i])*(1-Z[l][i]))
                        for l in range(1, layers):
                            if not(l == layers-1): #checking to see if its not the final layer
                                units_p_layer= units_per_layer 
                
                            else:
                                units_p_layer= len(Qn_arr) 
                            for i in range(units_p_layer):
                                bias[l-1][i][0]= (bias[l-1][i][0]) - (learn_rate*phi[l][i][0])

                                if(l-1==0):
                                    units_p= columns_train-1
                                else:
                                    units_p=units_per_layer
                                for j in range(units_p-1):
                                   
                                    weights[l-1][i][j][0]= (weights[l-1][i][j][0] - (learn_rate*phi[l][i][0]*Z[l-1][j]))
                    learn_rate =learn_rate*0.98

                                

                TEST(weights, bias, test_file, layers, units_per_layer, rounds, Qn_arr)
                        
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
        

def INIT_WEIGHTS_BIAS_2LAYER(layers, Qn_arr, columns_train):
    weights = []
    bias = []
    for l in range(layers-1):
        weights.append([])
        bias.append([])
    for u in range(len(Qn_arr)): 
        #layer 0
        weights[layers-2].append([])
        bias[layers-2].append([])
        bias[layers-2][u].append(random.uniform(-0.05, 0.05))
        for d in range(columns_train):
            weights[layers-2][u].append([])
            weights[layers-2][u][d].append(random.uniform(-0.05, 0.05))
    return weights, bias
        
        
        
#this of this process sequentially, one training example goes through at a time
#the first layers number of nodes is the same as the dimensionality of the input, which is the number of columns
#but for the first layer there are no weights
def INIT_WEIGHTS_BIAS(layers, units_per_layer, Qn_arr, columns_train):
    #intialize weights and bias
    weights= []
    bias= []
                
   # weights_l2= [units_per_layer][columns_train-1]
   # weights_hidden= [units_per_layer] [units_per_layer] 
  #weights_out_layer= [len(Qn_arr)] [units_per_layer]
                
    #intializing the layers arrays
    for l in range(layers-1):
        weights.append([])
        bias.append([])
                #Initializing weights and bias at layer 2(special case of hidden layer)
    for u in range(units_per_layer):
        weights[0].append([])
        bias[0].append([])
        bias[0][u].append(random.uniform(-0.05, 0.05))
        for d in range(columns_train-1):
            weights[0][u].append([])
            weights[0][u][d].append(random.uniform(-0.05, 0.05))
                
                #intializing all hidden layers weights and bias
    if(layers >3):
        for l in range(1, layers-2): #doesn't touch layer 2 or the last layer
            for u in range(units_per_layer):
                weights[l].append([])
                bias[l].append([])
                bias[l][u].append(random.uniform(-0.05, 0.05))
                for b in range(units_per_layer):
                    weights[l][u].append([])
                    weights[l][u][b].append(random.uniform(-0.05, 0.05))
                
                #intializing weights and bias for the last layer
    for u in range(len(Qn_arr)): 
        weights[layers-2].append([])
        bias[layers-2].append([])
        bias[layers-2][u].append(random.uniform(-0.05, 0.05))
        for d in range(units_per_layer):
            weights[layers-2][u].append([])
            weights[layers-2][u][d].append(random.uniform(-0.05, 0.05))
    return weights, bias


#===============TESTING STAGE======================================
#use original classes to test: Qn_arr
def TEST(weights, bias, test_file, layers, units_per_layer, rounds, Qn_arr):
    if os.path.exists(test_file):
        
        if os.path.isfile(test_file):
            input_arr = np.genfromtxt(test_file, dtype= 'str')
            columns_test= len(input_arr[0])  
            rows_test= len(input_arr)
            input_arr= np.hsplit(input_arr,([columns_test-1, columns_test-1])) #the input array split into 2 differen numpy sub arrs.
            test_arr=input_arr[0]  #input values array from text file
            features_test_arr=input_arr[2] #features array from text file
            
            test_arr= test_arr.astype(float) #converts train_arr from strings to floats

            #normalizing the numpy array by dividing ever value in the dataset by the highest value
            max_value= abs(np.amax(test_arr)) #finds absolute of the max value in 2d array
            for i in range(rows_test): 
                for j  in range(columns_test-1): #columns input data excluding last column which are features
                    test_arr[i][j]= test_arr[i][j]/max_value
    
            #--------------One Hot Vector-----------------------
            #Convert class labels to new class labels Sn between 1-k
            #Convert input file to one hot vector, where the dimensionality of every vector is the number of classes (1,0,0)^T
            classify= 0
            for t in range(rows_test): #for each training example
            
                #intializing Z and A array at the first layer
                Z= []
                a= []
                for l in range(layers):
                    Z.append([])
                    a.append([])
                    #very first layer takes dimensionality of input (AKA # of columns)
                for d in range(columns_test-1):
                        Z[0].append([])
                            
                for d in range(columns_test-1): #put each dimension of a specific training input t, into the first layer of z
                        Z[0][d]= test_arr[t][d]
        
        #initializing our Z and a arrays for all layers after the first
                for l in range(1, layers):
                    if not(l == layers-1): #checking to see if its not the final layer
                        units_p_layer= units_per_layer 
                
                    else:
                        units_p_layer= len(Qn_arr) #checking to see if its the final layer
                                
                                #intializing a and z for all layers
                    for i in range(units_p_layer): #adding the units per layer
                        a[l].append([])
                        Z[l].append([])
                                
                        a_val=0.0
                        if l==1: #if we are at the 2nd layer then the dimensionality is the dimensionality of the input
                            dimensionality= columns_test-1
                            
                        elif l==layers-1: #if we are at the last layer the dimensionality is the number of classes
                            dimensionality= len(Qn_arr)
                                    
                        else: #if we at a hidden layer between (not including) the 2nd and last layer
                            dimensionality= units_per_layer
                            
                        for j in range(dimensionality):
                            a_val+= (weights[l-1][i][j][0]) * (Z[l-1][j]) #current weight times previous layer activation function e.g. weight at layer 2 z at layer 1
                        a[l][i]= bias[l-1][i][0] + a_val
                        Z[l][i]= 1/(1+ (math.pow(math.e, (-a[l][i])))) #Activation function: Simoid of a
                
                
                index = Z[layers-1].index(max(Z[layers-1]))
                accuracy =0
                ties=0
                for p in range(len(Z[layers-1])):
                    if (max(Z[layers-1]) == Z[layers-1][p]):
                        ties +=1
                if ties > 1:
                    accuracy/ties
                if(Qn_arr[index]== features_test_arr[t]):
                    accuracy= 1
                print("ID=%5d, predicted=%10s, true=%10s, accuracy=%4.2f" %(t+1, Qn_arr[index], features_test_arr[t][0], accuracy))
                

                if(Qn_arr[index] == features_test_arr[t]):
                    classify += 1

            print("classification accuracy: %6.4f"%(classify/rows_test))
            
        else:
            print(test_file," is not a file.")
            exit(1)
            
    else:
        print("'", test_file, "' Path does not exist")
        exit(1)
    
    
    
neural_network()

