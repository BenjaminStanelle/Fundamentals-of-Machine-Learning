#Benjamin Stanelle    1001534907
import sys
from sys import exit
import os
import numpy as np

def linear_regression():
    training_file=sys.argv[1]
    test_file= sys.argv[2]
    degree = int(sys.argv[3])
    lambd = float(sys.argv[4])
    
    #Dimensionality of Phi = M =D * Degree + 1
    if os.path.exists(training_file):
        file_train= open(training_file, "r")
        
        if os.path.isfile(training_file):
            input_arr = np.loadtxt(file_train, dtype=float)
            columns_train= len(input_arr[0])  #total number of columns in training file
            M= ((columns_train-1)*degree) + 1 #input row length * degree +1
            Identity= np.identity(M, dtype =float)
            input_arr= np.hsplit(input_arr,([columns_train-1, columns_train-1]))
            train_arr=input_arr[0]  #input values array from text file
            target_arr=input_arr[2] #Target values array from text file
            phi= np.ones(shape=(len(train_arr),M))

            #building the Phi Matrix
            for i in range(len(train_arr)):
                temp=1
                for j in range(columns_train-1):
                    phi[i][temp:(temp+degree)]= train_arr[i][j]
                    temp =temp+degree
            
            #If the degree is 2, then φ(x) = (1, x1, (x1)2, x2, (x2)2..., xD, (xD)2)T.
            #This seg of code does this operatior for whatever the degree is on all elements
            #of our phi MATRIX except the first element in every row(which is the number 1)
            if degree>1:
                for i in range(len(train_arr)):
                    for j in range(1,M):
                        temp= ((j-1)%degree)+1
                        phi[i][j]= (phi[i][j]) ** temp

            #𝒘 = (𝜆I + Φ^(𝑇) * Φ)^(−1)Φ^(𝑇) * 𝐭
            w= (np.dot(lambd,Identity)) + (np.dot(phi.T, phi))
            w= (np.linalg.pinv(w))
            w= np.dot(w,phi.T)
            w= np.dot(w, target_arr)
            
            #w_list=w.tolist()
            i=0
            np.set_printoptions(precision=4)
            for x in w:
                print("w%d="%i, x, sep='')
                i+=1
            
        else:
            print(training_file," is not a file.")
            exit(1)
            
    else:
        print("'", training_file, "' Path does not exist")
        exit(1)
        
        
    #===============TESTING STAGE======================================
    #most of this is same as the training stage we just don't do the last part for finding w
    if os.path.exists(test_file):
        file_test= open(test_file, "r")
        
        if os.path.isfile(test_file):
            input_arr_t = np.loadtxt(file_test, dtype=float)
            columns_train_t= len(input_arr_t[0])  
            M_t= ((columns_train_t-1)*degree) + 1 
            input_arr_t= np.hsplit(input_arr_t,([columns_train_t-1, columns_train_t-1]))
            test_arr_t=input_arr_t[0]  
            target_arr_t=input_arr_t[2] 
            phi_t= np.ones(shape=(len(test_arr_t),M_t))
            
            #building the Phi Matrix for test data
            for i in range(len(test_arr_t)):
                temp=1
                for j in range(columns_train_t-1):
                    phi_t[i][temp:(temp+degree)]= test_arr_t[i][j]
                    temp =temp+degree
            
            #If the degree is 2, then φ(x) = (1, x1, (x1)2, x2, (x2)2..., xD, (xD)2)T.
            #This seg of code does this operatior for whatever the degree is on all elements
            #of our phi MATRIX except the first element in every row(which is the number 1)
            if degree>1:
                for i in range(len(test_arr_t)):
                    for j in range(1,M_t):
                        temp= ((j-1)%degree)+1
                        phi_t[i][j]= (phi_t[i][j]) ** temp
            
            #Linear regression on the test data
            #y(x,w) = 𝒘^(𝑇) * 𝜑(x)
            y= np.dot(w.T,phi_t.T)
            
            
            target_arr_t_list=target_arr_t.tolist()
            list1=[]
            for i in range(len(target_arr_t_list)):
                list1.append(target_arr_t_list[i].pop())
            stnd_error=[]
            for j in range(len(y[0])):      
                stnd_error.append((((list1[j]) - (y[0][j]))**2))

            #Printing information
            for j in range(len(y[0])):
                print("ID=%5d, output=%14.4f, target value =%10.4f, squared error = %.4f"%(j+1, y[0][j],list1[j], stnd_error[j]))
                    
        else:
            print(test_file," is not a file.")
            exit(1)
            
    else:
        print("'", test_file, "' Path does not exist")
        exit(1)
linear_regression()