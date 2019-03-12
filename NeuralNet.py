import numpy as np
import random
import cv2
import math
import HOG
#storing all the HOGs in this 3D matrix
trainHist=np.zeros((20,7524))
#this array stores the class, 1 for Human 0 for No Human
label = []
print("/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/Reading Training Images..../-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/")
trainHist[0]=HOG.HOG_Vec("pos1.bmp")
label.append(1)
trainHist[1]=HOG.HOG_Vec("pos2.bmp")
label.append(1)
trainHist[2]=HOG.HOG_Vec("pos3.bmp")
label.append(1)
trainHist[3]=HOG.HOG_Vec("pos4.bmp")
label.append(1)
trainHist[4]=HOG.HOG_Vec("pos5.bmp")
label.append(1)
trainHist[5]=HOG.HOG_Vec("pos6.bmp")
label.append(1)
trainHist[6]=HOG.HOG_Vec("pos7.bmp")
label.append(1)
trainHist[7]=HOG.HOG_Vec("pos8.bmp")
label.append(1)
trainHist[8]=HOG.HOG_Vec("pos9.bmp")
label.append(1)
trainHist[9]=HOG.HOG_Vec("pos10.bmp")
label.append(1)
trainHist[10]=HOG.HOG_Vec("neg1.bmp")
label.append(0)
trainHist[11]=HOG.HOG_Vec("neg2.bmp")
label.append(0)
trainHist[12]=HOG.HOG_Vec("neg3.bmp")
label.append(0)
trainHist[13]=HOG.HOG_Vec("neg4.bmp")
label.append(0)
trainHist[14]=HOG.HOG_Vec("neg5.bmp")
label.append(0)
trainHist[15]=HOG.HOG_Vec("neg6.bmp")
label.append(0)
trainHist[16]=HOG.HOG_Vec("neg7.bmp")
label.append(0)
trainHist[17]=HOG.HOG_Vec("neg8.bmp")
label.append(0)
trainHist[18]=HOG.HOG_Vec("neg9.bmp")
label.append(0)
trainHist[19]=HOG.HOG_Vec("neg10.bmp")
label.append(0)
print("/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/.....HOGs Loaded............./-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/")

c = list(zip(trainHist,label))
random.shuffle(c)

X, Y = zip(*c)

def createNeuralNetwork(X,Y,no_hidden_neurons):
    np.random.seed(1)
    w1 = np.random.randn(no_hidden_neurons, len(X[0])) * 0.01
    b1 = np.zeros((no_hidden_neurons,1))
    w2 = np.random.randn(1,no_hidden_neurons) * 0.01
    b2 = np.zeros((1,1))
     # X = input_data of 20 images
     # Y = class labels
    
    model_param = {} # dictionary to store the weights and biases
    print("training the  neural network")
    
    variable=0
    cost = 0
    for i in range(0,200):
        
        cost = 0
        for j in range(0,len(X)):
            features = X[j].shape[0]
            q = X[j].reshape(1,features)
            q = q.T
            '''Neural network train'''
            v1 = w1.dot(q)+ b1   #Multiplication for Level 1 hidden layer
            a1 = ReLu(v1)
            v2 = w2.dot(a1) + b2
            a2 = sigmoid(v2)
            
            
            # Backward Propogation
            diff2 = (a2-Y[j])  *  derSigmoid(a2)    #finding the differene in value
            dw2 = np.dot(diff2,a1.T)
            db2 = np.sum(diff2,axis=1, keepdims=True)

            diff1 = w2.T.dot(diff2) * ReLuDerivation(a1)
            
            dw1 =  np.dot(diff1,q.T)
            db1 =  np.sum(diff1,axis=1, keepdims=True)

            #updating weights
            w1 = w1 - 0.01*dw1
            w2 = w2 - 0.01*dw2
            b1 = b1 - 0.01*db1
            b2 = b2 - 0.01*db2
            model_param = {'w1':w1,'b1':b1,'w2':w2,'b2':b2}
            
            '''End of neural network'''
            cost += (1.0/2.0)*((abs(a2-Y[j]))**2)
        cost_avg = cost/20.0 #calculating the avg cost for current epoch
        print("Epoch = ",i,"cost_avg = ",cost_avg)
        if i>1:
            if(abs(cost_avg - variable)<0.00001): #finding the difference between the avg cost between current epoch and the last epoch, if lesser than .00001 then break the loop, no more epochs needed
                break
        variable = cost_avg
        
    return model_param #all the updated weights and biases are stored and returned after the neural network is trained




# Returns value between 0 and 1.
def sigmoid(val):
    return 1.0 / (1.0 + np.exp(-val))

def derSigmoid(x):
    return x * (1-x)

# For ForwardFeed, It returns original value if it's >0 , or 0 if value
def ReLu(val):
    return val*(val>0)

# In Backward Propogration, It returns either 1 or 0
def ReLuDerivation(x):
    return 1. * (x > 0)

model_param = createNeuralNetwork(X,Y,250) #Function to train the Neural Network, the third parameter is no. of neurons in the hidden layer

#taking in all the test images in array image_Data
n = 1
image_Data = []
for n in range(1,11):
    name = "test1("+str(n)+").bmp"
    Vector1=HOG.HOG_Vec(name)
    image_Data.append(Vector1)



def predict(X,model_param):
    w1 = model_param['w1']
    w2 = model_param['w2']
    b2 = model_param['b2']
    b1 = model_param['b1']
    features = X.shape[0]
    q = X.reshape(1,features)
    v1 = np.dot(w1,q.T) + b1   #Multiplication for Level 1 hidden layer
    a1 = ReLu(v1)
    v2 = np.dot(w2,a1) + b2  #Multiplication for Output layer
    a2 = sigmoid(v2)
    return a2

for TestImage in image_Data:
    if(predict(TestImage,model_param)>=0.5):
        print("Human Detected")
    else:
        print("Human Not Detected")
    print(predict(TestImage,model_param))