from __future__ import print_function
import numpy as np

# intial steps
dataSize = 10000  
X = np.array( [  np.random.randint(low = 0 ,high= 2  ,  size = dataSize ) 
                 , np.random.rand(dataSize)
                 , np.random.rand(dataSize) * 10  ] )

W = np.array( [  [-1.98760548 , -3.02631284 , -7.75574014] ,
                 [-2.78760548 , -2.82631284 , -9.75574014] ,
                 [-2.48760548 , -2.72631284 , -9.65574014] ,
                 [-2.98760548 , -3.02631284 , -6.75574014] 
                ]  )

B = np.random.randint(low = 0 ,high= 2  ,  size = 4 ).transpose()

F = np.array([ 0.3 , 0.9 , 0.2 , 0.1])

Y = np.random.randint(low = 0 ,high= 2  ,  size = dataSize).transpose()

N = 0.01

print('>>> Labels - Y ==> ' + str(Y))
print('\n>>> Inputs - X')
print (X)
  
# global variables
Y_ = Y 
Btemp = np.zeros( (dataSize , 4) )
for j in range(dataSize):
    Btemp[j] = B
B = np.transpose(Btemp)
print (B)

Z = np.dot(W, X) + B
H = sigmoid(Z).transpose()
E = np.absolute( np.power(Y - Y_ , 2) )
R = 0 
  
def sigmoid(x, derivative=False):
  sigm = 1. / (1. + np.exp(-x))
  if derivative:
      return sigm * (1. - sigm)
  return sigm
  
def start_iteration(Wtemp , Btemp , Xtemp ):
  Ztemp = np.dot(Wtemp, Xtemp) + Btemp
  Htemp = sigmoid(Ztemp).transpose()
  Y_temp = sigmoid(np.dot(Htemp , F))
  for j in range(dataSize):
    if Y_temp[j] > 0.5 :
      Y_temp[j] = 1
    else :
      Y_temp[j] = 0
 
  return Y_temp

def calc_error(Ytemp , Y_temp):
  Etemp = np.absolute( np.power(Ytemp - Y_temp , 2) )
  Rtemp = 0.5 * np.sum(Etemp)
  return Etemp , Rtemp

def show_state(index):
  print('----------------------------------------------------------\n@@@ Iteration ' + str(index) + ' results:')
  print('>>> Weights - W')
  print(W) 
  print('\n>>> Final weights - F ==> ' + str(F))
  print('>>> Expectations - Y_ ==> ' + str(Y_))
  print('>>> Error - E ==> ' + str(E))
  print('>>> Error Rate - R ==> ' + str(R))
  print('##############################################################')

def update_weights( W , B , F):
  global N
  global Y
  global Y_
  global X
  temp = N * (Y - Y_ ) * X
  temp2 = [ np.sum(temp[0]) , np.sum(temp[1]) , np.sum(temp[2]) ]
  W = W + temp2
  return W  , B , F

def do_epoch(index):
  global Y
  global Y_
  global E
  global R
  global W
  global X
  global F
  global B
  Y_ = start_iteration(W , B , X)
  E , R = calc_error(Y, Y_)
  show_state(index)
  W , B , F = update_weights(W , B , F)

def train_NN(times):
  for i in range(0, times):
      do_epoch(i)

train_NN(10)



