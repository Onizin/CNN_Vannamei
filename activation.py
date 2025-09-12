import numpy as np 
import torch 

class Activation():
    def Relu(Z):
        return np.maximum(Z,0)
    
    def Softmax(Z):
        expZ = np.exp(Z - np.max(Z))
        return expZ / expZ.sum(axis=0, keepdims=True)
        # A = np.exp(Z)/sum(np.exp(Z))
        # return A
    
    def ReLU_deriv(Z):
     return Z > 0
    
class Activation_Conv2d():
   def Relu_custom(X):
    return torch.maximum(X, torch.tensor(0.0, dtype=X.dtype))