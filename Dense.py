import numpy as np 
from activation import Activation

def init_params(input_size, hidden_size, output_size):
    # inisialisasi random dengan standar deviasi [-1 , 1]
    W1 = np.random.rand(hidden_size, input_size) * np.sqrt(2. / input_size) # untuk bobot input ke hidden
    b1 = np.zeros((hidden_size, 1))
    # b1 = np.random.rand(hidden_size, 1) * 2 - 1  # untuk bias input ke hidden
    W2 = np.random.rand(output_size, hidden_size) * np.sqrt(2. / input_size)  # untuk bobot hidden ke output
    b2 = np.zeros((output_size, 1))
    # b2 = np.random.rand(output_size, 1) * 2 - 1  # untuk bias hidden ke output
    return W1, b1, W2, b2 
        
def one_hot(Y):
    #input Y yang diharapkan array dari class label
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1    
    W2 = W2 - alpha * dW2  
    b2 = b2 - alpha * db2    
    return W1, b1, W2, b2

def update_params_adam(W, b, dW, db, mW, mb, vW, vb, beta1, beta2, alpha, epsilon, t):
    mW = beta1 * mW + (1 - beta1) * dW
    mb = beta1 * mb + (1 - beta1) * db
    vW = beta2 * vW + (1 - beta2) * (dW ** 2)
    vb = beta2 * vb + (1 - beta2) * (db ** 2)

    mW_corr = mW / (1 - beta1 ** t)
    mb_corr = mb / (1 - beta1 ** t)
    vW_corr = vW / (1 - beta2 ** t)
    vb_corr = vb / (1 - beta2 ** t)

    W -= alpha * mW_corr / (np.sqrt(vW_corr) + epsilon)
    b -= alpha * mb_corr / (np.sqrt(vb_corr) + epsilon)

    return W, b, mW, mb, vW, vb

def update_params_rmsprop(W, b, dW, db, vW, vb, beta2, alpha, epsilon):
    vW = beta2 * vW + (1 - beta2) * (dW ** 2)
    vb = beta2 * vb + (1 - beta2) * (db ** 2)

    W -= alpha * dW / (np.sqrt(vW) + epsilon)
    b -= alpha * db / (np.sqrt(vb) + epsilon)

    return W, b, vW, vb

