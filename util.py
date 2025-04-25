import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#def softmax(x):
#        c = np.max(x)
#        exp_x = np.exp(x - c)
#        sum_exp_x = np.sum(exp_x)
#        y = exp_x / sum_exp_x
#        
#        return y
 
def softmax(x):
    if x.ndim == 2:
        c = np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x - c)
        sum_exp_x = np.sum(exp_x, axis=1, keepdims=True)
        return exp_x / sum_exp_x
    else:
        c = np.max(x)
        exp_x = np.exp(x - c)
        return exp_x / np.sum(exp_x)
    
def ReLU(x):
    return np.maximum(0, x)
  
def CEE(y, t):
    if y.ndim == 1:
        y = y.reshape(1, -1)
        t = t.reshape(1)
    
    if t.ndim == 1:
        return -np.sum(np.log(y[np.arange(len(t)), t] + 1e-7)) / len(t)
    else:
        return -np.sum(t * np.log(y + 1e-7)) / y.shape[0]

def gradient(f, x):
    h = 0.0001

    shape = x.shape
    x_flatten = x.reshape(-1)
    size = len(x_flatten)

    grad = np.zeros_like(x_flatten)

    for i in range(size):
        x_val = x_flatten[i]

        x_flatten[i] = x_val + h
        fxh1 = f(x_flatten.reshape(shape))

        x_flatten[i] = x_val - h
        fxh2 = f(x_flatten.reshape(shape))

        grad[i] = (fxh1 - fxh2) / (2*h)

        x_flatten[i] = x_val
        
    return grad.reshape(shape)

def preprocess(text):
    text = text.lower().replace(".", " .")
    words = text.split(" ")
    
    word_to_id = {}
    id_to_word = {}
    corpus = np.array([], int)
    id = 0
    for word in words:
        if word not in word_to_id.keys():
            word_to_id[word] = id
            id_to_word[id] = word
            id += 1
        corpus = np.append(corpus, word_to_id[word])
    return corpus, word_to_id, id_to_word

def create_co_matrix(corpus, vocab_size, window_size=1):
    co_matrix = np.zeros((vocab_size, vocab_size))
    max_index = len(corpus) - 1
    for index, word_id in enumerate(corpus):
        window_left_index = index - window_size
        window_right_index = index + window_size
        
        if index < window_size:
            window_left_index = 0
        if index > max_index - window_size:
            window_right_index = max_index

        co_matrix[word_id][corpus[window_left_index:window_right_index+1]] += 1
        co_matrix[word_id][word_id] -= 1

    return co_matrix

def clip_grads(grads, max_norm):
    total_norm = 0
    for grad in grads:
        total_norm += np.sum(grad**2)
    total_norm = np.sqrt(total_norm)
    
    rate = max_norm / (total_norm + 1e-7)
    if rate < 1:
        for grad in grads:
            grad *= rate