import numpy as np
from util import *
import pickle

class MatMul:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.x = None
    
    def forward(self, x):
        W = self.params[0]
        self.x = x
        
        return np.dot(x, W)
     
    def backward(self, dout):
        W = self.params[0]
        dx = np.dot(dout, W.T)
        dW = np.dot(self.x.T, dout)
        self.grads[0][...] = dW
        
        return dx
         
class Sigmoid:
    def __init__(self):
        self.params = []
        self.grads = []
        self.x = None
        self.out = None
        
    def forward(self, x):
        self.x = x
        self.out = 1 / (1 + np.exp(-self.x))
        return self.out
    
    def backward(self, dout):
        return dout * (self.out * (1 - self.out))

class ReLU:
    def __init__(self):
        self.x = None
        
    def forward(self, x):
        self.x = x
        return (x > 0)*x
    
    def backward(self, dz):
        return dz*(self.x > 0).astype(int)
    
    
class Affine:
    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None 
        
    def forward(self, x):
        W, b = self.params
        self.x = x
        return np.dot(x, W) + b
    
    def backward(self, dout):
        W, _ = self.params
        
        db = np.sum(dout, sum=0)
        dW = np.dot(self.x.T, dout)
        dx = np.dot(dout, W.T)
        
        self.grads[0][...] = dW
        self.grads[1][...] = db
        
        return dx

class Softmax_with_Loss:
    def __init__(self):
        self.y = None
        self.t = None
        
    def forward(self, X, t):
        self.y = softmax(X)
        self.t = t
        loss = CEE(self.y, self.t)
        
        return loss
    
    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        
        # one-hot 벡터 레이블, 정수 레이블 모두 호환
        if self.t.ndim == 1:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
        else:
            dx = (self.y - self.t)
        
        return dx / batch_size

class Dropout:
    def __init__(self, dropout_ratio=0.5):
        self.params = []
        self.grads = []
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, train_flg=True):
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        return dout * self.mask
    
class Embedding:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.index = None
        
    def forward(self, index):
        W = self.params[0]
        self.index = index
        return W[index]
    
    #def backward(self, dout):
    #    self.grads[0][...] = 0
    #    for idx, id in enumerate(self.index):
    #        self.grads[0][id] += dout[idx]
    
    def backward(self, dout):
        dW = self.grads[0]
        dW[...] = 0

        # index와 dout 길이 맞추기
        index = self.index
        if index.ndim == 2:
            N, T = index.shape
            index = index.reshape(N * T)
            dout = dout.reshape(N * T, -1)

        for idx, word_id in enumerate(index):
            dW[word_id] += dout[idx]
            
class RNN:
    def __init__(self, W_x, W_h, b):
        self.params = [W_x, W_h, b]
        self.grads = [np.zeros_like(W_x), np.zeros_like(W_h), np.zeros_like(b)]
        self.cache = None
        
    def forward(self, x, h_prev):
        # x, W_x : N x D, D x H
        # h, W_h : N x H, H x H
        # b : N x 1
        W_x, W_h, b = self.params
        
        x_proj = np.dot(x, W_x)         # N x H
        h_proj = np.dot(h_prev, W_h)    # N x H
        proj_sum = x_proj + h_proj      # N x H
        h = proj_sum + b                # N x H
        h_next = np.tanh(h)             # N x H
        
        self.cache = (x, h_prev, h, h_next)
        return h_next
    
    def backward(self, dh_next):
        W_x, W_h, b = self.params
        x, h_prev, h, h_next = self.cache
        
        dtanh = dh_next * (1 - np.tanh(h)**2) # N x H
        
        db = np.sum(dtanh, axis=0) # 1 x H
        dproj_sum = dtanh # N x H
        
        dx_proj = dproj_sum # N x H
        dh_proj = dproj_sum # N x H
        
        dW_x = np.dot(x.T, dx_proj) # D x H
        dx = np.dot(dx_proj, W_x.T) # N x D
        
        dW_h = np.dot(h_prev.T, dh_proj)
        dh_prev = np.dot(dh_proj, W_h.T) # N x H
        
        self.grads[0][...] = dW_x
        self.grads[1][...] = dW_h
        self.grads[2][...] = db
        
        return dx, dh_prev
    
#class TimeEmbedding:
#    def __init__(self, W):
#        self.params = [W]
#        self.grads = [np.zeros_like(W)]
#        self.idx = None
#
#    def forward(self, sw):
#        W = self.params[0]
#        self.idx = sw
#        out = W[sw]  # numpy indexing: (N, T) → (N, T, D)
#        return out
#
#    def backward(self, dout):
#        dW = self.grads[0]
#        dW[...] = 0
#
#        if self.idx.ndim == 2:
#            N, T = self.idx.shape
#            idx = self.idx.reshape(N * T)
#            dout = dout.reshape(N * T, -1)
#        else:
#            idx = self.idx
#
#        for i, word_id in enumerate(idx):
#            dW[word_id] += dout[i]
#
#class TimeAffine:
#    def __init__(self, W, b):
#        self.params = [W, b]
#        self.grads = [np.zeros_like(W), np.zeros_like(b)]
#        self.x = None
#
#    def forward(self, x):
#        N, T, H = x.shape
#        W, b = self.params
#        out = np.dot(x.reshape(N * T, H), W) + b  # (N*T, V)
#        self.x = x
#        return out.reshape(N, T, -1)              # (N, T, V)
#
#    def backward(self, dout):
#        x = self.x
#        N, T, H = x.shape
#        W, b = self.params
#
#        dout = dout.reshape(N * T, -1)  # (N*T, V)
#        x_reshaped = x.reshape(N * T, H)
#
#        dW = np.dot(x_reshaped.T, dout)
#        db = np.sum(dout, axis=0)
#        dx = np.dot(dout, W.T).reshape(N, T, H)
#
#        self.grads[0][...] = dW
#        self.grads[1][...] = db
#        return dx
#
#class TimeSoftmaxWithLoss:
#    def __init__(self):
#        self.params = []
#        self.grads = []
#        self.loss = 0
#        self.layers = []
#        self.ts = None  
#
#    def forward(self, xs, ts):
#        N, T, V = xs.shape
#        self.ts = ts
#        self.loss = 0
#        self.layers = []
#
#        for i in range(T):
#            layer = Softmax_with_Loss()
#            loss = layer.forward(xs[:,i, :], ts[:, i])
#            self.loss += loss
#            self.layers.append(layer)
#
#        self.loss /= T
#        return self.loss
#
#    def backward(self, dout=1):
#        N, T = self.ts.shape
#        dxs = []
#
#        dout /= T  # 평균에 맞춰서 스케일링
#
#        for i in range(T-1, -1, -1):
#            dx = self.layers[i].backward(dout)  # (N, V)
#            dxs.append(dx)
#
#        # (T, N, V) → (N, T, V)
#        return np.flip(np.array(dxs), axis=0).transpose(1, 0, 2)

#######################################################################################

#class TimeEmbedding:
#    def __init__(self, W):
#        self.params = [W]
#        self.grads = [np.zeros_like(W)]
#        self.layers = None
#
#    def forward(self, xs):
#        W = self.params[0]
#        N, T = xs.shape
#        V, D = W.shape
#        
#        print("📌 W.shape =", W.shape)
#        print("📌 xs.max() =", xs.max())
#        
#        out = np.empty((N, T, D), dtype='f')
#        self.layers = []
#
#        for t in range(T):
#            layer = Embedding(W)
#            out[:, t, :] = layer.forward(xs[:, t])
#            self.layers.append(layer)
#
#        return out
#
#    def backward(self, dout):
#        N, T, D = dout.shape
#        W = self.params[0]
#    
#        grad = np.zeros_like(W)
#        for t in range(T):
#            layer = self.layers[t]
#            layer.backward(dout[:, t, :])
#            grad += layer.grads[0]
#
#        self.grads[0][...] = grad
#        return None

class TimeEmbedding:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.idx = None

    def forward(self, idx):
        W = self.params[0]
        self.idx = idx
        out = W[idx].astype('float32')  # (N, T, D) ← 자동으로 벡터화됨
        return out

    def backward(self, dout):
        dW = self.grads[0]
        dW[...] = 0

        if self.idx.ndim == 2:
            N, T = self.idx.shape
            dout = dout.reshape(N * T, -1)
            idx = self.idx.reshape(N * T)
        else:
            idx = self.idx

        for i, word_id in enumerate(idx):
            dW[word_id] += dout[i]

class TimeAffine:
    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None

    def forward(self, x):
        N, T, D = x.shape
        W, b = self.params
        
        x = x.astype('float32')
        rx = x.reshape(N*T, -1)
        out = np.dot(rx, W) + b
        self.x = x
        return out.reshape(N, T, -1)

    def backward(self, dout):
        x = self.x
        N, T, D = x.shape
        W, b = self.params

        dout = dout.reshape(N*T, -1)
        rx = x.reshape(N*T, -1)

        db = np.sum(dout, axis=0)
        dW = np.dot(rx.T, dout)
        dx = np.dot(dout, W.T)
        dx = dx.reshape(*x.shape)

        self.grads[0][...] = dW
        self.grads[1][...] = db

        return dx


class TimeSoftmaxWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.cache = None
        self.ignore_label = -1

    def forward(self, xs, ts):
        N, T, V = xs.shape

        if ts.ndim == 3:  # 정답 레이블이 원핫 벡터인 경우
            ts = ts.argmax(axis=2)

        mask = (ts != self.ignore_label)

        # 배치용과 시계열용을 정리(reshape)
        xs = xs.reshape(N * T, V)
        ts = ts.reshape(N * T)
        mask = mask.reshape(N * T)

        ys = softmax(xs)
        ls = np.log(ys[np.arange(N * T), ts])
        ls *= mask  # ignore_label에 해당하는 데이터는 손실을 0으로 설정
        loss = -np.sum(ls)
        loss /= mask.sum()

        self.cache = (ts, ys, mask, (N, T, V))
        return loss

    def backward(self, dout=1):
        ts, ys, mask, (N, T, V) = self.cache

        dx = ys
        dx[np.arange(N * T), ts] -= 1
        dx *= dout
        dx /= mask.sum()
        dx *= mask[:, np.newaxis]  # ignore_label에 해당하는 데이터는 기울기를 0으로 설정

        dx = dx.reshape((N, T, V))

        return dx
    
#######################################################################################

class TimeRNN:
    def __init__(self, W_x, W_h, b, stateful=True):
        self.params = [W_x, W_h, b]
        self.grads = [np.zeros_like(W_x), np.zeros_like(W_h), np.zeros_like(b)]
        self.stateful = stateful
        
        self.h, self.dh = None, None
        self.layers = []
        
    def set_state(self, h):
        self.h = h
        
    def reset_state(self):
        self.h = None
        
    def forward(self, xs):
        W_x, W_h, b = self.params
        N, T, D = xs.shape
        _, H = W_h.shape
        
        hs = []
        
        if not self.stateful or self.h is None:
            self.h = np.zeros((N, H), dtype='float')
            
        for i in range(T):
            rnn = RNN(W_x, W_h, b)
            self.h = rnn.forward(xs[:, i, :], self.h)
            hs.append(self.h)
            
            #hs[:, i, :] = self.h
            self.layers.append(rnn)
        
        return np.array(hs).transpose(1, 0, 2)
    
    def backward(self, dhs):
        _, T, _ = dhs.shape
        
        dxs = []
        dh_next = 0
        
        for i in range(len(self.grads)):
            self.grads[i][...] = 0
            
        for i in range(T-1, -1, -1):
            dh_cur = dhs[:, i, :]
            dx, dh_prev = self.layers[i].backward(dh_cur + dh_next)
            dxs.append(dx)
            
            for j, grad in enumerate(self.layers[i].grads):
                self.grads[j] += grad
                
            dh_next = dh_prev
        
        self.dh = dh_prev
        
        return np.flip(np.array(dxs), axis=0)
             
class SimpleRNNLM:
    def __init__(self, vocab_size, embedding_size, hidden_size):
        self.D = embedding_size
        self.H = hidden_size
        self.V = vocab_size
        self.params = []
        self.grads = []
        self.logit = None

        # 안정적인 초기화
        self.W_embed = 0.01 * np.random.randn(self.V, self.D)
        self.W_x_rnn = (1 / np.sqrt(self.D)) * np.random.randn(self.D, self.H)
        self.W_h_rnn = (1 / np.sqrt(self.H)) * np.random.randn(self.H, self.H)
        self.b_rnn = np.zeros(self.H, dtype='float32')
        self.W_affine = (1 / np.sqrt(self.H)) * np.random.randn(self.H, self.V)
        self.b_affine = np.zeros(self.V, dtype='float32')

        # 레이어 구성
        self.layers = []
        self.layers.append(TimeEmbedding(self.W_embed))
        self.layers.append(TimeRNN(self.W_x_rnn, self.W_h_rnn, self.b_rnn))
        self.layers.append(TimeAffine(self.W_affine, self.b_affine))
        self.last_layer = TimeSoftmaxWithLoss()

        for layer in self.layers:
            for param, grad in zip(layer.params, layer.grads):
                self.params.append(param)
                self.grads.append(grad)

    def forward(self, xs, ts):
        out = xs
        for layer in self.layers:
            out = layer.forward(out)
        self.logit = out
        loss = self.last_layer.forward(out, ts)
        return loss

    def backward(self):
        dout = self.last_layer.backward()
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def reset_state(self):
        self.layers[1].reset_state()

class LSTM:
    def __init__(self, W_x, W_h, b):
        # W_x : D x 4H
        # W_h : H x 4H
        # b : H x 1
        self.params = [W_x, W_h, b]
        self.grads = [np.zeros_like(W_x), np.zeros_like(W_h), np.zeros_like(b)]
        self.cache = None
        
    def forward(self, x, h_prev, c_prev):
        W_x, W_h, b = self.params
        _, H = h_prev.shape

        affine_res = np.matmul(x, W_x) + np.matmul(h_prev, W_h) + b # N x H
        
        f = sigmoid(affine_res[:, :H])
        g = np.tanh(affine_res[:, H:2*H])
        i = sigmoid(affine_res[:, 2*H:3*H])
        o = sigmoid(affine_res[:, 3*H:])
        
        c_next = (c_prev * f) + (g * i)
        h_next = np.tanh(c_next) * o
        
        self.cache = (x, h_prev, c_prev, f, g, i, o , c_next, h_next)
        
        return h_next, c_next
        
    def backward(self, dh_next, dc_next):
        Wx, Wh, b = self.params
        x, h_prev, c_prev, f, g, i, o, c_next, h_next = self.cache

        dintermediate = (dc_next + (dh_next * o) * (1 - np.tanh(c_next)**2))
        dc_prev = dintermediate * f

        do = dh_next * np.tanh(c_next)
        di = dintermediate * g
        dg = dintermediate * i
        df = dintermediate * c_prev

        # d/dx * sigomoid(x) = sigmoid(x) * (1-sigmoid(x))
        # d/dx * tanh(x) = 1 - tanh(x)**2
        di *= i * (1 - i) # N x H
        df *= f * (1 - f) # N x H
        do *= o * (1 - o) # N x H
        dg *= (1 - g**2)  # N x H

        dA = np.hstack((df, dg, di, do)) # dA: N x 4H

        db = dA.sum(axis=0)
        dW_x = np.dot(x.T, dA)
        dW_h = np.dot(h_prev.T, dA)

        dx = np.dot(dA, Wx.T)
        dh_prev = np.dot(dA, Wh.T)

        self.grads[0][...] = dW_x
        self.grads[1][...] = dW_h
        self.grads[2][...] = db

        return dx, dh_prev, dc_prev
    
class TimeLSTM:
    def __init__(self, W_x, W_h, b, stateful=True):
        self.params = [W_x, W_h, b]
        self.grads = [np.zeros_like(W_x), np.zeros_like(W_h), np.zeros_like(b)]
        self.stateful = stateful
        
        self.h, self.dh = None, None
        self.c, self.dc = None, None
        self.layers = []
    
    def set_state(self, h, c):
        self.h = h
        self.c = c
        
    def reset_state(self):
        self.h = None
        self.c = None
        
    def forward(self, xs):
        W_x, W_h, b = self.params
        N, T, D = xs.shape
        H, _ = W_h.shape
        
        hs = []
        
        if not self.stateful or self.h is None:
            self.h = np.zeros((N, H), dtype='float')
        
        if not self.stateful or self.c is None:
            self.c = np.zeros((N, H), dtype='float')
            
            
        for i in range(T):
            lstm = LSTM(W_x, W_h, b)

            self.h, self.c = lstm.forward(xs[:, i, :], self.h, self.c)
            hs.append(self.h)
            
            #hs[:, i, :] = self.h
            self.layers.append(lstm)
        
        return np.array(hs).transpose(1, 0, 2)
    
    def backward(self, dhs):
        _, T, _ = dhs.shape
        
        dxs = []
        dh_next = 0
        dc_next = 0
        for i in range(len(self.grads)):
            self.grads[i][...] = 0
            
        for i in range(T-1, -1, -1):
            dh_cur = dhs[:, i, :]
            dx, dh_prev, dc_prev = self.layers[i].backward(dh_cur + dh_next, dc_next)
            dxs.append(dx)
            
            for j, grad in enumerate(self.layers[i].grads):
                self.grads[j] += grad
                
            dh_next = dh_prev
            dc_next = dc_prev
            
        self.dh = dh_prev
        self.dc = dc_prev
        return np.flip(np.array(dxs), axis=0)
            
class Rnnlm:
    def __init__(self, vocab_size, embedding_size, hidden_size):
        self.D = embedding_size
        self.H = hidden_size
        self.V = vocab_size
        self.params = []
        self.grads = []
        self.logit = None

        self.W_embed = (0.01 * np.random.randn(self.V, self.D)).astype('float32')
        self.W_x_lstm = ((1 / np.sqrt(self.D)) * np.random.randn(self.D, 4*self.H)).astype('float32')
        self.W_h_lstm = ((1 / np.sqrt(self.H)) * np.random.randn(self.H, 4*self.H)).astype('float32')
        self.b_lstm = np.zeros(4*self.H).astype('float32')
        self.W_affine = ((1 / np.sqrt(self.H)) * np.random.randn(self.H, self.V)).astype('float32')
        self.b_affine = np.zeros(self.V).astype('float32')

        self.layers = []
        self.layers.append(TimeEmbedding(self.W_embed))
        self.layers.append(TimeLSTM(self.W_x_lstm, self.W_h_lstm, self.b_lstm))
        self.layers.append(TimeAffine(self.W_affine, self.b_affine))
        self.last_layer = TimeSoftmaxWithLoss()

        for layer in self.layers:
            for param, grad in zip(layer.params, layer.grads):
                self.params.append(param)
                self.grads.append(grad)

    def predict(self, xs):
        out = xs
        for layer in self.layers:
            out = layer.forward(out)
        
        self.logit = out
        return self.logit
    
    def forward(self, xs, ts):
        logit = self.predict(xs)
        loss = self.last_layer.forward(logit, ts)
        return loss

    def backward(self):
        dout = self.last_layer.backward().astype('float32')
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def reset_state(self):
        self.layers[1].reset_state()
        
    def save_params(self, file_name):
        with open(file_name, 'wb') as f:
            pickle.dump(self.params, f)
    
    def load_params(self, file_name):
        with open(file_name, 'rb') as f:
            loaded = pickle.load(f)
        for i, param in enumerate(self.params):
            param[...] = loaded[i]
            
class TimeDropout:
    def __init__(self, dropout_ratio=0.5):
        self.params = []
        self.grads = []
        self.dropout_ratio = dropout_ratio
        self.mask = None
        self.train_flag = True

    def forward(self, xs):
        if self.train_flag:
            flag = np.random.rand(*xs.shape) > self.dropout_ratio
            scale = 1 / (1.0 - self.dropout_ratio)
            self.mask = flag.astype(np.float32) * scale

            return xs * self.mask
        else:
            return xs

    def backward(self, dout):
        return dout * self.mask
    
    
class BetterRnnlm:
    def __init__(self, vocab_size, embedding_size, hidden_size, dropout_ratio):
        self.D = embedding_size
        self.H = hidden_size
        self.V = vocab_size
        self.params = []
        self.grads = []
        self.logit = None

        self.W_embed = (0.01 * np.random.randn(self.V, self.D)).astype('float32')
        self.W_x_lstm1 = ((1 / np.sqrt(self.D)) * np.random.randn(self.D, 4*self.H)).astype('float32')
        self.W_h_lstm1 = ((1 / np.sqrt(self.H)) * np.random.randn(self.H, 4*self.H)).astype('float32')
        self.b_lstm1 = np.zeros(4*self.H).astype('float32')
        self.W_x_lstm2 = ((1 / np.sqrt(self.H)) * np.random.randn(self.H, 4*self.H)).astype('float32')
        self.W_h_lstm2 = ((1 / np.sqrt(self.H)) * np.random.randn(self.H, 4*self.H)).astype('float32')
        self.b_lstm2 = np.zeros(4*self.H).astype('float32')
        
        self.b_affine = np.zeros(self.V).astype('float32')

        self.layers = []
        self.layers.append(TimeEmbedding(self.W_embed))
        self.layers.append(TimeDropout(dropout_ratio))
        self.layers.append(TimeLSTM(self.W_x_lstm1, self.W_h_lstm1, self.b_lstm1))
        self.layers.append(TimeDropout(dropout_ratio))
        self.layers.append(TimeLSTM(self.W_x_lstm2, self.W_h_lstm2, self.b_lstm2))
        self.layers.append(TimeDropout(dropout_ratio))
        self.layers.append(TimeAffine(self.W_embed, self.b_affine))
        
        self.last_layer = TimeSoftmaxWithLoss()

        for layer in self.layers:
            for param, grad in zip(layer.params, layer.grads):
                self.params.append(param)
                self.grads.append(grad)

    def predict(self, xs, train_flag=False):
        self.layers[1].train_flag = train_flag
        self.layers[3].train_flag = train_flag
        self.layers[5].train_flag = train_flag
        
        out = xs
        for layer in self.layers:
            out = layer.forward(out)
        
        self.logit = out
        return self.logit
    
    def forward(self, xs, ts, train_flag=True):
        logit = self.predict(xs, train_flag)
        loss = self.last_layer.forward(logit, ts)
        return loss

    def backward(self):
        dout = self.last_layer.backward().astype('float32')
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def reset_state(self):
        self.layers[2].reset_state()
        self.layers[4].reset_state()
        
    def save_params(self, file_name):
        with open(file_name, 'wb') as f:
            pickle.dump(self.params, f)
    
    def load_params(self, file_name):
        with open(file_name, 'rb') as f:
            loaded = pickle.load(f)
        for i, param in enumerate(self.params):
            param[...] = loaded[i]
            
class RnnlmGen(Rnnlm):
    def __init__(self, vocab_size, embedding_size, hidden_size):
        super().__init__(vocab_size, embedding_size, hidden_size)

    def generate(self, start_id, skip_ids, sample_size):
        input = start_id
        word_ids = [start_id]

        while len(word_ids) < sample_size:
            input = np.array(input, dtype=np.int32).reshape(1, 1)
            logit = self.predict(input).reshape(-1)
            prob = softmax(logit)

            sample = np.random.choice(len(prob), size=1, p=prob)[0]
            if sample in skip_ids:
                continue
            word_ids.append(sample)
            input = sample

        return word_ids

class Encoder:
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        self.V = vocab_size
        self.D = wordvec_size
        self.H = hidden_size
        
        self.params = []
        self.grads = []
        
        self.W_embed = (0.01 * np.random.randn(self.V, self.D)).astype('float32')
        self.W_x_lstm = ((1 / np.sqrt(self.D)) * np.random.randn(self.D, 4*self.H)).astype('float32')
        self.W_h_lstm = ((1 / np.sqrt(self.H)) * np.random.randn(self.H, 4*self.H)).astype('float32')
        self.b_lstm = np.zeros(4*self.H).astype('float32')
        
        self.layers = []
        self.layers.append(TimeEmbedding(self.W_embed))
        self.layers.append(TimeLSTM(self.W_x_lstm, self.W_h_lstm, self.b_lstm, stateful=False))
        
        for layer in self.layers:
            for param, grad in zip(layer.params, layer.grads):
                self.params.append(param)
                self.grads.append(grad)
    
    
    def forward(self, xs):
        out = xs
        for layer in self.layers:
            out = layer.forward(out)
        
        self.hs = out
        return out[:, -1, :]
    
    def backward(self, dh):
        dhs = np.zeros_like(self.hs)
        dhs[:, -1, :] = dh
        dout = dhs
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout
    
class Decoder:
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        self.V = vocab_size
        self.D = wordvec_size
        self.H = hidden_size
        
        self.params = []
        self.grads = []
        
        self.W_embed = (0.01 * np.random.randn(self.V, self.D)).astype('float32')
        self.W_x_lstm = ((1 / np.sqrt(self.D)) * np.random.randn(self.D, 4*self.H)).astype('float32')
        self.W_h_lstm = ((1 / np.sqrt(self.H)) * np.random.randn(self.H, 4*self.H)).astype('float32')
        self.b_lstm = np.zeros(4*self.H).astype('float32')
        self.W_affine = ((1 / np.sqrt(self.H)) * np.random.randn(self.H, self.V)).astype('float32')
        self.b_affine = np.zeros(self.V).astype('float32')
        
        self.layers = []
        self.layers.append(TimeEmbedding(self.W_embed))
        self.layers.append(TimeLSTM(self.W_x_lstm, self.W_h_lstm, self.b_lstm))
        self.layers.append(TimeAffine(self.W_affine, self.b_affine))
        
        for layer in self.layers:
            for param, grad in zip(layer.params, layer.grads):
                self.params.append(param)
                self.grads.append(grad)
                
    def forward(self, xs, h):
        out = xs
        self.layers[1].h = h
        
        for layer in self.layers:
            out = layer.forward(out)
        
        self.hs = out
        return out
    
    def backward(self, dout):
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
            
        self.dh = self.layers[1].dh
        
        return dout
    
    def generate(self, h, start_id, sample_size):
        self.layers[1].h = h
        
        input = start_id
        word_ids = [start_id]

        while len(word_ids) < sample_size:
            out = np.array(input, dtype=np.int32).reshape(1, 1)
            for layer in self.layers:
                out = layer.forward(out)

            sample = int(np.argmax(out))

            word_ids.append(sample)
            input = sample

        return word_ids
    
class Seq2seq:
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        self.V = vocab_size
        self.D = wordvec_size
        self.H = hidden_size
        
        self.params = []
        self.grads = []
        
        self.encoder = Encoder(vocab_size, wordvec_size, hidden_size)
        self.decoder = Decoder(vocab_size, wordvec_size, hidden_size)
        self.softmax_with_loss = TimeSoftmaxWithLoss()
        
        for param, grad in zip(self.encoder.params, self.encoder.grads):
            self.params.append(param)
            self.grads.append(grad)
            
        for param, grad in zip(self.decoder.params, self.decoder.grads):
            self.params.append(param)
            self.grads.append(grad)

    def forward(self, xs, ts):
        decoder_xs, decoder_ts = ts[:, :-1], ts[:, 1:]
        h = self.encoder.forward(xs)
        logit = self.decoder.forward(decoder_xs, h)
        loss = self.softmax_with_loss.forward(logit, decoder_ts)
        
        return loss
    
    def backward(self, dout=1):
        dout = self.softmax_with_loss.backward()
        dh = self.decoder.backward(dout)
        dxs = self.encoder.backward(dh)
        
        return dxs
    
    def generate(self, xs, start_id, sample_size):
        h = self.encoder.forward(xs)
        word_ids = self.decoder.generate(h, start_id, sample_size)
        return word_ids