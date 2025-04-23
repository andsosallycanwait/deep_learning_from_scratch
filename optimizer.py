import numpy as np

class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        for i in range(len(params)):  # 리스트로 처리
            params[i] -= self.lr * grads[i]
            
class Momentum:
    def __init__(self, lr, momentum):
        self.lr = lr
        self.momentum = momentum
        self.v = None
        
    def update(self, params, grads):
        if self.v == None:
            self.v = {}
            for key, param in params.items():
                self.v[key] = np.zeros_like(param)
        
        for key in params.keys():
            self.v[key] = self.momentum*self.v[key] - self.lr*grads[key]
            params[key] = params[key] + self.v[key]
            
class AdaGrad:
    def __init__(self, lr):
        self.lr = lr
        self.h = None
        
    def update(self, params, grads):
        if self.h == None:
            self.h = {}
            for key, param in params.items():
                self.h[key] = np.zeros_like(param)
        
        for key in params.keys():
            self.h[key] = self.h[key] + grads[key]*grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)
            
class Adam:
    pass