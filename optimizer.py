import numpy as np

class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        if type(params) is list:
            for i in range(len(params)):
                params[i] -= self.lr * grads[i]
        if type(params) is dict:
            for key in params.keys():
                params[key] -= self.lr * grads[key]
            
class Momentum:
    def __init__(self, lr, momentum):
        self.lr = lr
        self.momentum = momentum
        self.v = None
        
    def update(self, params, grads):
        if self.v is None:
            if type(params) is list:
                self.v = [np.zeros_like(param) for param in params]
            elif type(params) is dict:
                self.v = {key: np.zeros_like(param) for key, param in params.items()}
        
        if type(params) is list:
            for i in range(len(params)):
                self.v[i] = self.momentum * self.v[i] - self.lr * grads[i]
                params[i] += self.v[i]
        elif type(params) is dict:
            for key in params.keys():
                self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
                params[key] += self.v[key]
            
class AdaGrad:
    def __init__(self, lr):
        self.lr = lr
        self.h = None
        
    def update(self, params, grads):
        if self.h is None:
            if type(params) is list:
                self.h = [np.zeros_like(param) for param in params]
            elif type(params) is dict:
                self.h = {key: np.zeros_like(param) for key, param in params.items()}
        
        if type(params) is list:
            for i in range(len(params)):
                self.h[i] += grads[i] * grads[i]
                params[i] -= self.lr * grads[i] / (np.sqrt(self.h[i]) + 1e-7)
        elif type(params) is dict:
            for key in params.keys():
                self.h[key] += grads[key] * grads[key]
                params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)
            
class Adam:
    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.m = None
        self.v = None
        self.t = 0
        
    def update(self, params, grads):
        if self.m is None:
            if type(params) is list:
                self.m = [np.zeros_like(param) for param in params]
                self.v = [np.zeros_like(param) for param in params]
            elif type(params) is dict:
                self.m = {key: np.zeros_like(param) for key, param in params.items()}
                self.v = {key: np.zeros_like(param) for key, param in params.items()}
        
        self.t += 1
        
        if type(params) is list:
            for i in range(len(params)):
                self.m[i] = self.beta1*self.m[i] + (1 - self.beta1)*grads[i]
                self.v[i] = self.beta2*self.v[i] + (1 - self.beta2)*(grads[i] ** 2)

                m_hat = self.m[i] / (1 - self.beta1**self.t)
                v_hat = self.v[i] / (1 - self.beta2**self.t)
                
                params[i] -= self.lr * (m_hat / (np.sqrt(v_hat) + 1e-7))

        elif type(params) is dict:
            for key in params.keys():
                self.m[key] = self.beta1*self.m[key] + (1 - self.beta1)*grads[key]
                self.v[key] = self.beta2*self.v[key] + (1 - self.beta2)*(grads[key] ** 2)

                m_hat = self.m[key] / (1 - self.beta1**self.t)
                v_hat = self.v[key] / (1 - self.beta2**self.t)
                
                params[key] -= self.lr * (m_hat / (np.sqrt(v_hat) + 1e-7))