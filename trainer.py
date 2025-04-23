import numpy as np
import matplotlib.pyplot as plt
from util import clip_grads

class Trainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.ppl_list = []

    def fit(self, xs, ts, max_epoch=10, batch_size=20, time_size=35, max_grad=None, eval_interval=20):
        data_size = len(xs)
        max_iters = data_size // (batch_size * time_size)
        time_idx = 0
        total_loss = 0
        loss_count = 0

        jump = (data_size - 1) // batch_size
        offsets = [i * jump for i in range(batch_size)]

        for epoch in range(max_epoch):
            for iter in range(max_iters):
                # 미니배치 생성
                batch_x = np.empty((batch_size, time_size), dtype='i')
                batch_t = np.empty((batch_size, time_size), dtype='i')

                for t in range(time_size):
                    for i, offset in enumerate(offsets):
                        idx = (offset + time_idx) % data_size
                        batch_x[i, t] = xs[idx]
                        batch_t[i, t] = ts[idx]
                    time_idx += 1

                # forward / backward / update
                loss = self.model.forward(batch_x, batch_t)
                self.model.backward()

                params, grads = self.model.params, self.model.grads
                if max_grad is not None:
                    clip_grads(grads, max_grad)

                self.optimizer.update(params, grads)

                total_loss += loss
                loss_count += 1

                if (iter + 1) % eval_interval == 0:
                    ppl = np.exp(total_loss / loss_count)
                    print(f'| epoch {epoch + 1} | iter {iter + 1}/{max_iters} | perplexity {ppl:.2f}')
                    self.ppl_list.append(ppl)
                    total_loss, loss_count = 0, 0

    def plot(self, title='Perplexity over Time'):
        x = np.arange(len(self.ppl_list))
        plt.plot(x, self.ppl_list, label='train')
        plt.xlabel('Evaluation step')
        plt.ylabel('Perplexity')
        plt.title(title)
        plt.grid(True)
        plt.legend()
        plt.show()

class RnnlmTrainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.time_idx = None
        self.ppl_list = []

    def fit(self, xs, ts, max_epoch=10, batch_size=20, time_size=35, max_grad=None, eval_interval=20):
        data_size = len(xs)
        max_iters = data_size // (batch_size * time_size)
        time_idx = 0
        total_loss = 0
        loss_count = 0

        jump = (data_size - 1) // batch_size
        offsets = [i * jump for i in range(batch_size)]

        for epoch in range(max_epoch):
            for iter in range(max_iters):
                # 미니배치 구성
                batch_x = np.empty((batch_size, time_size), dtype='i')
                batch_t = np.empty((batch_size, time_size), dtype='i')

                for t in range(time_size):
                    for i, offset in enumerate(offsets):
                        idx = (offset + time_idx) % data_size
                        batch_x[i, t] = xs[idx]
                        batch_t[i, t] = ts[idx]
                    time_idx += 1

                # forward / backward
                loss = self.model.forward(batch_x, batch_t)
                self.model.backward()
                params, grads = self.model.params, self.model.grads

                # gradient clipping
                if max_grad is not None:
                    clip_grads(grads, max_grad)

                # optimizer update
                self.optimizer.update(params, grads)

                total_loss += loss
                loss_count += 1

                # eval_interval마다 perplexity 측정
                if (iter + 1) % eval_interval == 0:
                    ppl = np.exp(total_loss / loss_count)
                    print(f"| epoch {epoch+1} | iter {iter+1}/{max_iters} | perplexity {ppl:.2f}")
                    self.ppl_list.append(ppl)
                    total_loss, loss_count = 0, 0

    def plot(self, title="Training Perplexity"):
        x = np.arange(len(self.ppl_list))
        plt.plot(x, self.ppl_list, label='train')
        plt.xlabel("Evaluation step")
        plt.ylabel("Perplexity")
        plt.title(title)
        plt.grid(True)
        plt.legend()
        plt.show()
