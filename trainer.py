from model import Model
from chainer import optimizers, Variable
import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split

from tqdm import tqdm # library to show progress bar

class Trainer(object):
    def __init__(self, n_epoch=50, batch_size=16):
        self.model=Model()
        self.n_epoch = n_epoch
        self.batch_size = batch_size
    def train(self):
        print('preparing the dataset')
        x_train, x_test, y_train, y_test = self.prepare_data()
        print('Done')
        optimizer = optimizers.Adam()
        optimizer.setup(self.model)
        
        for epoch in range(self.n_epoch):
            print('Epoch %d'%(epoch))
            idxs = np.random.permutation(len(x_train))
            accuracies = []
            losses = []
            for i in tqdm(range(0, len(x_train), self.batch_size)):
                batch_acc = []
                batch_loss = []
                x = Variable(x_train[idxs[i : i + self.batch_size]])
                y = Variable(y_train[idxs[i : i + self.batch_size]])
                optimizer.zero_grads() # to initialize the gradients
                loss, accuracy = self.model(x, y)
                loss.backward()
                optimizer.update()
                losses.append(loss.data)
                accuracies.append(accuracy.data)
            print('Accuracy: %.4f, Loss: %.4f'%(np.mean(np.array(accuracies)), np.mean(np.array(losses))))

            test_accuracy = []
            print('Testing for Epoch: %d'%(epoch))
            for i in tqdm(range(0, len(x_test), self.batch_size)):
                x = x_test[i : i + self.batch_size]
                y = y_test[i : i + self.batch_size]
                acc = self.model(x, y, train=False, test=True)
                test_accuracy.append(acc.data)
            print('Test Accuracy: %.4f'%(np.mean(np.array(test_accuracy))))
            

    def prepare_data(self):
        mnist_data = fetch_mldata('MNIST original')
        x_whole = mnist_data.data
        x_whole = x_whole.astype(np.float32) / np.max(x_whole)
        x_whole = np.reshape(x_whole, (x_whole.shape[0], 1, int(np.sqrt(x_whole.shape[1])), int(np.sqrt(x_whole.shape[1]))))
        y_whole = mnist_data.target
        y_whole = y_whole.astype(np.int32)
        return train_test_split(x_whole, y_whole, test_size=0.2)
        
        
if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()
