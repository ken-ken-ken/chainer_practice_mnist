"""
practice space for chaienr
"""
import chainer
import chainer.links as L
import chainer.functions as F
class Model(chainer.Chain):
    def __init__(self):
        super(Model, self).__init__(
            conv1 = L.Convolution2D(1, 32, 5, stride=1, pad=2),
            conv2 = L.Convolution2D(32, 64, 5, stride=1, pad=2),
            fc1 = L.Linear(7 * 7 * 64, 1024),
            fc2 = L.Linear(1024, 10)
        )

    def __call__(self, x, y, train=True):
        h = F.relu(self.conv1(x))
        h = F.max_pooling_2d(h, 2, stride=2)
        h = F.relu(self.conv2(h))
        h = F.max_pooling_2d(h, 2, stride=2)
        h = self.fc1(h)
        h = F.dropout(h, train=train)
        h = self.fc2(h)

        if train:
            self.loss = F.softmax_cross_entropy(h, y)
            self.accuracy = F.accuracy(h,y)
            return self.loss, self.accuracy

        self.pred = F.softmax(h)
        return self.pred

