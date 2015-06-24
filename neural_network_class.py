from __future__ import division

import numpy as np

sigmoid = np.vectorize(lambda x: 1.0/(1.0+np.exp(-x)))
sig = lambda x: 1.0/(1.0+np.exp(-x))
sigmoid_d = np.vectorize(lambda x: sig(x) * (1 - sig(x)))

tanh_d = lambda x: 1 - np.square(np.tanh(x))

def log_loss(y, yhat):
    return np.sum(-(y*np.log(yhat) + (1 - y)*np.log(1 - yhat)))

class neural_network:

    def __init__(self, sizes, activation = sigmoid, act_d = sigmoid_d):
        '''
        one required arguement: a list with the layer sizes
        can be used for classification or regression
        '''
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(x) for x in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
        self.a = [np.zeros(x) for x in sizes]
        self.activation = activation
        self.act_d = act_d
        self.z = [np.zeros(x) for x in sizes[1:]]


    def predict(self, x):
        '''
        input:
            - x, a 2D numpy array
        output:
            predicted values
        '''
        out = []
        for i in x:
            for w, b in zip(self.weights, self.biases):
                i = self.activation(np.dot(w, i)+b)

            out.append(i)

        return np.array(out)

    def forward_prop(self, x):
        '''
        input:
            - x, a single observation
        output:
            predictions for x

        This method can be used for prediction, but it should only be used
        as part of training the neural network. The main difference is that
        forward_prop() records the activation vectors(a) and output vectors(z)
        '''
        self.a[0] = x
        for i, w in enumerate(self.weights):
            b = self.biases[i]
            self.z[i] = np.dot(w,x) + b
            x = self.activation(self.z[i])
            self.a[i+1] = x


        return self.a[-1]


    def gradient_descent(self, x, y, lr):
        m = len(x)
        delta_weights = [np.zeros((self.sizes[l+1], self.sizes[l]))
                 for l in range(len(self.sizes[:-1]))]

        delta_biases = [np.zeros(l) for l in self.sizes[1:]]

        for i in range(m):
            w_grads, b_grads = self.back_prop(x[i], y[i])
            delta_weights = [dw + wg for dw, wg in zip(delta_weights, w_grads)]
            delta_biases = [db + bg for db, bg in zip(delta_biases, b_grads)]

        self.weights = [w - lr * wg/m for
                        wg, w in zip(delta_weights, self.weights)]
        self.biases = [b - lr * bg/m for bg,b in zip(delta_biases, self.biases)]


    def back_prop(self, x, y):
        '''
        computes the gradients for gradient_descent
        '''
        nabla_w = [None for l in self.weights]
        nabla_b = [None for l in self.weights]
        deltas = [None for l in self.sizes]
        yhat = self.forward_prop(x)
        deltas[-1] = yhat - y
        for l in range(1, self.num_layers - 1)[::-1]:
            w = self.weights[l]
            a = self.a[l]
            z = self.z[l-1]
            deltas[l] = np.dot(w.T, deltas[l+1]) * self.act_d(z)
        for l in range(0, self.num_layers-1):
            nabla_w[l] =  np.outer(deltas[l+1], self.a[l].T)
            nabla_b[l] = deltas[l+1]
        return nabla_w, nabla_b


    def gradient_checking(self, x, y, epsilon):
        '''
        function to check the implementation of back_prop()
        '''
        original_weights = [l*1 for l in self.weights]
        grad_approx = [np.zeros((j, i))
                       for i, j in zip(self.sizes[:-1], self.sizes[1:])]
        w_grads, b_grads = self.back_prop(x, y)
        for l in xrange(self.num_layers - 1):
            lshape = self.weights[l].shape
            indicies = [(r, c) for c in xrange(lshape[1])
                       for r in xrange(lshape[0])]
            for i in indicies:
                self.weights[l][i] += epsilon
                theta_plus = log_loss(y, self.predict([x])[0])
                self.weights[l][i] -= 2*epsilon
                theta_minus = log_loss(y, self.predict([x])[0])
                grad_approx[l][i] = (theta_plus - theta_minus)/(2*epsilon)
                self.weights[l][i] += epsilon

        b_grad_approx = [np.zeros(l) for l in self.sizes[1:]]
        original_biases = [b*1 for b in self.biases]
        for l in xrange(self.num_layers - 1):
            b_shape = len(self.biases[l])
            for i in xrange(b_shape):
                self.biases[l][i] += epsilon
                theta_plus = log_loss(y, self.predict([x])[0])
                self.biases[l][i] -= 2*epsilon
                theta_minus = log_loss(y, self.predict([x])[0])
                b_grad_approx[l][i] = (theta_plus - theta_minus)/(2*epsilon)
                self.biases[l][i] += epsilon
        # print some information for gradient checking
        for l in xrange(self.num_layers - 1):
            print l
            print 'weights'
            print grad_approx[l]
            print w_grads[l]
            print 'difference'
            print grad_approx[l] - w_grads[l]
            print '----------------'
            print 'biases'
            print b_grad_approx[l]
            print b_grads[l]
            print 'difference'
            print b_grad_approx[l] - b_grads[l]
            print '_________________________________'
