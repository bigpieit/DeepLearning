from __future__ import print_function, division
from builtins import range

from VAE_GAN import utils
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn


class BayesClassifier:
    def fit(self, X, Y):
        self.K = len(set(Y))
        self.gaussians = []

        for k in range(self.K):
            Xk = X[Y==k]
            mean = Xk.mean(axis=0)
            cov = np.cov(Xk.T)
            print("Xk {}shape : ".format(k), Xk.size)
            print("fit mean.size:", mean.size)
            print("fit cov size :", cov.size)
            g = {'m': mean , 'c': cov}
            self.gaussians.append(g)

    def  sample_given_y(self, y):
        g = self.gaussians[y]
        print("mean.size:", g['m'].size)
        print("cov size :", g['c'].size)

        print("")
        return mvn.rvs(mean = g['m'], cov = g['c'])

    def sample(self):
        y = np.random.randint(self.K)
        return self.sample_given_y(y)

if __name__ == "__main__":
    print("hello Generative Model!")
    X, Y = utils.get_mnist()
    clf = BayesClassifier()
    clf.fit(X,Y)

    for k in range(clf.K):
        sample = clf.sample_given_y(k).reshape(28,28)
        mean = clf.gaussians[k]['m'].reshape(28, 28)

        plt.subplot(1,2,1)
        plt.imshow(sample, cmap='gray')
        plt.title("Sample")
        plt.subplot(1,2,2)
        plt.imshow(mean, cmap='gray')
        plt.title("Mean")
        plt.savefig("./gauss/"+str(k)+".png")

    sample = clf.sample().reshape(28, 28)
    plt.imshow(sample, cmap = 'gray')
    plt.title("Random Sample from Random Class")