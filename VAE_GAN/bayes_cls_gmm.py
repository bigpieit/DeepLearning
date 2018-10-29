from __future__ import print_function, division
from builtins import range

from VAE_GAN import utils
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import BayesianGaussianMixture

class BayesClassifier:
    def fit(self, X, Y):
        self.K = len(set(Y))
        self.gaussians = []

        for k in range(self.K):
            Xk = X[Y==k]
            # mean = Xk.mean(axis=0)
            # cov = np.cov(Xk.T)
            # print("Xk {}shape : ".format(k), Xk.size)
            # print("fit mean.size:", mean.size)
            # print("fit cov size :", cov.size)
            # g = {'m': mean , 'c': cov}
            gmm = BayesianGaussianMixture(10)
            gmm.fit(Xk)
            self.gaussians.append(gmm)

    def sample_given_y(self, y):
        gmm = self.gaussians[y]
        sample = gmm.sample()
        # print("mean.size:", g['m'].size)
        # print("cov size :", g['c'].size)z
        # print("")
        mean = gmm.means_[sample[1]]
        return sample[0].reshape(28, 28), mean.reshape(28, 28)

    def sample(self):
        y = np.random.randint(self.K)
        return self.sample_given_y(y)

if __name__ == "__main__":
    print("hello Generative Model!")
    X, Y = utils.get_mnist()
    clf = BayesClassifier()
    clf.fit(X,Y)

    for k in range(clf.K):
        sample, mean = clf.sample_given_y(k)

        plt.subplot(1,2,1)
        plt.imshow(sample, cmap='gray')
        plt.title("Sample")
        plt.subplot(1,2,2)
        plt.imshow(mean, cmap='gray')
        plt.title("Mean")
        plt.savefig("./gmm/"+str(k)+".png")

    sample, mean = clf.sample()
    plt.imshow(sample, cmap = 'gray')
    plt.title("Random Sample from Random Class")
    plt.savefig("./gmm/random.png")