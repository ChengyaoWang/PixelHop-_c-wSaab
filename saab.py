# v June 10 2020
# Saab transformation
# modified from https://github.com/USC-MCL/EE569_2020Spring

import numpy as np
# import numba

# @numba.jit(forceobj = True, parallel = True)
def pca_cal(X: np.ndarray):
    cov = X.transpose() @ X
    eva, eve = np.linalg.eigh(cov)
    # Sign Alignment of Eigenvectors
    max_abs_cols = np.argmax(np.abs(eve), axis = 0)
    signs = np.sign(eve[max_abs_cols, range(eve.shape[1])])
    eve *= signs
    # Resorting Kernels
    inds = eva.argsort()[::-1]
    eva, kernels = eva[inds], eve.transpose()[inds]
    return kernels, eva / (X.shape[0] - 1)

# @numba.jit(forceobj = True, parallel = True)
def remove_mean(X: np.ndarray, feature_mean: np.ndarray):
    return X - feature_mean

# @numba.jit(nopython = True, parallel = True)
def feat_transform(X: np.ndarray, kernel: np.ndarray):
    return X @ kernel.transpose()


class Saab():
    def __init__(self, num_kernels=-1, useDC=True, needBias=True):
        self.par = None
        self.Kernels = []
        self.Bias = []
        self.Mean0 = []
        self.Energy = []
        self.num_kernels = num_kernels
        self.useDC = useDC
        self.needBias = needBias
        self.trained = False

    def remove_mean(self, X: np.ndarray, axis: int):
        feature_mean = np.mean(X, axis = axis, keepdims = True)
        X = X - feature_mean
        # return X, feature_mean
        return X

    # @numba.jit(forceobj = True)
    def fit(self, X): 
        assert (len(X.shape) == 2), "Input must be a 2D array!"
        X = X.astype('float32')
        self.Mean0 = np.mean(X, axis = 0, keepdims = True)
        # X = remove_mean(X, self.Mean0)
        X -= self.Mean0
        dc = np.mean(X, axis = 1, keepdims = True)
        # X = remove_mean(X, dc)
        X -= dc

        self.Bias = np.max(np.linalg.norm(X, axis = 1)) * 1 / np.sqrt(X.shape[1])
        if self.num_kernels == -1:
            self.num_kernels = X.shape[-1]
        
        '''Rewritten PCA Using Numpy'''
        kernels, eva = pca_cal(X)
        energy = eva / np.sum(eva)

        # pca = IncrementalPCA(n_components = self.num_kernels).fit(X)
        # pca = PCA(n_components = self.num_kernels, svd_solver='auto').fit(X)

        # kernels = pca.components_
        # energy = pca.explained_variance_ / np.sum(pca.explained_variance_)

        if self.useDC == True:  
            largest_ev = np.var(dc * np.sqrt(X.shape[-1]))  
            dc_kernel = 1 / np.sqrt(X.shape[-1]) * np.ones((1, X.shape[-1])) / np.sqrt(largest_ev)
            kernels = np.concatenate((dc_kernel, kernels[:-1]), axis = 0)
            energy = np.concatenate((np.array([largest_ev]), eva[:-1]), axis = 0)
            # energy = np.concatenate((np.array([largest_ev]), pca.explained_variance_[:-1]), axis = 0)
            energy = energy / np.sum(energy)
        self.Kernels, self.Energy = kernels.astype('float32'), energy
        self.trained = True

    # @numba.jit(forceobj = True)
    def transform(self, X):
        assert (self.trained == True), "Must call fit first!"
        X = X.astype('float32')
        X -= self.Mean0
        if self.needBias == True:
            X += self.Bias
        dc = np.mean(X, axis = 1, keepdims = True)
        # X = feat_transform(X, self.Kernels)
        X = X @ self.Kernels.transpose()
        if self.needBias == True and self.useDC == True:
            X[:, 0] -= self.Bias
        return X, dc
    
    def inverse_transform(self, X, DC):
        assert (self.trained == True), "Must call fit first!"
        assert (DC.shape[0] == X.shape[0]), "Input shape not match! 'X' and 'DC'"
        X = X.astype('float32')
        DC = DC.astype('float32')
        if self.needBias == True and self.useDC == True:
            X[:, 0] += self.Bias
        X = np.dot(X, self.Kernels)
        if self.needBias == True:
            X -= self.Bias 
        #X += DC
        X += self.Mean0
        return X

if __name__ == "__main__":

    import time 
    start = time.time()


    from sklearn import datasets
    print(" > This is a test example: ")
    digits = datasets.load_digits()
    data = digits.images.reshape((len(digits.images), 8, 8, 1))
    print(" input feature shape: %s"%str(data.shape))
        
    print(" --> test inv")
    print(" -----> num_kernels=-1, needBias=False, useDC=True")
    X = data.copy()
    X = X.reshape(X.shape[0], -1)[0:100]
    saab = Saab(num_kernels=-1, useDC=True, needBias=True)
    saab.fit(X)
    Xt, dc = saab.transform(X)
    Y = saab.inverse_transform(Xt, dc)
    print(np.mean(np.abs(X-Y)))
    assert (np.mean(np.abs(X-Y)) < 1e-5), "invSaab error!"
    print(" -----> num_kernels=-1, needBias=True, useDC=True")
    X = data.copy()
    X = X.reshape(X.shape[0], -1)[0:100]
    saab = Saab(num_kernels=-1, useDC=True, needBias=True)
    saab.fit(X)
    Xt, dc = saab.transform(X)
    Y = saab.inverse_transform(Xt, dc)
    assert (np.mean(np.abs(X-Y)) < 1e-5), "invSaab error!"
    print(" -----> num_kernels=-1, needBias=False, useDC=False")
    X = data.copy()
    X = X.reshape(X.shape[0], -1)[0:100]
    saab = Saab(num_kernels=-1, useDC=False, needBias=False)
    saab.fit(X)
    Xt, dc = saab.transform(X)
    Y = saab.inverse_transform(Xt, dc)
    assert (np.mean(np.abs(X-Y)) < 1e-5), "invSaab error!"
    print(" -----> num_kernels=-1, needBias=True, useDC=False")
    X = data.copy()
    X = X.reshape(X.shape[0], -1)[0:100]
    saab = Saab(num_kernels=-1, useDC=False, needBias=True)
    saab.fit(X)
    Xt, dc = saab.transform(X)
    Y = saab.inverse_transform(Xt, dc)
    assert (np.mean(np.abs(X-Y)) < 1e-5), "invSaab error!"

