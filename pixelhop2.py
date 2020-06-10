# v June 10 2020
# Feature Extraction Part for PixelHop++
# modified from https://github.com/USC-MCL/EE569_2020Spring

import numpy as np 
from cwSaab import cwSaab
import pickle

class Pixelhop2(cwSaab):
    def __init__(self, depth=1, TH1=0.005, TH2=0.001, SaabArgs=None, shrinkArgs=None, concatArg=None):
        super().__init__(depth=depth, energyTH=TH1, SaabArgs=SaabArgs, shrinkArgs=shrinkArgs, concatArg={'func':lambda X, concatArg: X})
        self.TH1 = TH1
        self.TH2 = TH2
        self.idx = []        
        self.concatArg = concatArg

    def select_(self, X):
        #print('select discarded nodes')
        for i in range(self.depth):
            #print('depth {}: shape before = {}'.format(i,X[i].shape))
            X[i] = X[i][:, :, :, self.Energy[i] >= self.TH2]
            #print('depth {}: shape after = {}'.format(i,X[i].shape))
        return X

    def select2_(self, X):
        #print('select discarded nodes')
        for i in range(self.depth):
            #print('depth {}: shape before = {}'.format(i,X[i].shape))
            if(i == self.depth - 1): # if last layer use only TH2
                X[i] = X[i][:, :, :, self.Energy[i] >= self.TH2]
            else:
                tmp_discarded = self.Energy[i] >= self.TH2
                tmp_intermd = self.Energy[i] < self.TH1
                X[i] = X[i][:, :, :, tmp_discarded & tmp_intermd]
            #print('depth {}: shape after = {}'.format(i,X[i].shape))
        return X

    def fit(self, X):
        super().fit(X)
        #X = self.select_(X)
        #return self.concatArg['func'](X, self.concatArg)
        return self

    def transform(self, X):
        # print('pixelhop2 transform')
        X, _ = super().transform(X)
        X = self.select2_(X)
        return self.concatArg['func'](X, self.concatArg)


    '''Methods for Saving & Loading'''
    def save(self, filename: str):
        assert (self.trained == True), "Need to Train First"
        par = {}
        par['kernel'] = self.par
        par['depth'] = self.depth
        par['energyTH'] = self.energyTH
        par['energy'] = self.Energy
        par['SaabArgs'] = self.SaabArgs
        par['shrinkArgs'] = self.shrinkArgs
        par['concatArgs'] = self.concatArg
        par['concatArg_pixel2'] = self.concatArg
        par['TH1'] = self.TH1
        par['TH2'] = self.TH2

        with open(filename + '.pkl','wb') as f:
            pickle.dump(par, f)
        return

    def load(self, filename: str):
        par = pickle.load(open(filename + '.pkl','rb'))
        self.par = par['kernel']
        self.depth = par['depth']
        self.energyTH = par['energyTH']
        self.Energy = par['energy']
        self.SaabArgs = par['SaabArgs']
        self.shrinkArgs = par['shrinkArgs']
        self.concatArg = par['concatArgs']
        self.trained = True

        self.concatArg = par['concatArg_pixel2']
        self.TH1 = par['TH1']
        self.TH2 = par['TH2']
        
        return self


if __name__ == "__main__":
    # example useage
    from sklearn import datasets
    from skimage.util import view_as_windows

    # example callback function for collecting patches and its inverse
    def Shrink(X, shrinkArg):
        win = shrinkArg['win']
        X = view_as_windows(X, (1, win, win, 1), (1, win, win, 1))
        return X.reshape(X.shape[0], X.shape[1], X.shape[2], -1)

    # example callback function for how to concate features from different hops
    def Concat(X, concatArg):
        return X

    # read data
    import cv2
    print(" > This is a test example: ")
    digits = datasets.load_digits()
    X = digits.images.reshape((len(digits.images), 8, 8, 1))
    print(" input feature shape: %s"%str(X.shape))

    # set args
    SaabArgs = [{'num_AC_kernels':-1, 'needBias':False, 'useDC':True, 'batch':None, 'cw':False}, 
                {'num_AC_kernels':-1, 'needBias':True, 'useDC':True, 'batch':None, 'cw':True}]
    shrinkArgs = [{'func':Shrink, 'win':2}, 
                {'func': Shrink, 'win':2}]
    concatArg = {'func':Concat}

    print(" --> test inv")
    print(" -----> depth=1")
    p2 = Pixelhop2(depth=1, TH1=0.005, TH2=0.001, SaabArgs=SaabArgs, shrinkArgs=shrinkArgs, concatArg=concatArg)
    p2.fit(X)
    output = p2.transform(X)
    print(" -----> depth=2")
    p2 = Pixelhop2(depth=2, TH1=0.005, TH2=0.001, SaabArgs=SaabArgs, shrinkArgs=shrinkArgs, concatArg=concatArg)
    p2.fit(X)

    '''Test for Save / Load'''
    p2.save('./dummy')
    p2_new = Pixelhop2(SaabArgs=SaabArgs, shrinkArgs=shrinkArgs, concatArg=concatArg).load('./dummy')
    output1, DC1 = p2.transform(X)
    output2, DC2 = p2_new.transform(X)
    print(type(output1), type(output2), type(DC1), type(DC2))
    if not np.array_equal(np.array(output1), np.array(output2)) or not np.array_equal(np.array(DC1), np.array(DC2)):
        raise ValueError('Loading Method Error')

    print("------- DONE -------\n")




