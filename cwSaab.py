# v June 10 2020
# c/w Saab transformation
# modified from https://github.com/USC-MCL/EE569_2020Spring

import numpy as np 
import pickle, gc, time
from saab import Saab

# Python Virtual Machine's Garbage Collector
def gc_invoker(func):
    def wrapper(*args, **kw):
        value = func(*args, **kw)
        gc.collect()
        time.sleep(0.5)
        return value
    return wrapper

class cwSaab():
    def __init__(self, depth = 1, energyTH = 0.01, SaabArgs = None, shrinkArgs = None, concatArg = None):
        self.par = {}
        assert (depth > 0), "'depth' must > 0!"
        self.depth = (int)(depth)
        self.energyTH = energyTH
        assert (SaabArgs != None), "Need parameter 'SaabArgs'!"
        self.SaabArgs = SaabArgs
        assert (shrinkArgs != None), "Need parameter 'shrinkArgs'!"
        self.shrinkArgs = shrinkArgs
        assert (concatArg != None), "Need parameter 'concatArg'!"
        self.concatArg = concatArg
        self.Energy = []
        self.trained = False
        self.split = False
        if depth > np.min([len(SaabArgs), len(shrinkArgs)]):
            self.depth = np.min([len(SaabArgs), len(shrinkArgs)])
            print("       <WARNING> Too few 'SaabArgs/shrinkArgs' to get depth %s, actual depth: %s"%(str(depth),str(self.depth)))

    @gc_invoker
    def SaabTransform(self, X, saab, train, layer):
        shrinkArg, SaabArg = self.shrinkArgs[layer], self.SaabArgs[layer]
        assert ('func' in shrinkArg.keys()), "shrinkArg must contain key 'func'!"
        X = shrinkArg['func'](X, shrinkArg)
        ''''''
        gc.collect()
        time.sleep(1)
        # print('Neighborhood Reconstruction')
        S = list(X.shape)
        X = X.reshape(-1, S[-1])
        if SaabArg['num_AC_kernels'] != -1:
            S[-1] = SaabArg['num_AC_kernels']
        if train == True:
            saab = Saab(num_kernels=SaabArg['num_AC_kernels'], useDC=SaabArg['useDC'], needBias=SaabArg['needBias'])
            saab.fit(X)
        transformed, dc = saab.transform(X)
        transformed = transformed.reshape(S)
        return saab, transformed, dc

    @gc_invoker
    def cwSaab_1_layer(self, X, train):
        if train == True:
            saab_cur = []
        else:
            saab_cur = self.par['Layer'+str(0)]
        transformed, eng, DC = [], [], []
        if self.SaabArgs[0]['cw'] == True:
            S = list(X.shape)
            S[-1] = 1
            X = np.moveaxis(X, -1, 0)
            for i in range(X.shape[0]):
                X_tmp = X[i].reshape(S)
                if train == True:
                    saab, tmp_transformed, dc = self.SaabTransform(X_tmp, saab=None, train=True, layer=0)
                    saab_cur.append(saab)
                    eng.append(saab.Energy)
                else:
                    if len(saab_cur) == i:
                        break
                    _, tmp_transformed, dc = self.SaabTransform(X_tmp, saab=saab_cur[i], train=False, layer=0)
                transformed.append(tmp_transformed)
                DC.append(dc)
            transformed = np.concatenate(transformed, axis=-1)
        else:
            if train == True:
                saab, transformed, dc = self.SaabTransform(X, saab=None, train=True, layer=0)
                saab_cur.append(saab)
                eng.append(saab.Energy)
            else:
                _, transformed, dc = self.SaabTransform(X, saab=saab_cur[0], train=False, layer=0)
            DC.append(dc)
                
        if train == True:
            self.par['Layer'+str(0)] = saab_cur
            self.Energy.append(np.concatenate(eng, axis=0))
        return transformed, DC

    @gc_invoker
    def cwSaab_n_layer(self, X, train, layer):
        output, eng_cur, DC, ct, pidx = [], [], [], -1, 0
        S = list(X.shape)
        S[-1] = 1
        X = np.moveaxis(X, -1, 0)
        saab_prev = self.par['Layer'+str(layer-1)]
        if train == True:
            saab_cur = []
        else:
            saab_cur = self.par['Layer'+str(layer)]
        for i in range(len(saab_prev)):
            for j in range(saab_prev[i].Energy.shape[0]):
                ct += 1
                if saab_prev[i].Energy[j] < self.energyTH:
                    continue
                self.split = True
                X_tmp = X[ct].reshape(S)
                if train == True:
                    saab, out_tmp, dc = self.SaabTransform(X_tmp, saab=None, train=True, layer=layer)
                    saab.Energy *= saab_prev[i].Energy[j]
                    saab_cur.append(saab)
                    eng_cur.append(saab.Energy) 
                else:
                    _, out_tmp, dc = self.SaabTransform(X_tmp, saab=saab_cur[pidx], train=False, layer=layer)
                    pidx += 1
                output.append(out_tmp)
                DC.append(dc)
                '''Clean the Cache'''
                saab = None
                X_tmp = None
                out_tmp = None
                dc = None
                gc.collect()
                ''''''
        if self.split == True:
            output = np.concatenate(output, axis=-1)
            if train == True:
                self.par['Layer'+str(layer)] = saab_cur
                self.Energy.append(np.concatenate(eng_cur, axis = 0))
        return output, DC
    
    def fit(self, X):
#        output, DC = [], []
        X, dc = self.cwSaab_1_layer(X, train=True)
#        output.append(X)
#        DC.append(dc)
        print('=' * 45 + '>c/w Saab Train Hop 0')
        for i in range(1, self.depth):
            X, dc = self.cwSaab_n_layer(X, train = True, layer = i)
#            output.append(X)
#            DC.append(dc)
            if self.split == False:
                self.depth = i
                print("       <WARNING> Cannot futher split, actual depth: %s"%str(i))
                break
            print('=' * 45 + f'>c/w Saab Train Hop {i}')
        self.trained = True
#        output = self.concatArg['func'](output, self.concatArg)
#        return output, DC
        return self

    def transform(self, X):
        assert (self.trained == True), "Must call fit first!"
        output, DC = [], []
        X, dc = self.cwSaab_1_layer(X, train = False)
        output.append(X)
        DC.append(dc)
        for i in range(1, self.depth):
            X, dc = self.cwSaab_n_layer(X, train=False, layer=i)
            output.append(X)
            DC.append(dc)
        assert ('func' in self.concatArg.keys()), "'concatArg' must have key 'func'!"
        output = self.concatArg['func'](output, self.concatArg)
        return output, DC
    
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
        return self
        
        
if __name__ == "__main__":
    # example useage
    import cv2
    from sklearn import datasets
    from skimage.util import view_as_windows
    # example callback function for collecting patches and its inverse
    def Shrink(X, shrinkArg):
        win = shrinkArg['win']
        stride = shrinkArg['stride']
        ch = X.shape[-1]
        X = view_as_windows(X, (1,win,win,ch), (1,stride,stride,ch))
        return X.reshape(X.shape[0], X.shape[1], X.shape[2], -1)

    # example callback function for how to concate features from different hops
    def Concat(X, concatArg):
        return X

    # read data
    print(" > This is a test example: ")
    digits = datasets.load_digits()
    X = digits.images.reshape((len(digits.images), 8, 8, 1))
    print(" input feature shape: %s"%str(X.shape))

    # set args
    SaabArgs = [{'num_AC_kernels':-1, 'needBias':False, 'useDC':True, 'batch':None, 'cw': False}, 
                {'num_AC_kernels':-1, 'needBias':True, 'useDC':True, 'batch':None, 'cw': True}]
    shrinkArgs = [{'func':Shrink, 'win':2, 'stride': 1}, 
                {'func': Shrink, 'win':2, 'stride': 1}]
    concatArg = {'func':Concat}

    print(" -----> depth=2")
    cwsaab = cwSaab(depth=2, energyTH=0.001, SaabArgs=SaabArgs, shrinkArgs=shrinkArgs, concatArg=concatArg)
    cwsaab.fit(X)

    
    '''Test for Save / Load'''
    cwsaab.save('./dummy')
    newSaab = cwSaab(SaabArgs=SaabArgs, shrinkArgs=shrinkArgs, concatArg=concatArg).load('./dummy')
    output1, DC1 = cwsaab.transform(X)
    output2, DC2 = newSaab.transform(X)
    print(type(output1), type(output2), type(DC1), type(DC2))
    print(len(output1), len(output2), len(DC1), len(DC2))
    for a, b, c, d in zip(output1, output2, DC1, DC2):
        print(a.shape, b.shape, np.array(a).shape, np.array(b).shape)
        if not np.array_equal(a, b) or not np.array_equal(np.array(a), np.array(b)):
            raise ValueError('Loading Method Error')

    print("------- DONE -------\n")
