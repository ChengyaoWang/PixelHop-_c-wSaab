# v June 10 2020
# Feature Extraction Part for PixelHop++, Spark Version
# modified from https://github.com/USC-MCL/EE569_2020Spring


import os, time, random
import numpy as np
from numpy import linalg as LA
from pyspark import SparkContext, SparkConf
from skimage.measure import block_reduce
from skimage.util.shape import view_as_windows
# Source of Toy Dataset: Currently CIFAR10
import torch
import torchvision
import torchvision.transforms as transforms


'''
    This is the toy dataset for validating the pipeline of this system
    Uses CIFAR10 
'''
class ToyDataset(object):

    def __init__(self, TrainSubset = 1.0):
        self.TrainSubset = TrainSubset
        self.DATASET_SAVE_PTH = './CIFAR10/'
        self.MODEL_SAVE_PTH = './CIFAR10_Buffer/'

    def __str__(self):
        return  f"X_Train: {type(self.X_train)} {self.X_train.shape}\t{self.X_train.dtype}\n" \
                f"y_Train: {type(self.y_train)} {self.y_train.shape}\t\t{self.y_train.dtype}\n" \
                f"X_Test:  {type(self.X_test)} {self.X_test.shape}  \t{self.X_test.dtype}\n" \
                f"y_Test:  {type(self.y_test)} {self.y_test.shape}  \t\t{self.y_test.dtype}"

    def __len__(self):
        return self.X_train.shape[0]

    def LoadDataset(self):
        if os.path.exists(os.path.join(self.MODEL_SAVE_PTH, 'cifar10_train.npz')):
            train = np.load(os.path.join(self.MODEL_SAVE_PTH, 'cifar10_train.npz'))
        if os.path.exists(os.path.join(self.MODEL_SAVE_PTH, 'cifar10_test.npz')):
            test = np.load(os.path.join(self.MODEL_SAVE_PTH, 'cifar10_test.npz'))
        self.X_train = train['x_train']
        self.y_train = train['y_train']
        self.X_test = test['x_test']
        self.y_test = test['y_test']
        print('=' * 50 + '>DataSet Successfully Loaded')
        return self.X_train, self.y_train, self.X_test, self.y_test

    def FetchDataset(self):
        # Load the Dataset
        transform = transforms.Compose([transforms.Pad(0, fill = 0, padding_mode = 'constant'),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
        DatasetTrain = torchvision.datasets.CIFAR10(root = './CIFAR10/', train = True, download = True, transform = transform)
        DatasetTest = torchvision.datasets.CIFAR10(root = './CIFAR10/', train = False, download = True, transform = transform)
        # Subsample Dataset
        selectedIdx = random.sample(range(len(DatasetTrain)), int(len(DatasetTrain) * self.TrainSubset))
        DatasetTrain = torch.utils.data.Subset(DatasetTrain, selectedIdx)
        # Create torch.Dataset Loader
        TrainLoader = torch.utils.data.DataLoader(DatasetTrain, batch_size = len(DatasetTrain), shuffle = True)
        TestLoader = torch.utils.data.DataLoader(DatasetTest,   batch_size = len(DatasetTest), shuffle = True)
        # Transform to Numpy data
        for img, label in TrainLoader:
            self.X_train = img.numpy()
            self.y_train = label.numpy()
        for img, label in TestLoader:
            self.X_test = img.numpy()
            self.y_test = label.numpy()
        # Channel Last transform
        self.X_train = np.moveaxis(self.X_train, 1, 3)
        self.X_test = np.moveaxis(self.X_test, 1, 3)
        # Simple Anomaly Check
        assert(self.X_train.shape[0] == self.y_train.shape[0]), 'TrainX != Trainy'
        assert(self.X_test.shape[0] == self.y_test.shape[0]), 'TestX != Testy'
        # Save
        np.savez(f'{self.MODEL_SAVE_PTH}cifar10_train.npz', x_train = self.X_train, y_train = self.y_train)
        np.savez(f'{self.MODEL_SAVE_PTH}cifar10_test.npz',  x_test = self.X_test,   y_test = self.y_test)
        print('=' * 50 + '>DataSet Successfully Fetched')
        return self.X_train, self.y_train, self.X_test, self.y_test


def Saab_Train(pixelRDD,
               EnergyMultiplier: float = 1.0,
               sign_alignment: bool = True,
               use_DC: bool = True):
    pixelRDD = pixelRDD.map(neighborhood_reconstruction)
    originalDims = pixelRDD.map(lambda x: (x.shape, 1)) \
                           .reduceByKey(lambda x, y: x + y) \
                           .collect() 
    if len(originalDims) > 1:
        raise ValueError('The Images have different shapes, please check')
    originalDims = originalDims[0][0][:-1]
    pixelRDD = pixelRDD.map(lambda x: x.reshape(-1, x.shape[-1]))
    # DC kernel Subtraction
    dc = pixelRDD.map(lambda x: np.mean(x, axis = 1)) \
                 .reduce(lambda x, y: np.concatenate([x, y], axis = 0))
    pixelRDD = pixelRDD.map(lambda x: x - np.mean(x, axis = 1, keepdims = True))
    # Mean Subtraction
    mean0 = pixelRDD.map(lambda x: (1, (np.sum(x, axis = 0), x.shape[0]))) \
                    .reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1])) \
                    .map(lambda x: x[1][0] / float(x[1][1])).collect()
    mean0 = np.squeeze(mean0)
    pixelRDD = pixelRDD.map(lambda x: x - mean0)
    # Calculation of DC-kernel
    num_channels = mean0.shape[0]
    largest_eva = [np.var(dc) * num_channels]
    dc_kernel = 1 / np.sqrt(num_channels) * np.ones((1, num_channels)) / np.sqrt(largest_eva)
    # Covariance Calculation
    cov = pixelRDD.map(lambda x: x.transpose() @ x) \
                  .reduce(lambda x, y: x + y) / dc.shape[0]
    eva, eve = LA.eigh(cov)
    if sign_alignment:
        max_abs_cols = np.argmax(np.abs(eve), axis = 0)
        signs = np.sign(eve[max_abs_cols, range(eve.shape[1])])
        eve *= signs
    # Energy Ranking
    inds = eva.argsort()[::-1]
    eva, eve = eva[inds], eve.transpose()[inds]
    if use_DC:
        eve = np.concatenate((dc_kernel, eve), axis = 0)[:-1]
        eva = np.concatenate((largest_eva, eva), axis = 0)[:-1]
    # Normalization & Filtering
    eva *= (EnergyMultiplier / sum(eva))
    energy, kernel = eva, eve
    # Transform
    pixelRDD = pixelRDD.map(lambda x: x @ kernel.transpose()) \
                       .map(lambda x: x.reshape(originalDims[0], originalDims[1], -1))
    return kernel, energy, pixelRDD

def cwSaab_Train(pixelRDD,
                 EnergyMultiplier: np.ndarray,
                 sign_alignment: bool = True,
                 use_DC: bool = True):
    InputDims = pixelRDD.map(lambda x: (x.shape, 1)) \
                        .reduceByKey(lambda x, y: x + y) \
                        .collect()
    if len(InputDims) > 1:
        raise ValueError('The Images have different shapes, please check')
    num_channels = InputDims[0][0][-1]
    # cwSaab Starts
    kernels, energies, retRDD = [], [], None
    for i in range(num_channels):
        cwRDD = pixelRDD.map(lambda x: x[:, :, [i]])
        cwKernel, cwEnergy, cwRDD = Saab_Train(cwRDD, EnergyMultiplier[i])
        kernels.append(cwKernel)
        energies.append(cwEnergy)
        if retRDD is None:
            retRDD = cwRDD
        else:
            retRDD = retRDD.zip(cwRDD).map(lambda x: np.concatenate(x, axis = -1))

    energies = np.concatenate(energies, axis = 0)
    kernels = np.concatenate(kernels, axis = 0)
    return kernels, energies, retRDD


def cwSaab_Filter(pixelRDD,
                  kernels: np.ndarray,
                  energies: np.ndarray,
                  shrinkFunc = np.max,
                  TH1: float = 0.001,
                  TH2: float = 0.0001,
                  sort: bool = False):
    interIdx, = np.where(energies > TH1)
    outIdx, = np.where((energies < TH1) & (energies > TH2))

    interRDD = pixelRDD.map(lambda x: block_reduce(x, block_size = (2, 2, 1), func = shrinkFunc)) \
                       .map(lambda x: x[:, :, interIdx])
    interKernel = kernels[interIdx]
    interEnergy = energies[interIdx]

    outRDD = pixelRDD.map(lambda x: block_reduce(x, block_size = (2, 2, 1), func = shrinkFunc)) \
                     .map(lambda x: x[:, :, outIdx])
    outKernel = kernels[outIdx]
    outEnergy = energies[outIdx]
    
    print(f"Filter Result:\t" \
          f"Intermediate {interIdx.shape[0]}\t" \
          f"Output {outIdx.shape[0]}\t" \
          f"Discarded {energies.shape[0] - interIdx.shape[0] - outIdx.shape[0]}\t")

    return interRDD, interKernel, interEnergy, outRDD, outKernel, outEnergy

def collectOutput():
    pass


def neighborhood_reconstruction(X):
    ch = X.shape[-1]
    X = view_as_windows(X, (5, 5, ch), (1, 1, ch))
    X = X.reshape(X.shape[0], X.shape[1], -1)
    return X

if __name__ == '__main__':
    # import warnings
    # warnings.filterwarnings("ignore")

    '''This is the CIFAR10 Dataset'''
    dataset = ToyDataset(TrainSubset = 1.0)
    X_train, y_train, X_test, y_test = dataset.FetchDataset()
    # X_train, y_train, X_test, y_test = dataset.LoadDataset()
    print(dataset)
    
    time_start = time.time()
    config = SparkConf().setAll(
        [('spark.driver.memory', '12g'),
         ('spark.driver.maxResultSize', '10g')]).setAppName('cwSaab').setMaster('local[*]')
    sc = SparkContext(conf = config)


    # sc = SparkContext().setAppName('bbb').setMaster('local')
    sc.setLogLevel("ERROR")


    pixelRDD = sc.parallelize(X_train, numSlices = 32)
    Kernel_Dict, Output_Dict = {}, {}

    kernels, energies, pixelRDD = Saab_Train(pixelRDD)    

    for i in range(2):
        interR, interK, interE, outR, outK, outE = cwSaab_Filter(pixelRDD = pixelRDD, 
                                                                kernels = kernels,
                                                                energies = energies)
        if i == 1:
            break
        pixelRDD = interR
        energy = interE
        kernels, energies, pixelRDD = cwSaab_Train(pixelRDD, energy)
        # Do something
        # Kernel_Dict[str()] = outR
        # Output_Dict[str()] = outK

    sc.stop()