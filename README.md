# Memory & Time Optimized c/w Saab

### **Warning**
The code is still under construction

### Introduction
This is implemented by Chengyao Wang

**OOP / Numpy** version is an modified version of Yifan Wang's [EE569_2020Spring](https://github.com/USC-MCL/EE569_2020Spring)

**FP / Pyspark** version is an modified version of Min Zhang's [PointHop++](https://github.com/minzhang-1/PointHop-PointHop2_Spark)

Original Paper [PixelHop++: A Small Successive-Subspace-Learning-Based (SSL-based) Model for Image Classification](https://arxiv.org/abs/2002.03141)

Note that this is not the official implementation.

### Highlights

  - Substituting SVD by Eigen-decomposition
  - Manual invocation of Python Vitual Machine's garbage collector
  - Spark Framework
  - Save / Load options using * *pickle* *


### Major Dependencies
Numpy, Scikit-Learn, pyspark, pytorch / tensorflow (for CIFAR10 dataset)
