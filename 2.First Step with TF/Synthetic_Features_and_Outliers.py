'''
学习目标：

    1.创建一个合成特征，它是两个其他特征的比率混合
    2.将此新特征用作线性回归模型的输入
    3.通过识别并剪切（移除）输入数据中的异常值，提高模型的有效性
'''

import math
from IPython import display
from matplotlib import gridspec,cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics as metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows=10
pd.options.display.float_format = '{:.1f}'.format

california_housing_dataframe = pd.read_csv('../data/california_housing_train.csv', sep=',')

california_housing_dataframe = california_housing_dataframe.reindex(
    np.random.permutation(california_housing_dataframe.index))
california_housing_dataframe['median_house_value'] /= 1000.0
print(california_housing_dataframe)