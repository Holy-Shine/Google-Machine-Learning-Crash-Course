
import math
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows=10
pd.options.display.float_format = '{:.1f}'.format

# 指示TF如何对数据进行预处理，以及如何在模型训练期间进行批处理，洗牌和重复。
def my_input_fn(features, targets, batch_size=1, shuffle=True,num_epochs=None):
    '''Trains a linear regression model of one feature.
	
    Args:
      features: pandas DataFrame of features
      targets: pandas DataFrame of targets
      batch_size: Size of batches to be passed to the model
      shuffle: True or False. Whether to shuffle the data.
      num_epochs: Number of epochs for which data should be repeated. None = repeat indefinitely
    Returns:
      Tuple of (features, labels) for next data batch
    '''
	# 将pandas转换为numpy矩阵
	features = {key:np.array(value) for key,value in dict(features).items()}
	# 构建数据集，配置batch和epoch
	ds = Dataset.from_tensor_slices((features,targets))  # warning: 2GB limit
	ds = ds.batch(batch_size).repeat(num_epochs)
	
	# shuffle数据，如果明确声明了
	if shuffle:
		ds = ds.shuffle(buffer_size=10000)
	
	# 返回下一batch的数据
	features, labels = ds.make_one_shot_iterator().get_next()
	return features, labels

if __name__=='__main__':
	# 使用pandas的dataframe加载数据
	california_housing_dataframe = pd.read_csv('california_housing_train.csv',sep=',')

	# shuffle和缩放数据
	california_housing_dataframe = california_housing_dataframe.reindex(
		np.random.permutation(california_housing_dataframe.index))
	california_housing_dataframe['median_house_value'] /= 1000.0

	#print(california_housing_dataframe)
	#california_housing_dataframe.describe()

	'''
	构建TF-LinearRegressor模型
	'''
	# 【step1】定义特征和特征列(这里使用total_rooms)
	my_feature = california_housing_dataframe[['total_rooms']]  # 内部是个列表表明特征可以不止一维
	feature_columns = [tf.feature_column.numeric_column('total_rooms')]
	# 【step2】定义目标
	targets = california_housing_dataframe['median_house_value']
	# 【step3】配置线性回归器
	my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0000001)
	my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)

	linear_regressor = tf.estimator.LinearRegressor(
		feature_columns = feature_columns,
		optimizer = my_optimizer
	)
	# 【step4】定义一个输入函数，指示TF如何对数据进行预处理，以及如何在模型训练期间进行批处理，洗牌和重复。
	# define my_input_fn
	# 【step5】训练模型
	_ = linear_regressor.train(
		input_fn=lambda:my_input_fn(my_feature, targets),
		steps=100
	)
	# 【step6】评估模型
	prediction_input_fn = lambda:my_input_fn(my_feature,targets,num_epochs=1,shuffle=False)
	predictions = linear_regressor.predict(input_fn=prediction_input_fn)
	
	predictions = np.array([item['predictions'][0] for item in predictions])
	mean_squared_error = metrics.mean_squared_error(predictions, targets)
	root_mean_squared_error = math.sqrt(mean_squared_error)
	print("MSE on training data: %0.3f" % mean_squared_error)
	print("RMSE on training data: %0.3f" % root_mean_squared_error)