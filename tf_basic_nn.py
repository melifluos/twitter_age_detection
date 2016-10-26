from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from utils import *

__author__ = 'benchamberlain'

# Data sets
IRIS_TRAINING = "local_resources/iris_training.csv"
IRIS_TEST = "local_resources/iris_test.csv"

# Load datasets.
training_set = tf.contrib.learn.datasets.base.load_csv(filename=IRIS_TRAINING, target_column=-1,
                                                       target_dtype=np.int)
test_set = tf.contrib.learn.datasets.base.load_csv(filename=IRIS_TEST, target_column=-1,
                                                   target_dtype=np.int)

x_path = 'resources/X.p'
y_path = 'resources/y.p'
X = read_pickle(x_path)
targets = read_pickle(y_path)
y = np.array(targets['cat'])
print(X.shape)
print(y.shape)
# filter the sparse matrix
X, _ = remove_sparse_features(X, threshold=1)
# tensor flow can't accept scipy sparse matrices as input
features = np.array(X.todense()).astype(np.float64)

# images_placeholder = tf.placeholder(tf.float32, shape=(batch_size,
#                                                        mnist.IMAGE_PIXELS))
# labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))

# Specify that all features have real-value data
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=features.shape[1])]

# Build 3 layer DNN with 10, 20, 10 units respectively.
classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[10, 20, 10],
                                            n_classes=10,
                                            model_dir="/tmp/age2")

# Fit model.
# classifier.fit(x=training_set.data,tensorflow.python.framework.errors.InternalError: Unsupported feed type
#                y=training_set.target,
#                steps=2000)

classifier.fit(x=features,
               y=y,
               steps=2000)

# Evaluate accuracy.
# accuracy_score = classifier.evaluate(x=test_set.data,
#                                      y=test_set.target)["accuracy"]

accuracy_score = classifier.evaluate(x=features,
                                     y=y)["accuracy"]

print('Accuracy: {0:f}'.format(accuracy_score))

# Classify two new flower samples.
# new_samples = np.array(
#     [[6.4, 3.2, 4.5, 1.5], [5.8, 3.1, 5.0, 1.7]], dtype=float)
# y = classifier.predict(new_samples)
# print('Predictions: {}'.format(str(y)))
