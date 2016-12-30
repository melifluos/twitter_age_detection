"""
This is worth keeping as an example of how even for very simple models SGD can fail quite badly.
despite y being a deterministic function here with W=(2,5) and b=10, SGD tends to find a local minima at W~(0,0), b=13.5
This occurs because the data is drawn U(0,1) and so the average contribution from W.x is 2/2 + 5/2 = 3.5
"""

__author__ = 'benchamberlain'

import tensorflow as tf
import numpy as np

# Create 100 phony x, y data points in NumPy, y = x * 0.1 + 0.3
x_data = np.random.rand(1000, 2).astype(np.float32)
w_true = np.array([2.0, 5.0])
y_data = np.dot(x_data, w_true) + 10.0
print y_data.shape

# Here None means determine the number of rows at runtime. This allows us to vary the batch size
with tf.name_scope('input') as scope:
    x = tf.placeholder(tf.float32, [None, 2], 'x')
    y_ = tf.placeholder(tf.float32, [None], 'y_')
# Try to find values for W and b that compute y_data = W * x_data + b
# (We know that W = 0.1 and b = 0.3, but TensorFlow will
# figure that out for us.)
with tf.name_scope('model') as scope:
    W = tf.Variable(tf.random_uniform([2, 1], -10.0, 10.0))
    W_hist = tf.histogram_summary('W', W)
    b = tf.Variable(tf.zeros([1]))
    b_hist = tf.histogram_summary('b', b)
    y = tf.matmul(x, W) + b

with tf.name_scope('optimisation') as scope:
    # Minimize the mean squared errors.
    loss = tf.reduce_mean(tf.square(y - y_))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
    loss_sum = tf.scalar_summary('loss', loss)

merged = tf.merge_all_summaries()

# Before starting, initialize the variables.  We will 'run' this first.
init = tf.initialize_all_variables()

# Launch the graph.
sess = tf.Session()
sess.run(init)
writer = tf.train.SummaryWriter('./tf_logs/run4/', sess.graph)

# Fit the line.
for step in range(1000):
    idx = np.random.choice(len(y_data), 100)
    feed = {x: x_data[idx, :], y_: y_data[idx]}
    sess.run(train_step, feed_dict=feed)
    if step % 10 == 0:
        print(step, sess.run(W), sess.run(b))
        results = sess.run(merged, feed_dict=feed)
        writer.add_summary(results, step)

print """despite y being a deterministic function, SGD tends to find a local minima at W~(0,0), b=13.5
This occurs because the data is drawn U(0,1) and so the average contribution from W is 2/2 + 5/2 = 3.5
"""