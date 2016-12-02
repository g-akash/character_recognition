from __future__ import print_function

direc = "./train_small/"
valdirec = "./valid_small/"

import tensorflow as tf
import numpy
from skimage import io

# Parameters
learning_rate = 0.003
training_epochs = 500
batch_size = 100
display_step = 1

# Network Parameters
n_hidden_1 = 1024 # 1st layer number of features
n_hidden_2 = 256 # 2nd layer number of features
n_input = 6400 #80x80
n_classes = 104 # total classes (0-103 digits)

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

xhat = numpy.zeros([17205, n_input])
for i in range(17205):
    image = io.imread(direc + str(i) + ".png")
    image = numpy.reshape(image, n_input)
    image = image.astype(float)
    image=image/numpy.max(image)
    xhat[i] = image

xvhat = numpy.zeros([1829, n_input])
for i in range(1829):
    image = io.imread(valdirec + str(i) + ".png")
    image = numpy.reshape(image, n_input)
    image = image.astype(float)
    image=image/numpy.max(image)
    xvhat[i] = image

#caching y values for training data
yhat = numpy.zeros([17205])
with open(direc + "labels.txt") as fp:
    for k, line in enumerate(fp):
        yhat[k] = int(line)

#caching y values for validation data
yvhat = numpy.zeros([1829])
with open(valdirec + "labels.txt") as fp:
    for k, line in enumerate(fp):
        yvhat[k] = int(line)

#input function taking x and y
def input_next(i, batch_size):
    ret_x = numpy.zeros((batch_size, n_input))
    ret_y = numpy.zeros((batch_size, n_classes))
    for j in range(i*batch_size, (i+1)*batch_size):
        ret_x[j - i*batch_size] = xhat[j]
        ret_y[j - i*batch_size][int(yhat[j])] = 1               #y is cached in yhat
    return ret_x, ret_y.astype(numpy.float)

def input_valid(i, batch_size):
    ret_x = numpy.zeros((batch_size, n_input))
    ret_y = numpy.zeros((batch_size, n_classes))
    for j in range(i*batch_size, (i+1)*batch_size):
        ret_x[j - i*batch_size] = xvhat[j]
        ret_y[j - i*batch_size][int(yvhat[j])] = 1              #yvalid is cached in yvhat
    return ret_x, ret_y.astype(numpy.float)

# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with relu6 activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.tanh(layer_1)
    # Hidden layer with relu6 activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.tanh(layer_2)
    # Output layer with linear activation
    out_layer = tf.nn.softmax(tf.matmul(layer_2, weights['out']) + biases['out'])
    return out_layer

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1], stddev=0.01)),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], stddev=0.01)),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes], stddev=0.01))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1], stddev=0.01)),
    'b2': tf.Variable(tf.random_normal([n_hidden_2], stddev=0.01)),
    'out': tf.Variable(tf.random_normal([n_classes], stddev=0.01))
}

# Construct model
pred = multilayer_perceptron(x, weights, biases)

# Define loss and optimizer
#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred + 1e-6, y))
cost = -tf.reduce_sum(pred*tf.log(y+1e-10))
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)


# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(17205/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = input_next(i, batch_size)        # Reads the next batch_size input files
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", \
                "{:.9f}".format(avg_cost))
    print("Optimization Finished!")
    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    '''total_valid = int(1829/batch_size)
    accu = 0.0;
    for i in range(total_valid):
        valid_x, valid_y = input_valid(i, batch_size)
        accu += accuracy.eval({x: valid_x, y: valid_y})'''
    valid_x, valid_y = input_valid(0,1829)
    accu = accuracy.eval({x: valid_x, y: valid_y})
    print("Accuracy:", accu)
