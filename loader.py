from __future__ import print_function
import sys



direc = "./train_small/"
valdirec = "./valid_small/"

import tensorflow as tf
import numpy
from skimage import io, filters
# import Image
from skimage import transform
from scipy import misc



#buf = str(sys.argv[1])

#print "buf is ", buf

#buf = "image.png"

# Parameters
learning_rate = 0.001
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

 #subject to change.

'''image = io.imread(buf)
image = numpy.invert(image)
image = filters.gaussian(image, 5)
image = transform.resize(image, (80, 80))
image = numpy.reshape(image, n_input)
image = image.astype(float)
image=image/numpy.max(image)

im = numpy.zeros([1,n_input])
im[0] = image'''

'''xhat = numpy.zeros([17205, n_input])
for i in range(17205):
    image = io.imread(direc + str(i) + ".png")
    image = numpy.reshape(image, n_input)
    image = image.astype(float)
    image=image/numpy.max(image)
    xhat[i] = image'''

xvhat = numpy.zeros([1829, n_input])
for i in range(1829):
    image = io.imread(valdirec + str(i) + ".png")
    image = numpy.reshape(image, n_input)
    image = image.astype(float)
    image=image/numpy.max(image)
    xvhat[i] = image

#caching y values for training data
'''yhat = numpy.zeros([17205])
with open(direc + "labels.txt") as fp:
    for k, line in enumerate(fp):
        yhat[k] = int(line)'''

#caching y values for validation data
yvhat = numpy.zeros([1829])
with open(valdirec + "labels.txt") as fp:
    for k, line in enumerate(fp):
        yvhat[k] = int(line)

#input function taking x and y
'''def input_next(i, batch_size):
    ret_x = numpy.zeros((batch_size, n_input))
    ret_y = numpy.zeros((batch_size, n_classes))
    for j in range(i*batch_size, (i+1)*batch_size):
        ret_x[j - i*batch_size] = xhat[j]
        ret_y[j - i*batch_size][int(yhat[j])] = 1               #y is cached in yhat
    return ret_x, ret_y.astype(numpy.float)'''

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
    layer_1 = tf.nn.sigmoid(layer_1)
    # Hidden layer with relu6 activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.sigmoid(layer_2)
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
cost = -tf.reduce_sum(pred*tf.log(y+1e-10)) + 0.001*( tf.nn.l2_loss(weights['h1']) + tf.nn.l2_loss(weights['h2']) + tf.nn.l2_loss(weights['out'])+   tf.nn.l2_loss(biases['b1']) + tf.nn.l2_loss(biases['b2']) + tf.nn.l2_loss(biases['out']))
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

saver = tf.train.Saver()

# Launch the graph
with tf.Session() as sess:

    saver.restore(sess, "./model.ckpt")
    print("Model restored.")
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
    #print("Accuracy:", accu)
