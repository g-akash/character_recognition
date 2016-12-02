import tensorflow as tf

a = tf.ones((1,4))

b = tf.ones((1,4))

c = tf.reduce_sum(a*(a))

with tf.Session() as sess:
    print("here it is: ")
    print(c.eval())
