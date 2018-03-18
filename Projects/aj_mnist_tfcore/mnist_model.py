import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

# Hyper parameters
learning_rate = .0001
epochs = 10
batch_size = 50

# NOTE: what is None here? Perhaps similar to the -1 for x
# 28x28 = 784
x = tf.placeholder(tf.float32, [None, 784])

# -1 for variable batches, 28w, 28h, 1 color channel
x_shaped = tf.reshape(x, [-1, 28, 28, 1])

y = tf.placeholder(tf.float32, [None, 10])


def create_new_conv_layer(input_data, num_input_channels, num_filters, filter_shape, pool_shape, name):
    # 4th dimension is the number of feature maps
    conv_filt_shape = [filter_shape[0], filter_shape[1],
                       num_input_channels, num_filters]

    # set up the weights for every single filter pixel
    weights = tf.Variable(tf.truncated_normal(
        conv_filt_shape, stddev=0.03), name=name + '_W')

    # add a bias to each filter
    bias = tf.Variable(tf.truncated_normal([num_filters]), name=name + '_b')

    out_layer = tf.nn.conv2d(input_data, weights, [1, 1, 1, 1], padding='SAME')

    # WHAT DOES THIS DO MATHEMATICALLY???? 4D TENSOR + BIAS??? DOES NOT COMPUTE
    out_layer += bias

    out_layer = tf.nn.relu(out_layer)

    # max pool, for 1 input, wxh, 1 color channel
    ksize = [1, pool_shape[0], pool_shape[1], 1]
    # stride by 1 input, two squares, 1 channel
    strides = [1, 2, 2, 1]
    out_layer = tf.nn.max_pool(
        out_layer, ksize=ksize, strides=strides, padding='SAME')

    return out_layer


layer1 = create_new_conv_layer(x_shaped, 1, 32, [5, 5], [2, 2], name='layer1')
layer2 = create_new_conv_layer(layer1, 32, 64, [5, 5], [2, 2], name='layer2')
flattened = tf.reshape(layer2, [-1, 7 * 7 * 64])

# input: 7*7*64 vector, output 1000 vector
wd1 = tf.Variable(tf.truncated_normal(
    [7 * 7 * 64, 1000], stddev=.03), name='wd1')
bd1 = tf.Variable(tf.truncated_normal([1000], stddev=.01), name='bd1')
dense_layer1 = tf.matmul(flattened, wd1) + bd1
dense_layer1 = tf.nn.relu(dense_layer1)

wd2 = tf.Variable(tf.truncated_normal([1000, 10], stddev=.03), name='wd2')
bd2 = tf.Variable(tf.truncated_normal([10], stddev=.01), name='bd2')
dense_layer2 = tf.matmul(dense_layer1, wd2) + bd2
y_ = tf.nn.softmax(dense_layer2)

# get the softmax of each output, then reduce it to a single mean
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits=dense_layer2, labels=y))

optimizer = tf.train.AdamOptimizer(
    learning_rate=learning_rate).minimize(cross_entropy)

# boolean to say whether the prediction y_ was equal to the true y
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
# let accuracy be the average number of correct predictions
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    total_batch = int(len(mnist.train.labels) / batch_size)
    for epoch in range(epochs):
        avg_cost = 0
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size=batch_size)
            # cost for this batch is calculated via CE
            _, cost = sess.run([optimizer, cross_entropy],
                               feed_dict={x: batch_x, y: batch_y})
            avg_cost += cost / total_batch
        # after each epoch, see the accuracy on the test set
        test_acc = sess.run(accuracy, feed_dict={
                            x: mnist.test.images, y: mnist.test.labels})
        print("Epoch:", epoch + 1,
              "cost =", "{:.3f}".format(avg_cost),
              "test accuracy: ", "{:.3f}".format(test_acc))

    print("\nTraining Complete!")
    # print the final accuracy
    print(sess.run(accuracy, feed_dict={
          x: mnist.test.images, y: mnist.test.labels}))
