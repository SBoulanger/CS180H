#Samuel Boulanger 3.14.18

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

MNIST_IMAGE_SIZE = 784
MNIST_HEIGHT = 28
MNIST_WIDTH  = 28

NUM_OUTPUT_CLASSES = 10

#will be playing with these hyperparameters
learning_rate = 0.0001
epochs = 10
batch_size = 50

tblogs_path = './logs/3'


def run_MNIST_network():
    #declare the training data placeholders
    #Input x is 28x28 = 748
    x = tf.placeholder(tf.float32,[None,MNIST_IMAGE_SIZE])
    #reshape x into [i,j,k,l]
    #i = number of training examples
    #j = height, k = width, i = depth
    # -1 is a placeholder that will be filled dynamically by tensorflow
    x_shaped = tf.reshape(x,[-1, MNIST_HEIGHT,MNIST_WIDTH, 1])
    #'truth' output placeholder
    y = tf.placeholder(tf.float32, [None,NUM_OUTPUT_CLASSES])#confused on the None

    #create conv layers
    layer1 = create_new_conv_layer(input_data=x_shaped, num_input_channels=1,
                                  num_filters=32,filter_shape=[5,5],pool_shape=[2,2],name='layer1') 
    layer2 = create_new_conv_layer(input_data=layer1, num_input_channels=32,
                                  num_filters=64,filter_shape=[5,5],pool_shape=[2,2],name='layer2') 

    # 7 * 7 because we use 2 maxpooling filters with 2x2 kernels and stride x:2,y:2
    # therefore it halfs the output size twice: 28x28->14x14->7x7
    # * 64 because there are 64 filters
    # -1 will be dynamically loaded depending on the batch size
    flattened = tf.reshape(layer2, [-1, 7 * 7 * 64])

    #Just some classic ML xw+b
    #not using sigmoid here though
    wd1 = tf.Variable(tf.truncated_normal([7 * 7 * 64, 1000], stddev=0.03), name='wd1')
    bd1 = tf.Variable(tf.truncated_normal([1000], stddev=0.01), name='bd1')
    dense_layer1 = tf.matmul(flattened, wd1) + bd1
    dense_layer1 = tf.nn.relu(dense_layer1)

    # another layer with softmax activations
    wd2 = tf.Variable(tf.truncated_normal([1000, 10], stddev=0.03), name='wd2')
    bd2 = tf.Variable(tf.truncated_normal([10], stddev=0.01), name='bd2')
    dense_layer2 = tf.matmul(dense_layer1, wd2) + bd2
    y_ = tf.nn.softmax(dense_layer2)

    #cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_, labels=labels_))
    cost = tf.reduce_mean(tf.square(y_ - y))/2

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    #define accuracy assessment operation
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    #setup the initialisation operator
    init_op = tf.global_variables_initializer()
    
    tf.summary.scalar('accuracy',accuracy)

    #create Tensorboard SummaryWriter
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tblogs_path)

    with tf.Session() as sess:
        sess.run(init_op)
        total_batch = int(len(mnist.train.labels) / batch_size)
        for epoch in range(epochs):
            avg_cost = 0
            for i in range(total_batch):
                batch_x, batch_y = mnist.train.next_batch(batch_size=batch_size)
                _, c = sess.run([optimizer, cost],
                                feed_dict={x: batch_x, y: batch_y})
                avg_cost += c/total_batch
            test_acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y:mnist.test.labels})
            print("Epoch:", (epoch +1), "cost = ", "{:.3f}".format(avg_cost), "test accuracy: {: .3f}".format(test_acc))
            summary = sess.run(merged, feed_dict={x: mnist.test.images,y:mnist.test.labels})
            writer.add_summary(summary, epoch)
        print("\nTraining complete!")
        writer.add_graph(sess.graph)
        print(sess.run(accuracy, feed_dict={x: mnist.test.images, y:mnist.test.labels}))


def create_new_conv_layer(input_data, num_input_channels, num_filters,
                          filter_shape, pool_shape, name):
    conv_filt_shape = [filter_shape[0],filter_shape[1], num_input_channels,
                       num_filters]
    
    weights = tf.Variable(tf.truncated_normal(conv_filt_shape, stddev=0.03),
                            name = name + "_W")
    bias = tf.Variable(tf.truncated_normal([num_filters]), name=name+'_b')

    # setup the convolutional layer operation
    out_layer = tf.nn.conv2d(input_data, weights, [1, 1, 1, 1], padding='SAME')

    # add the bias
    out_layer += bias

    # apply a ReLU non-linear activation
    out_layer = tf.nn.relu(out_layer)

    # now perform max pooling
    ksize     = [1, pool_shape[0], pool_shape[1], 1]
    strides   = [1, 2, 2, 1]
    out_layer = tf.nn.max_pool(out_layer, ksize=ksize, strides=strides, 
                               padding='SAME')

    return out_layer

if __name__ == "__main__":
    run_MNIST_network()
