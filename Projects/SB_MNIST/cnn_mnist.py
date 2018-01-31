#Samuel Boulanger 1.27.18
#Beginning MNIST tutorial for Tensorflow

#
# Layer1:       Layer2: Layer3:       Layer4: Layer5:         Layer6: 
# Conv 32 Nodes MaxPool Conv 64 Nodes MaxPool Dense 1024 N.   Dense 10
# 5x5 filters   2x2 [2] 5x5 filter[2] 2x2 [2] Regularize (.4) Logits Layer
# ReLU ActFunc          ReLU           
# 
# [stride]
#
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

def cnn_model_fn(features, labels, mode):
    """Model function for CNN"""
    #input layer
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

    #convolutional layer #1
    conv1 = tf.layers.conv2d(inputs=input_layer, filters=32, kernel_size=[5,5],padding="same", activation=tf.nn.relu)
    #pooling layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2,2],strides=2)
    #convolutional layer #2
    conv2 = tf.layers.conv2d(inputs=pool1,filters=64,kernel_size=[5,5],padding="same",activation=tf.nn.relu)
    #pooling layer #2
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2,2], strides=2)
    #flatten pool output
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64]) # Height * Width * Channels
    #dense layer #1
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    #remove 40% of elements will randomly be dropped (only used in training mode)
    dropout = tf.layers.dropout(inputs=dense, rate=.4, training=mode == tf.estimator.ModeKeys.TRAIN)
    #logit layer
    logits = tf.layers.dense(inputs=dropout, units=10)
 
    predictions = { 
    #get the predictions
    "classes":tf.argmax(input=logits, axis=1),
    #add a sofmax
    "probablities":tf.nn.softmax(logits, name="softmax_tensor") 
    }
   
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    #calculate loss (TRAIN and EVAL)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # config training optimizer 
    if mode == tf.estimator.ModeKeys.TRAIN:
      optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
      train_op = optimizer.minimize(
          loss=loss,
          global_step=tf.train.get_global_step())
      return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # add evaluation metrics (EVAL)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
def main(unused_argv):
    #load training and eval data
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
    #create estimator object
    mnist_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir="./mnist_convnet_model")
    #set up logging for predictions
    tensors_to_log = {'probabilities':'softmax_tensor'}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)
    train_input_fn = tf.estimator.inputs.numpy_input_fn( x = { 'x' : train_data},
                                                         y = train_labels,
                                                         batch_size=100,
                                                         num_epochs=None,
                                                         shuffle=True)
    mnist_classifier.train(input_fn=train_input_fn,steps=20000,hooks=[logging_hook])

    #Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x":eval_data},
                                                                    y=eval_labels,
                                                                    num_epochs=1,
                                                                    shuffle=False)
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)



if __name__ == "__main__":
    tf.app.run()

