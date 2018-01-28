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
    conv2 = tf.layer.conv2d(inputs=pool1,filters=64,kernel_size=[5,5],padding="same",activation=tf.nn.relu)
    #pooling layer #2
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2,2], strides=2)
    #flatten pool output
    poll2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64]) # Height * Width * Channels
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


if __name__ == "__main__":
    tf.app.run()

