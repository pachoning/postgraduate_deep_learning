import tensorflow as tf
import dataGenerator as ex01
import os


def main():
    graph = tf.Graph()
    with graph.as_default():
        # Constants
        learning_rate = tf.constant(value=1e-6, name='learning_rate')

        # Input data
        with tf.variable_scope("input_data"):
            x = tf.placeholder(tf.float32, shape = [], name = 'x')
            y = tf.placeholder(tf.float32, shape =[], name = 'y')

        # Parameters
        with tf.variable_scope("parameters"):
            W = tf.get_variable('W', shape=[], dtype=tf.float32, initializer=tf.initializers.random_normal())
            b = tf.get_variable('b', shape=[], dtype=tf.float32, initializer=tf.initializers.random_normal())

        # Predictions
        with tf.variable_scope("predictors"):
            z_pred = W*x
            y_pred = z_pred + b
            loss = (y_pred - y)**2

        # Gradients
        with tf.variable_scope("gradients"):
            loss_grad = 2 * (y_pred - y)
            W_grad = x * loss_grad
            b_grad = loss_grad

        # Update parameters
        with tf.variable_scope("update_parameters"):
            W_update = W.assign(W - learning_rate * W_grad)
            b_update = b.assign(b - learning_rate * b_grad)
            x_grad = W * loss_grad
            train_op = tf.group(W_update, b_update)


    sess = tf.Session(graph = graph)
    with sess:
        sess.run(tf.global_variables_initializer())
        dataGen = ex01.DataDistribution()
        for input_data, label in  dataGen.generate(500):
            prediction , loss_val, _ = sess.run([y_pred, loss, train_op], feed_dict={x: input_data, y: label})
            print("Input {} --> Prediction {}, label {} and loss {}".format(input_data, prediction, label, loss_val))


    writer = tf.summary.FileWriter(os.path.expanduser(os.getcwd()), graph=graph)
    pass


if __name__ == '__main__':
    main()


