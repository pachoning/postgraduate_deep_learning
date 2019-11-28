import tensorflow as tf
import dataGenerator as ex01


def main():
    # TODO: E02: Define Linear Regressor graph --> Definition phase (use graph, placeholder, variable, operations)
    graph = tf.Graph()
    with graph.as_default():
        # Constants
        learning_rate = tf.constant(value=0.0001, name='learning_rate')

        # Input data
        x = tf.placeholder(tf.float32, shape = [], name = 'x')
        y = tf.placeholder(tf.float32, shape =[], name = 'y')

        # Parameters
        W = tf.get_variable('W', shape=[], dtype=tf.float32, initializer=tf.initializers.random_normal())
        b = tf.get_variable('b', shape=[], dtype=tf.float32, initializer=tf.initializers.random_normal())

        # Predictions
        z_pred = W*x
        y_pred = z_pred + b
        loss = (y_pred - y)**2

        # Gradients
        loss_grad = 2 * (y_pred - y)
        W_grad = x * loss_grad
        b_grad = loss_grad

        # Update parameters
        W_update = W.assign(W - learning_rate * W_grad)
        b_update = b.assign(b - learning_rate * b_grad)
        x_grad = W * loss_grad
        train_op = tf.group(W_update, b_update)


    # TODO: E03: Run a forward pass --> Run phase (use session and the DataDistribution class from previous exercise)
    sess = tf.Session(graph = graph)
    with sess:
        sess.run(tf.global_variables_initializer())
        dataGen = ex01.DataDistribution()
        for input_data, label in  dataGen.generate(50):
            prediction , loss_val, _ = sess.run([y_pred, loss, train_op], feed_dict={x: input_data, y: label})
            print("Input {} --> Prediction {}, label {} and loss {}".format(input_data, prediction, label, loss_val))


    # TODO: E04: Implement optimization step manually!
    pass


if __name__ == '__main__':
    main()


