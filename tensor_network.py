import tensorflow as tf
import numpy as np
import pandas as pd


def my_model(features, labels, mode, params):
    """

    :param features:
    :param labels:
    :param mode: PREDICT, EVAL, TRAIN
    :param params: Required paramters: hidden_units
    :return:
    """
    # Building network
    weight = []
    bias = []

    for m, n in zip(params["size"][1:], params["size"][:-1]):
        weight.append(tf.Variable(np.random.randn(m, n), dtype=tf.dtypes.float64, name="weight"))
        bias.append(tf.Variable(np.random.rand(m, 1), dtype=tf.dtypes.float64, name="bias"))

    # Compute
    a = features
    for weight, bias in zip(weight, bias):
        z = tf.matmul(weight, a) + bias
        a = tf.sigmoid(z)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'probabilities': a,
            'a': a
        }
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions
        )
    cost = -tf.reduce_sum(labels * tf.log(a) + (1 - labels) * tf.log(1 - a))  # Cross entropy loss function
    accuracy = tf.metrics.accuracy(labels=labels, predictions=a, name='acc_op')
    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy[1])

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode=mode, loss=cost, eval_metric_ops=metrics)

    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.05)
    train = optimizer.minimize(cost)
    return tf.estimator.EstimatorSpec(mode, loss=cost, train_op=train)


def input_fn(features, labels, batch_size):
    # TODO How to define features
    df = tf.data.Dataset.from_tensor_slices((tf.cast(features, tf.float64), tf.cast(labels, tf.float64)))

    return df.shuffle(100).repeat().batch(batch_size)


def main():
    df = pd.read_csv("./digit-recognizer/train.csv")

    x_train, y_train = df.iloc[:, 1:], df.iloc[:, 0]
    with tf.get_default_graph
    y_train = tf.one_hot(y_train, depth=10)
    classifier = tf.estimator.Estimator(model_dir="./model_dir",
                                        model_fn=my_model,
                                        params={
                                            "size": [784, 4, 10]
                                        })

    result = classifier.train(input_fn=lambda: input_fn(x_train, y_train, 1))


if __name__ == '__main__':
    main()
