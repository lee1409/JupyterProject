import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler


def architecture(features, params):
    with tf.name_scope("input"):
        a = features
        a = tf.transpose(a)
    for hidden_layer in params["size"]:
        with tf.name_scope("layer"):
            weight = tf.Variable(tf.random_normal(shape=(hidden_layer, a.get_shape()[0].value)), name="weight")
            bias = tf.Variable(tf.zeros(shape=(hidden_layer, 1)), name="bias")
            a = tf.matmul(weight, a) + bias
    return a


def modal_fn(features, labels, mode, params):
    logits = architecture(features, params)

    class_idx = tf.arg_max(logits)
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode,
            predictions={
                "probability": tf.nn.softmax(logits),
                "logits": logits
            }
        )
    labels = tf.transpose(labels)
    diff = logits - labels
    loss = tf.math.reduce_mean(tf.square(diff)) / 2
    tf.print("Loss: {}".format(loss))
    metric = tf.metrics.accuracy(tf.argmax(labels), predictions=logits)
    tf.summary.scalar('accuracy', metric[1])
    summary = tf.summary.scalar("loss", loss)
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode,
            eval_metric_ops={
                'cost': summary
            }
        )

    assert mode == tf.estimator.ModeKeys.TRAIN
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001)
    train = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(
        mode,
        loss=loss,
        train_op=train,
    )


def input_fn(features, labels, batch_size):
    # Do for transformation into tensor

    labels = tf.one_hot(labels, 10)

    tensor = tf.data.Dataset.from_tensor_slices(
        (tf.cast(features, tf.float32),
         tf.cast(labels, tf.float32))
    )

    return tensor.shuffle(42).repeat().batch(batch_size)


def main():
    train = pd.read_csv("digit-recognizer/train.csv")
    train_x, train_y = train.iloc[:, 1:], train.iloc[:, 0]
    scalar = StandardScaler()
    train_x = scalar.fit_transform(train_x)

    config = tf.estimator.RunConfig(
        model_dir="modal_dir",
        save_summary_steps=10,
        log_step_count_steps=10,
        save_checkpoints_steps=50,
    )

    estimator = tf.estimator.Estimator(
        model_fn=modal_fn,
        config=config,
        params={
            "size": [10, 20, 10]
        }
    )
    result = estimator.train(
        input_fn=lambda: input_fn(train_x, train_y, 200),
    )


if __name__ == '__main__':
    main()
