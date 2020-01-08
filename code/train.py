import os
import json
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

dirname = os.path.dirname(__file__)
DATASET = input_data.read_data_sets(os.path.join(dirname, "../data/"), one_hot=True)
OUT = os.path.join(dirname, "../models/mnist")

batch_size = 128
num_steps = 2600
learning_rate = 0.05
start = time.time()

def vega_loss(loss_data):
    data = {
        "$schema": "https://vega.github.io/schema/vega-lite/v4.json",
        "width": 300,
        "height": 200,
        "description": "Loss",
        "data":  {
            "values": []
        },
        "mark": "line",
        "encoding": {
            "x": {"field": "step", "type": "quantitative"},
            "y": {"field": "loss", "type": "quantitative"}
        }
    }

    for value in loss_data:
        data['data']['values'].append({ "step": value[0], "loss": str(value[1]) })

    return data

def vega_accuracy(accu_data):
    data = {
        "$schema": "https://vega.github.io/schema/vega-lite/v4.json",
        "width": 300,
        "height": 200,
        "description": "Accuracy",
        "data":  {
            "values": []
        },
        "mark": "line",
        "encoding": {
            "x": {"field": "step", "type": "quantitative"},
            "y": {"field": "acc", "type": "quantitative"}
        }
    }

    for value in accu_data:
        data['data']['values'].append({ "step": value[0], "acc": str(value[1] * 100) })

    return data

# input
x = tf.placeholder(tf.float32, [None, 784], "x")
y_ = tf.placeholder(tf.float32, [None, 10], "y")

# weight
W = tf.Variable(tf.zeros([784, 10]))
# bias
b = tf.Variable(tf.zeros([10]))
# test_data * W + b
y = tf.matmul(x, W) + b
sm = tf.nn.softmax(y, name="softmax")

# cross entropy (loss function)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_), name="loss")

# train step
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

# evaluating the model
correct_prediction = tf.equal(tf.argmax(sm, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")

# init
saver = tf.train.Saver()
init = tf.global_variables_initializer()

loss_metric = []
acc_metric = []
with tf.Session() as session:
    session.run(init)

    # training
    for step in range(num_steps):
        batch_data, batch_labels = DATASET.train.next_batch(batch_size)
        feed_dict = {x: batch_data, y_: batch_labels}

        loss_out, ts_out, acc_out = session.run([loss, train_step, accuracy], feed_dict=feed_dict)
        loss_metric.append((step, loss_out))
        acc_metric.append((step, acc_out))

    save_path = saver.save(session, OUT)

    with open(os.path.join(dirname, '../metrics/train.json'), 'w') as outfile:
        json.dump({ "batch_size": batch_size, "num_steps": num_steps, "learning_rate": learning_rate,
            "took" : (time.time() - start) / 1000 }, outfile)

    with open(os.path.join(dirname, '../metrics/vega_acc.json'), 'w') as outfile:
        json.dump(vega_accuracy(acc_metric), outfile)

    with open(os.path.join(dirname, '../metrics/vega_loss.json'), 'w') as outfile:
        json.dump(vega_loss(loss_metric), outfile)


