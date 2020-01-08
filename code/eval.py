import os
import json
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

dirname = os.path.dirname(__file__)
DATASET = input_data.read_data_sets(os.path.join(dirname, '../data'), one_hot=True)
META = os.path.join(dirname, '../models/mnist.meta')
MODELS = os.path.join(dirname, '../models/')

def vega_heatmap(matrix):
    data = {
        "$schema": "https://vega.github.io/schema/vega-lite/v4.json",
        "description": "Confusion Matrix",
        "data":  {
            "values": []
        },
         "mark": "rect",
        "encoding": {
            "y": {"field": "y", "type": "ordinal"},
            "x": {"field": "x", "type": "ordinal"},
            "color": {"field": "color", "type": "quantitative"}
        }
    }

    for idx,row in enumerate(matrix):
        for idy,column in enumerate(row):
            data['data']['values'].append({ "x": idy, "y": idx, "color": str(column) })

    return data


init = tf.global_variables_initializer()
with tf.Session() as sess:
    saver = tf.train.import_meta_graph(META)
    saver.restore(sess, tf.train.latest_checkpoint(MODELS))

    graph = tf.get_default_graph()

    x = graph.get_tensor_by_name("x:0")
    y = graph.get_tensor_by_name("y:0")
    softmax = graph.get_tensor_by_name("softmax:0")
    accuracy = graph.get_tensor_by_name("accuracy:0")
    feed_dict = { x: DATASET.test.images, y: DATASET.test.labels }

    pred = sess.run([softmax, accuracy], feed_dict=feed_dict)

    confusion_matrix = tf.confusion_matrix(labels=tf.argmax(DATASET.test.labels, 1), predictions=tf.argmax(pred[0], 1), num_classes=10)

    with open(os.path.join(dirname, '../metrics/eval.json'), 'w') as outfile:
        json.dump({ "accuracy" : str(pred[1]) }, outfile)

    with open(os.path.join(dirname, '../metrics/vega_confusion_matrix.json'), 'w') as outfile:
        json.dump(vega_heatmap(confusion_matrix.eval()), outfile)

