from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

tf.compat.v1.disable_eager_execution()


def load_graph(model_file):
    graph = tf.compat.v1.Graph()
    graph_def = tf.compat.v1.GraphDef()

    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.compat.v1.import_graph_def(graph_def)

    return graph


def load_labels(label_file):
    label = []
    proto_as_ascii_lines = tf.compat.v1.gfile.GFile(label_file).readlines()
    for l in proto_as_ascii_lines:
        label.append(l.rstrip())
    return label


def read_tensor_from_image_file2(image,
                                 input_height=224,
                                 input_width=224,
                                 input_mean=0,
                                 input_std=255):
    try:
        image_arr = np.asarray(bytearray(image.read()), dtype=np.uint8)
        img = cv2.imdecode(image_arr, cv2.IMREAD_GRAYSCALE)
        image_reader = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    except:
        image_reader = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    float_caster = tf.compat.v1.cast(image_reader, tf.compat.v1.float32)
    dims_expander = tf.compat.v1.expand_dims(float_caster, 0)
    resized = tf.compat.v1.image.resize_bilinear(dims_expander, [input_height, input_width])
    normalized = tf.compat.v1.divide(tf.compat.v1.subtract(resized, [input_mean]), [input_std])
    sess = tf.compat.v1.Session()
    result = sess.run(normalized)

    return result


def text_type_classification(image):
    model_file = "../src/model_text_classificator/output_graph.pb"
    label_file = "../src/model_text_classificator/output_labels.txt"
    input_layer = "Placeholder"
    output_layer = "final_result"

    graph = load_graph(model_file)
    t = read_tensor_from_image_file2(image)

    input_name = "import/" + input_layer
    output_name = "import/" + output_layer
    input_operation = graph.get_operation_by_name(input_name)
    output_operation = graph.get_operation_by_name(output_name)

    with tf.compat.v1.Session(graph=graph) as sess:
        results = sess.run(output_operation.outputs[0], {
            input_operation.outputs[0]: t
        })
    results = np.squeeze(results)

    top_k = results.argsort()[-5:][::-1]
    labels = load_labels(label_file)

    result = dict()

    for i in top_k:
        result[labels[i]] = results[i]

    return result
