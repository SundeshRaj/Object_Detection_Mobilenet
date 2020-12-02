import numpy as np
import pandas as pd
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2
from PIL import Image

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

if __name__ == "__main__":

    MODEL_NAME = 'ssd_mobilenet_model'
    PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
    LABELS_FILE = MODEL_NAME + '/mscoco_label_map.pbtxt'

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        graph_def = tf.compat.v1.GraphDef()
        with tf.compat.v2.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(graph_def, name='')

    TEST_IMAGES_PATH = 'object_detection/test_images/image1.jpg'

    img = Image.open(TEST_IMAGES_PATH)
    image_np = load_image_into_numpy_array(img)

    cv2.imshow("Output image", image_np)
    cv2.waitKey(0)

    cv2.destroyAllWindows()

    print("Done")

