"""
Usage:
  # From tensorflow/models/
  # Create train data:
  python generate_tfrecord.py --csv_input=images/train_labels.csv --image_dir=images/train --output_path=train.record

  # Create test data:
  python generate_tfrecord.py --csv_input=images/test_labels.csv  --image_dir=images/test --output_path=test.record
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import pandas as pd
import tensorflow as tf

from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict

flags = tf.app.flags
flags.DEFINE_string('csv_input', '', 'Path to the CSV input')
flags.DEFINE_string('image_dir', '', 'Path to the image directory')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
FLAGS = flags.FLAGS


# TO-DO replace this with label map
def class_text_to_int(row_label):
    if row_label == 'parachute hair cream':
        return 1
    elif row_label == 'dabur anmol gold coconut oil':
        return 2
    elif row_label == 'catch kashmiri mirch':
        return 3
    elif row_label == 'patanjali chyawanprash':
        return 4
    elif row_label == 'loreal smooth intense serum':
        return 5
    elif row_label == 'emami mustard oil':
        return 6
    elif row_label == 'dabur chyawanprash':
        return 7
    elif row_label == 'good life hing':
        return 8
    elif row_label == 'catch hing':
        return 9
    elif row_label == 'ponds moisturizing lotion':
        return 10
    elif row_label == 'catch garam masala':
        return 11
    elif row_label == 'dabur chyawanprash 2x immunity':
        return 12
    elif row_label == 'dabur honey':
        return 13
    elif row_label == 'everest biryani masala':
        return 14
    elif row_label == 'patanjali aloe vera gel':
        return 15
    elif row_label == 'grofers kishmish':
        return 16
    elif row_label == 'grofers room freshener':
        return 17
    elif row_label == 'kiwi express shine sponge':
        return 18
    elif row_label == 'garnier cleansing water':
        return 19
    elif row_label == 'bc gulab jamun':
        return 20
    elif row_label == 'bourn vita':
        return 21
    elif row_label == 'catch chicken masala':
        return 22
    elif row_label == 'everest pav bhaji masala':
        return 23
    elif row_label == 'cinthol talcum powder':
        return 24
    elif row_label == 'savemore glass cleaner':
        return 25
    elif row_label == 'park avenue voyage perfume':
        return 26
    elif row_label == 'orange coconut oil':
        return 27
    elif row_label == 'everest hing':
        return 28
    elif row_label == 'amul spray milk powder':
        return 29
    elif row_label == 'yardley body spray':
        return 30
    elif row_label == 'goodnight':
        return 31
    elif row_label == 'fair and handsome':
        return 32
    else:
        None


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path):
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main(_):
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
    path = os.path.join(os.getcwd(), FLAGS.image_dir)
    examples = pd.read_csv(FLAGS.csv_input)
    grouped = split(examples, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group, path)
        writer.write(tf_example.SerializeToString())

    writer.close()
    output_path = os.path.join(os.getcwd(), FLAGS.output_path)
    print('Successfully created the TFRecords: {}'.format(output_path))


if __name__ == '__main__':
    tf.app.run()
