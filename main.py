import tensorflow as tf
import numpy as np
from angularSR import AngularSR

FLAGS = tf.compat.v1.flags.FLAGS
tf.compat.v1.flags.DEFINE_integer("batch_size", 64, "batch size")
tf.compat.v1.flags.DEFINE_integer("patch_width", 20, "width of sub-images")
tf.compat.v1.flags.DEFINE_integer("patch_height", 20, "height of sub-images")
tf.compat.v1.flags.DEFINE_integer("input_c", 2, "input channel")
tf.compat.v1.flags.DEFINE_float("learning_rate1", 1e-4, "learning rate for the first two layers")
tf.compat.v1.flags.DEFINE_float("learning_rate2", 1e-5, "learning rate for the last layers")
tf.compat.v1.flags.DEFINE_integer("epochs", 100, "training epochs")
tf.compat.v1.flags.DEFINE_boolean("is_train", True, "is training or not")
tf.compat.v1.flags.DEFINE_string("checkpoint_dir", "save", "checkpoint directory")
tf.compat.v1.flags.DEFINE_string("model_name", "angularSR.model", "model name for checkpoint")
tf.compat.v1.flags.DEFINE_string("img_file", "../img.mat", "dataset for input image")
tf.compat.v1.flags.DEFINE_string("gt_file", "../gt_a.mat", "ground truth for angular SR")
tf.compat.v1.flags.DEFINE_string("img_val_file", "../img_val.mat", "validation dataset for input image")
tf.compat.v1.flags.DEFINE_string("gt_val_file", "../gt_a_val.mat", "validation ground truth for angular SR")
tf.compat.v1.flags.DEFINE_string("img_test_file", "../img_test.mat", "test dataset for input image")
tf.compat.v1.flags.DEFINE_string("gt_test_file", "../gt_a_test.mat", "test ground truth for angular SR")


def main(unused_argv):
    angular_sr = AngularSR()
    if FLAGS.is_train:
        print("Start training!")
        angular_sr.train()
    else:
        print("Start testing!")
        angular_sr.test()


if __name__ == '__main__':
    tf.compat.v1.app.run()


