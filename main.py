import tensorflow as tf
from trainSpatial import trainSpatial
from trainBoth import trainBoth, testBoth
from trainAngular import trainAngular

FLAGS = tf.compat.v1.flags.FLAGS
tf.compat.v1.flags.DEFINE_integer("sort", 2, "different functions of main")
tf.compat.v1.flags.DEFINE_integer("scale", 2, "scale factor in spatial SR")
tf.compat.v1.flags.DEFINE_integer("batch_size", 128, "batch size")
tf.compat.v1.flags.DEFINE_integer("patch_width", 20, "width of sub-images")
tf.compat.v1.flags.DEFINE_integer("patch_height", 20, "height of sub-images")
tf.compat.v1.flags.DEFINE_integer("input_c", 2, "input channel")
tf.compat.v1.flags.DEFINE_float("learning_rate1", 1e-4, "learning rate for the first two layers")
tf.compat.v1.flags.DEFINE_float("learning_rate2", 1e-5, "learning rate for the last layers")
tf.compat.v1.flags.DEFINE_float("learning_rate_s", 2e-4, "learning rate for spatialSR net")
tf.compat.v1.flags.DEFINE_integer("epochs_a", 100, "training epochs for angularSR")
tf.compat.v1.flags.DEFINE_integer("epochs_s", 80, "training epochs for spatialSR")
tf.compat.v1.flags.DEFINE_boolean("is_train_a", True, "is training angularSR or not")
tf.compat.v1.flags.DEFINE_boolean("is_train_s", True, "is training spatialSR or not")
tf.compat.v1.flags.DEFINE_string("checkpoint_dir", "save_sum", "checkpoint directory for both")
tf.compat.v1.flags.DEFINE_string("save_dir", "save_sum", "save directory for both")
tf.compat.v1.flags.DEFINE_string("checkpoint_a_dir", "save", "checkpoint directory for angularSR")
tf.compat.v1.flags.DEFINE_string("save_a_dir", "save", "save directory for angularSR")
tf.compat.v1.flags.DEFINE_string("checkpoint_s_dir", "save_s", "checkpoint directory for spatialSR")
tf.compat.v1.flags.DEFINE_string("save_s_dir", "save_s", "save directory for spatialSR")
tf.compat.v1.flags.DEFINE_string("model_name", "SR.model", "model name for both")
tf.compat.v1.flags.DEFINE_string("model_a_name", "angularSR.model", "model name for angularSR")
tf.compat.v1.flags.DEFINE_string("model_s_name", "spatialSR.model", "model name for spatialSR")
tf.compat.v1.flags.DEFINE_string("img_file", "../img.mat", "dataset for input image")
tf.compat.v1.flags.DEFINE_string("gta_file", "../gt_a.mat", "ground truth for angular SR")
tf.compat.v1.flags.DEFINE_string("gts_file", "../gt_s.mat", "ground truth for spatial SR")
tf.compat.v1.flags.DEFINE_string("img_val_file", "../img_val.mat", "validation dataset for input image")
tf.compat.v1.flags.DEFINE_string("gta_val_file", "../gt_a_val.mat", "validation ground truth for angular SR")
tf.compat.v1.flags.DEFINE_string("gts_val_file", "../gt_s_val.mat", "validation ground truth for spatial SR")
tf.compat.v1.flags.DEFINE_string("img_test_file", "../img_test.mat", "test dataset for input image")
tf.compat.v1.flags.DEFINE_string("gta_test_file", "../gt_a_test.mat", "test ground truth for angular SR")
tf.compat.v1.flags.DEFINE_string("gts_test_file", "../gt_s_test.mat", "test ground truth for spatial SR")


def main(unused_argv):
    if FLAGS.sort == 1:
        trainBoth()
    elif FLAGS.sort == 2:
        testBoth()
    elif FLAGS.sort == 3:
        trainAngular()
    elif FLAGS.sort == 4:
        trainSpatial()


if __name__ == '__main__':
    tf.compat.v1.app.run()


