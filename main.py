import tensorflow as tf
import numpy as np
from angularSR import AngularSR
from PASSRnet import PASSRnet
import h5py
import time
import os

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
tf.compat.v1.flags.DEFINE_boolean("is_train", True, "is training or not")
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


def load_a(sess, saver_load_a):
    ckpt = tf.compat.v1.train.get_checkpoint_state(FLAGS.checkpoint_a_dir)

    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver_load_a.restore(sess, os.path.join(FLAGS.checkpoint_a_dir, ckpt_name))

        return True
    else:
        return False


def load_s(sess, saver_load_s):
    ckpt = tf.compat.v1.train.get_checkpoint_state(FLAGS.checkpoint_s_dir)

    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver_load_s.restore(sess, os.path.join(FLAGS.checkpoint_s_dir, ckpt_name))

        return True
    else:
        return False


def save_s(sess, step, saver):
    if not os.path.exists(FLAGS.save_s_dir):
        os.makedirs(FLAGS.save_s_dir)

    saver.save(sess, os.path.join(FLAGS.save_s_dir, FLAGS.model_s_name), global_step=step)


def trainAngular():
    sess = tf.compat.v1.Session()
    angular_sr = AngularSR(sess)
    if FLAGS.is_train:
        print("Start training!")
        angular_sr.train()
    else:
        print("Start testing!")
        angular_sr.test()


def testBoth():
    sess = tf.compat.v1.Session()
    # initial angularSR net
    angular_sr = AngularSR(sess)
    spatial_sr = PASSRnet(sess)

    image_l = tf.compat.v1.placeholder(tf.float32, shape=[None, FLAGS.patch_width, FLAGS.patch_height, 1],
                                       name='left_image')
    image_r = tf.compat.v1.placeholder(tf.float32, shape=[None, FLAGS.patch_width, FLAGS.patch_height, 1],
                                       name='right_image')

    pred_s = spatial_sr.outputPred(image_l, image_r)
    label_s = tf.compat.v1.placeholder(tf.float32, shape=[None, FLAGS.patch_width * 2, FLAGS.patch_height * 2, 1],
                                       name='label_of_spatialSR')
    loss = tf.compat.v1.losses.mean_squared_error(pred_s, label_s)

    # get different variables for two networks
    variable_a = [var for var in tf.compat.v1.trainable_variables() if "angular_SR" in var.name]
    variable_s = [var for var in tf.compat.v1.trainable_variables() if "PASSRnet" in var.name]

    # load data
    saver_a = tf.compat.v1.train.Saver(variable_a)
    if load_a(sess, saver_a):
        print("[* Load angularSR Successfully]")
    else:
        print("[* Load angularSR failed]")

    saver_s = tf.compat.v1.train.Saver(variable_s)
    if load_s(sess, saver_s):
        print("[* Load spatialSR Successfully]")
    else:
        print("[* Load spatialSR failed]")

    sess.run(tf.compat.v1.global_variables_initializer())

    # read test dataset
    data = h5py.File(FLAGS.img_test_file, 'r')
    img_test = data["data"][()]

    data_gt = h5py.File(FLAGS.gts_test_file, 'r')
    gts_test = data_gt["gt_s"][()]

    N_test = img_test.shape[0]
    img_test = np.transpose(img_test, (0, 3, 2, 1))
    test_l = img_test[:, :, :, 0]
    size = test_l.shape
    test_l = np.reshape(test_l, (size[0], size[1], size[2], 1))
    gts_test = np.transpose(gts_test, (0, 3, 2, 1))

    test_batch_number = N_test // FLAGS.batch_size

    test_temp_error = []
    for k in range(test_batch_number):
        start = k * FLAGS.batch_size
        end = (k + 1) * FLAGS.batch_size
        batch_img = img_test[start:end, :, :, :]
        batch_input_l = test_l[start:end, :, :, :]
        batch_gt = gts_test[start:end, :, :, :]

        batch_input_r = angular_sr.inputSpatial(batch_img)
        feed_dict = {image_l: batch_input_l, image_r: batch_input_r, label_s: batch_gt}
        ls = sess.run(loss, feed_dict=feed_dict)

        test_temp_error.append(ls)

    print("Loss: %.4f" % np.mean(test_temp_error).squeeze())


def trainBoth():
    sess = tf.compat.v1.Session()
    # initial angularSR net
    angular_sr = AngularSR(sess)
    spatial_sr = PASSRnet(sess)

    image_l = tf.compat.v1.placeholder(tf.float32, shape=[None, FLAGS.patch_width, FLAGS.patch_height, 1],
                                       name='left_image')
    image_r = tf.compat.v1.placeholder(tf.float32, shape=[None, FLAGS.patch_width, FLAGS.patch_height, 1],
                                       name='right_image')

    pred_s = spatial_sr.outputPred(image_l, image_r)
    label_s = tf.compat.v1.placeholder(tf.float32, shape=[None, FLAGS.patch_width*2, FLAGS.patch_height*2, 1],
                                       name='label_of_spatialSR')
    loss = tf.compat.v1.losses.mean_squared_error(pred_s, label_s)

    # get different variables for two networks
    variable_a = [var for var in tf.compat.v1.trainable_variables() if "angular_SR" in var.name]
    variable_s = [var for var in tf.compat.v1.trainable_variables() if "PASSRnet" in var.name]

    # load data
    saver_a = tf.compat.v1.train.Saver(variable_a)
    if load_a(sess, saver_a):
        print("[* Load angularSR Successfully]")
    else:
        print("[* Load angularSR failed]")

    saver_s = tf.compat.v1.train.Saver(variable_s)
    if load_s(sess, saver_s):
        print("[* Load spatialSR Successfully]")
    else:
        print("[* Load spatialSR failed]")

    # read training dataset
    data = h5py.File(FLAGS.img_file, 'r')
    img = data["data"][()]

    data_gt = h5py.File(FLAGS.gts_file, 'r')
    gts = data_gt["gt_s"][()]

    N = img.shape[0]
    idx = np.arange(N)
    np.random.shuffle(idx)

    img = img[idx, :, :, :]
    gts = gts[idx, :, :, :]

    img = np.transpose(img, (0, 3, 2, 1))
    input_l = img[:, :, :, 0]
    size = input_l.shape
    input_l = np.reshape(input_l, (size[0], size[1], size[2], 1))
    gts = np.transpose(gts, (0, 3, 2, 1))

    batch_number = N // FLAGS.batch_size

    # train spatialSR net only
    # learning rate decay constantly
    gs = tf.compat.v1.Variable(0, trainable=False)
    boundaries = [30 * batch_number, 60 * batch_number]
    values = [2e-4, 1e-4, 5e-5]
    learning_rate = tf.compat.v1.train.piecewise_constant(gs, boundaries, values)
    train_op = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, var_list=variable_s,
                                                                                      global_step=gs)
    sess.run(tf.compat.v1.global_variables_initializer())

    # read validation dataset
    data = h5py.File(FLAGS.img_val_file, 'r')
    img_val = data["data"][()]

    data_gt = h5py.File(FLAGS.gts_val_file, 'r')
    gts_val = data_gt["gt_s"][()]

    N_val = img_val.shape[0]
    img_val = np.transpose(img_val, (0, 3, 2, 1))
    val_l = img_val[:, :, :, 0]
    size = val_l.shape
    val_l = np.reshape(val_l, (size[0], size[1], size[2], 1))
    gts_val = np.transpose(gts_val, (0, 3, 2, 1))

    val_batch_number = N_val // FLAGS.batch_size

    Loss = []
    for i in range(FLAGS.epochs_s):
        temp_error = []
        valid_temp_error = []
        start_time = time.time()
        for j in range(batch_number):
            start = j * FLAGS.batch_size
            end = (j + 1) * FLAGS.batch_size
            batch_img = img[start:end, :, :, :]
            batch_input_l = input_l[start:end, :, :, :]
            batch_gt = gts[start:end, :, :, :]

            batch_input_r = angular_sr.inputSpatial(batch_img)

            feed_dict = {image_l: batch_input_l, image_r: batch_input_r, label_s: batch_gt}
            _, ls = sess.run([train_op, loss], feed_dict=feed_dict)

            temp_error.append(ls)

        Loss.append(np.mean(temp_error).squeeze())
        save_s(sess, i + 1, saver_s)

        for k in range(val_batch_number):
            start = k * FLAGS.batch_size
            end = (k + 1) * FLAGS.batch_size
            batch_img = img_val[start:end, :, :, :]
            batch_input_l = val_l[start:end, :, :, :]
            batch_gt = gts_val[start:end, :, :, :]

            batch_input_r = angular_sr.inputSpatial(batch_img)
            feed_dict = {image_l: batch_input_l, image_r: batch_input_r, label_s: batch_gt}
            ls = sess.run(loss, feed_dict=feed_dict)

            valid_temp_error.append(ls)

        lr = learning_rate.eval(session=sess)
        print("Epochs: %d, Loss: %.8f, valid Loss: %.8f, Time: %.4f, Learning rate: %.5f"
              % (i + 1, Loss[-1], np.mean(valid_temp_error).squeeze(), time.time() - start_time, lr))

    with h5py.File("spatial_loss.h5", 'w') as hf:
        hf.create_dataset("loss", data=Loss)


def main(unused_argv):
    if FLAGS.sort == 1:
        trainBoth()
    elif FLAGS.sort == 2:
        testBoth()


if __name__ == '__main__':
    tf.compat.v1.app.run()


