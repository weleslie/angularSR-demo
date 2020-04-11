import tensorflow as tf
import numpy as np
from PASSRnet import PASSRnet
import h5py
import os
import time

FLAGS = tf.compat.v1.flags.FLAGS
tf.compat.v1.disable_eager_execution()


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


def trainSpatial():
    sess = tf.compat.v1.Session()
    spatial_sr = PASSRnet(sess)

    image_l = tf.compat.v1.placeholder(tf.float32, shape=[None, FLAGS.patch_width, FLAGS.patch_height, 1],
                                       name='left_image')
    image_r = tf.compat.v1.placeholder(tf.float32, shape=[None, FLAGS.patch_width, FLAGS.patch_height, 1],
                                       name='right_image')

    pred_s = spatial_sr.outputPred(image_l, image_r)
    label_s = tf.compat.v1.placeholder(tf.float32, shape=[None, FLAGS.patch_width * 2, FLAGS.patch_height * 2, 1],
                                       name='label_of_spatialSR')

    loss = tf.compat.v1.losses.mean_squared_error(pred_s, label_s)

    variable_s = [var for var in tf.compat.v1.trainable_variables() if "PASSRnet" in var.name]

    if FLAGS.is_train_s is True:
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
        img_l, img_r = np.split(img, 2, axis=3)
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

        saver_s = tf.compat.v1.train.Saver(variable_s)

        sess.run(tf.compat.v1.global_variables_initializer())

        # load data
        if load_s(sess, saver_s):
            print("[* Load angularSR Successfully]")
        else:
            print("[* Load angularSR failed]")

        # read test dataset
        data = h5py.File(FLAGS.img_test_file, 'r')
        img_test = data["data"][()]

        data_gt = h5py.File(FLAGS.gts_test_file, 'r')
        gts_test = data_gt["gt_s"][()]

        N_test = img_test.shape[0]
        img_test = np.transpose(img_test, (0, 3, 2, 1))
        img_test_l, img_test_r = np.split(img_test, 2, axis=3)
        gts_test = np.transpose(gts_test, (0, 3, 2, 1))

        test_batch_number = N_test // FLAGS.batch_size

        Loss = []
        for i in range(FLAGS.epochs_s):
            temp_error = []
            test_temp_error = []
            start_time = time.time()
            for j in range(batch_number):
                start = j * FLAGS.batch_size
                end = (j + 1) * FLAGS.batch_size
                batch_input_l = img_l[start:end, :, :, :]
                batch_input_r = img_r[start:end, :, :, :]
                batch_gt = gts[start:end, :, :, :]

                feed_dict = {image_l: batch_input_l, image_r: batch_input_r, label_s: batch_gt}
                _, ls = sess.run([train_op, loss], feed_dict=feed_dict)

                temp_error.append(ls)

            Loss.append(np.mean(temp_error).squeeze())
            save_s(sess, i + 1, saver_s)

            for k in range(test_batch_number):
                start = k * FLAGS.batch_size
                end = (k + 1) * FLAGS.batch_size
                batch_input_l = img_test_l[start:end, :, :, :]
                batch_input_r = img_test_r[start:end, :, :, :]
                batch_gt = gts_test[start:end, :, :, :]

                feed_dict = {image_l: batch_input_l, image_r: batch_input_r, label_s: batch_gt}
                ls = sess.run(loss, feed_dict=feed_dict)

                test_temp_error.append(ls)

            lr = learning_rate.eval(session=sess)
            print("Epochs: %d, Loss: %.8f, valid Loss: %.8f, Time: %.4f, Learning rate: %.5f"
                  % (i + 1, Loss[-1], np.mean(test_temp_error).squeeze(), time.time() - start_time, lr))

        with h5py.File("spatial_loss.h5", 'w') as hf:
            hf.create_dataset("loss", data=Loss)

    else:
        sess = tf.compat.v1.Session()
        # initial spatialSR net
        spatial_sr = PASSRnet(sess)

        image_l = tf.compat.v1.placeholder(tf.float32, shape=[None, FLAGS.patch_width, FLAGS.patch_height, 1],
                                           name='left_image')
        image_r = tf.compat.v1.placeholder(tf.float32, shape=[None, FLAGS.patch_width, FLAGS.patch_height, 1],
                                           name='right_image')

        pred_s = spatial_sr.outputPred(image_l, image_r)
        label_s = tf.compat.v1.placeholder(tf.float32, shape=[None, FLAGS.patch_width * 2, FLAGS.patch_height * 2, 1],
                                           name='label_of_spatialSR')
        loss = tf.compat.v1.losses.mean_squared_error(pred_s, label_s)

        # get variables for spatialSR networks
        variable_s = [var for var in tf.compat.v1.trainable_variables() if "PASSRnet" in var.name]

        sess.run(tf.compat.v1.global_variables_initializer())

        # load data
        saver_s = tf.compat.v1.train.Saver(variable_s)
        if load_s(sess, saver_s):
            print("[* Load spatialSR Successfully]")
        else:
            print("[* Load spatialSR failed]")

        # read test dataset
        data = h5py.File(FLAGS.img_test_file, 'r')
        img_test = data["data"][()]

        data_gt = h5py.File(FLAGS.gts_test_file, 'r')
        gts_test = data_gt["gt_s"][()]

        N_test = img_test.shape[0]
        img_test = np.transpose(img_test, (0, 3, 2, 1))
        img_test_l, img_test_r = np.split(img_test, 2, axis=3)
        gts_test = np.transpose(gts_test, (0, 3, 2, 1))

        test_batch_number = N_test // FLAGS.batch_size

        test_temp_error = []
        for k in range(test_batch_number):
            start = k * FLAGS.batch_size
            end = (k + 1) * FLAGS.batch_size
            batch_input_l = img_test_l[start:end, :, :, :]
            batch_input_r = img_test_r[start:end, :, :, :]
            batch_gt = gts_test[start:end, :, :, :]

            feed_dict = {image_l: batch_input_l, image_r: batch_input_r, label_s: batch_gt}
            ls = sess.run(loss, feed_dict=feed_dict)

            test_temp_error.append(ls)

        print("Loss: %.4f" % np.mean(test_temp_error).squeeze())
