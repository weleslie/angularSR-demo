import tensorflow as tf
# sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
import os
import numpy as np
import h5py
import time

FLAGS = tf.compat.v1.flags.FLAGS
tf.compat.v1.disable_eager_execution()


class AngularSR(object):
    def __init__(self, sess):
        self.batch_size = FLAGS.batch_size
        self.learning_rate1 = FLAGS.learning_rate1
        self.learning_rate2 = FLAGS.learning_rate2
        self.epochs = FLAGS.epochs_a
        self.width = FLAGS.patch_width
        self.height = FLAGS.patch_height
        self.channel = FLAGS.input_c
        self.sess = sess

        self._init_graph()

    def _init_graph(self):
        self.input = tf.compat.v1.placeholder(tf.float32, shape=(None, self.width, self.height, self.channel),
                                              name="image_pairs_input")
        self.label = tf.compat.v1.placeholder(tf.float32, shape=(None, self.width, self.height, 1), name="groundtruth")

        self.pred = self.angularSR()

        self.loss = tf.compat.v1.losses.mean_squared_error(self.label, self.pred)

        # load partial parameters
        self.variable = [var for var in tf.compat.v1.trainable_variables() if "angular_SR" in var.name]
        self.saver_load = tf.compat.v1.train.Saver(self.variable)
        self.saver_save = tf.compat.v1.train.Saver(max_to_keep=10)

    def angularSR(self):
        with tf.name_scope("angular_SR"):
            layer1 = tf.compat.v1.keras.layers.Conv2D(64, kernel_size=(9, 9), padding="same", activation='relu',
                                                      kernel_initializer=tf.compat.v1.keras.initializers.random_normal(
                                                          mean=0.0, stddev=1e-3),
                                                      name="layer1")(self.input)
            layer2 = tf.compat.v1.keras.layers.Conv2D(32, kernel_size=(5, 5), padding="same", activation="relu",
                                                      kernel_initializer=tf.compat.v1.keras.initializers.random_normal(
                                                          mean=0.0, stddev=1e-3),
                                                      name="layer2")(layer1)
            output = tf.compat.v1.keras.layers.Conv2D(1, kernel_size=(5, 5), padding="same",
                                                      kernel_initializer=tf.compat.v1.keras.initializers.random_normal(
                                                          mean=0.0, stddev=1e-3),
                                                      name="output")(layer2)

        return output

    def train(self):
        if self.load():
            print("[* Load Successfully]")
        else:
            print("[* Load failed]")

        # setting learning rate hierarchically
        var1 = tf.compat.v1.trainable_variables()[0:4]
        var2 = tf.compat.v1.trainable_variables()[4:]

        train_op1 = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate1).minimize(self.loss,
                                                                                                 var_list=var1)
        train_op2 = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate2).minimize(self.loss,
                                                                                                 var_list=var2)

        train_op = tf.compat.v1.group(train_op1, train_op2)

        self.sess.run(tf.compat.v1.global_variables_initializer())

        # read training dataset
        data = h5py.File(FLAGS.img_file, 'r')
        img = data["data"][()]

        data_gt = h5py.File(FLAGS.gt_file, 'r')
        gt = data_gt["gt_a"][()]

        N = img.shape[0]
        idx = np.arange(N)
        np.random.shuffle(idx)

        img = img[idx, :, :, :]
        gt = gt[idx, :, :, :]

        img = np.transpose(img, (0, 3, 2, 1))
        gt = np.transpose(gt, (0, 3, 2, 1))

        batch_number = N // self.batch_size

        # read validation dataset
        data = h5py.File(FLAGS.img_val_file, 'r')
        img_val = data["data"][()]

        data_gt = h5py.File(FLAGS.gt_val_file, 'r')
        gt_val = data_gt["gt_a"][()]

        N_val = img_val.shape[0]
        img_val = np.transpose(img_val, (0, 3, 2, 1))
        gt_val = np.transpose(gt_val, (0, 3, 2, 1))

        val_batch_number = N_val // self.batch_size

        Loss = []
        for i in range(self.epochs):
            temp_error = []
            valid_temp_error = []
            start_time = time.time()
            for j in range(batch_number):
                start = j * self.batch_size
                end = (j + 1) * self.batch_size
                batch_img = img[start:end, :, :, :]
                batch_gt = gt[start:end, :, :, :]

                feed_dict = {self.input: batch_img, self.label: batch_gt}
                _, loss = self.sess.run((train_op, self.loss), feed_dict=feed_dict)

                temp_error.append(loss)

            Loss.append(np.mean(temp_error).squeeze())
            self.save(i + 1)

            for k in range(val_batch_number):
                start = k * self.batch_size
                end = (k + 1) * self.batch_size
                batch_img = img_val[start:end, :, :, :]
                batch_gt = gt_val[start:end, :, :, :]

                feed_dict = {self.input: batch_img, self.label: batch_gt}
                loss = self.sess.run(self.loss, feed_dict=feed_dict)

                valid_temp_error.append(loss)

            print("Epochs: %d, Loss: %.8f, valid Loss: %.8f, Time: %.4f"
                  % (i + 1, Loss[-1], np.mean(valid_temp_error).squeeze(), time.time() - start_time))

        with h5py.File("loss.h5", 'w') as hf:
            hf.create_dataset("loss", data=Loss)

    def test(self):
        # read validation dataset
        data = h5py.File(FLAGS.img_test_file, 'r')
        img_test = data["data"][()]

        data_gt = h5py.File(FLAGS.gt_test_file, 'r')
        gt_test = data_gt["gt_a"][()]

        img_test = np.transpose(img_test, (0, 3, 2, 1))
        gt_test = np.transpose(gt_test, (0, 3, 2, 1))

        N = img_test.shape[0]
        batch_number = N // self.batch_size

        Loss = []
        for i in range(batch_number):
            start = i * self.batch_size
            end = (i + 1) * self.batch_size
            batch_img = img_test[start:end, :, :, :]
            batch_gt = gt_test[start:end, :, :, :]

            feed_dict = {self.input: batch_img, self.label: batch_gt}

            loss = self.sess.run(self.loss, feed_dict=feed_dict)

            Loss.append(loss)

        print("Loss: %.8f" % np.mean(Loss).squeeze())

    def inputSpatial(self, batch_img, batch_gta):
        feed_dict = {self.input: batch_img, self.label: batch_gta}
        loss, output = self.sess.run([self.loss, self.pred], feed_dict=feed_dict)

        return loss, output

    def load(self):
        ckpt = tf.compat.v1.train.get_checkpoint_state(FLAGS.checkpoint_a_dir)

        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver_load.restore(self.sess, os.path.join(FLAGS.checkpoint_a_dir, ckpt_name))

            return True
        else:
            return False

    def save(self, step):
        if not os.path.exists(FLAGS.save_a_dir):
            os.makedirs(FLAGS.save_a_dir)

        self.saver_save.save(self.sess, os.path.join(FLAGS.save_a_dir, FLAGS.model_a_name), global_step=step)