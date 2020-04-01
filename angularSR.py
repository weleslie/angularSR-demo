import tensorflow as tf
import os
import numpy as np
import h5py

FLAGS = tf.app.flags.FLAGS


class AngularSR(object):
    def __init__(self):
        self.batch_size = FLAGS.batch_size
        self.learning_rate1 = FLAGS.learning_rate1
        self.learning_rate2 = FLAGS.learning_rate2
        self.is_train = FLAGS.is_train
        self.epochs = FLAGS.epochs
        self.width = FLAGS.patch_width
        self.height = FLAGS.patch_height
        self.channel = FLAGS.input_c

        self._init_graph()

    def _init_graph(self):
        self.input = tf.placeholder(tf.float32, shape=(None, self.width, self.height, self.channel), name="image_pairs_input")
        self.label = tf.placeholder(tf.float32, shape=(None, self.width, self.height, 1), name="groundtruth")

        self.pred = self.angularSR()

        self.loss = tf.losses.mean_squared_error(self.label, self.pred)

        self.sess = tf.Session()

        self.saver = tf.train.Saver()

    def angularSR(self):
        layer1 = tf.keras.layers.Conv2D(64, kernel_size=(9, 9), padding="same", activation='relu',
                                        kernel_initializer=tf.keras.initializers.random_normal(mean=0.0, stddev=1e-3),
                                        name="layer1")(self.input)
        layer2 = tf.keras.layers.Conv2D(32, kernel_size=(5, 5), padding="same", activation="relu",
                                        kernel_initializer=tf.keras.initializers.random_normal(mean=0.0, stddev=1e-3),
                                        name="layer2")(layer1)
        output = tf.keras.layers.Conv2D(1, kernel_size=(5, 5), padding="same",
                                        kernel_initializer=tf.keras.initializers.random_normal(mean=0.0, stddev=1e-3),
                                        name="output")(layer2)

        return output

    def train(self):
        # setting learning rate hierarchically
        var1 = tf.trainable_variables()[0:4]
        var2 = tf.trainable_variables()[4:]

        opt1 = tf.train.AdamOptimizer(learning_rate=self.learning_rate1)
        opt2 = tf.train.AdamOptimizer(learning_rate=self.learning_rate2)

        grad = tf.gradients(self.loss, var1 + var2)
        grad1 = grad[0:len(var1)]
        grad2 = grad[len(var1):]

        train_op1 = opt1.apply_gradients(zip(grad1, var1))
        train_op2 = opt2.apply_gradients(zip(grad2, var2))
        train_op = tf.group(train_op1, train_op2)

        self.sess.run(tf.global_variables_initializer())

        # load checkpoint
        if self.load():
            print("[* Load Successfully]")
        else:
            print("[* Load failed]")

        data = h5py.File(FLAGS.img_file, 'r')
        img = data["data"][()]

        data_gt = h5py.File(FLAGS.gt_file, 'r')
        gt = data_gt["gt_a"][()]

        N = img.shape[3]
        idx = np.arange(N)
        np.random.shuffle(idx)

        img = img[:, :, :, idx]
        gt = gt[:, :, :, idx]

        tf.transpose(img, perm=[3, 0, 1, 2])
        tf.transpose(gt, perm=[3, 0, 1, 2])

        batch_number = N // self.batch_size

        Loss = []
        for i in range(self.epochs):
            temp_error = []
            for j in range(batch_number):
                start = j * self.batch_size
                end = (j + 1) * self.batch_size
                batch_img = img[start:end, :, :, :]
                batch_gt = gt[start:end, :, :, :]

                feed_dict = {self.input: batch_img, self.label: batch_gt}
                _, loss = self.sess.run((train_op, self.loss), feed_dict=feed_dict)

                temp_error.append(loss)

            Loss.append(np.mean(temp_error).squeeze())
            self.save(i)

            print("Epochs: %d, Loss: %.4f" % (i+1, Loss[-1]))

    def load(self):
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)

        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(FLAGS.checkpoint_dir, ckpt_name))

            return True
        else:
            return False

    def save(self, step):
        if not os.path.exists(FLAGS.checkpoint_dir):
            os.makedirs(FLAGS.checkpoint_dir)

        self.saver.save(self.sess, os.path.join(FLAGS.checkpoint_dir, FLAGS.model_name), global_step=step)

