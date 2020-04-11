import tensorflow as tf
from angularSR import AngularSR

FLAGS = tf.compat.v1.flags.FLAGS
tf.compat.v1.disable_eager_execution()


def trainAngular():
    sess = tf.compat.v1.Session()
    angular_sr = AngularSR(sess)
    if FLAGS.is_train_a:
        print("Start training!")
        angular_sr.train()
    else:
        print("Start testing!")
        angular_sr.test()
