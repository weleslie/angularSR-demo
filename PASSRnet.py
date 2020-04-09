import tensorflow as tf


class PASSRnet(object):
    def __init__(self, width, height, scale):
        self.width = width
        self.height = height
        self.scale = scale

    def forward(self, image_l, image_r):
        with tf.name_scope("PASSRnet"):
            model = self.shareBlock()
            img_l = model(image_l)
            img_r = model(image_r)

            pam = self.PAM(img_l, img_r)
            up1 = self.resB(pam, '3_1')
            up2 = self.resB(up1, '3_2')
            up3 = self.resB(up2, '3_3')
            up4 = self.resB(up3, '3_4')

            up5 = tf.compat.v1.keras.layers.Conv2D(filters=64 * self.scale ** 2, kernel_size=(1, 1), strides=(1, 1),
                                                   padding='same', kernel_initializer='he_normal', activation=None,
                                                   use_bias=False, name='sub_pixel')(up4)
            up6 = self.pixelShuffle(up5, self.scale)
            up7 = tf.compat.v1.keras.layers.Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1),
                                                   padding='same', kernel_initializer='he_normal', activation=None,
                                                   use_bias=False, name='conv3_a')(up6)
            output = tf.compat.v1.keras.layers.Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1),
                                                      padding='same', kernel_initializer='he_normal', activation=None,
                                                      use_bias=False, name='conv3_b')(up7)

        return output

    def shareBlock(self):
        img_input = tf.compat.v1.keras.layers.Input(shape=(self.width, self.height, 1))
        l1 = tf.compat.v1.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),
                                              kernel_initializer='he_normal', padding='same', use_bias=False,
                                              activation=None, name='layer1')(img_input)
        l1 = tf.compat.v1.keras.layers.LeakyReLU(alpha=0.1)(l1)

        l2 = self.resB(l1, '0')
        l3 = self.resASPPB(l2, '1_1')
        l4 = self.resB(l3, '1_1')
        l5 = self.resASPPB(l4, '1_2')
        l6 = self.resB(l5, '1_2')

        model = tf.compat.v1.keras.models.Model(img_input, l6)

        return model

    def resB(self, input_, num):
        res1 = tf.compat.v1.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),
                                                kernel_initializer='he_normal', padding='same', use_bia=False,
                                                activation=None, name='res_layer_' + num + '_1')(input_)
        res1 = tf.compat.v1.keras.layers.LeakyReLU(alpha=0.1)(res1)

        res2 = tf.compat.v1.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),
                                                kernel_initializer='he_normal', padding='same', use_bias=False,
                                                activation=None, name='res_layer_' + num + '_2')(res1)

        output = tf.compat.v1.add(input_, res2)
        return output

    def resASPPB(self, input_, num):
        res1_1 = tf.compat.v1.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), dilation_rate=(1, 1),
                                                  kernel_initializer='he_normal', padding='same', use_bias=False,
                                                  activation=None, name='aspp_' + num + '_1')(input_)
        res1_1 = tf.compat.v1.keras.layers.LeakyReLU(alpha=0.1)(res1_1)

        res1_2 = tf.compat.v1.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), dilation_rate=(4, 4),
                                                  kernel_initializer='he_normal', padding='same', use_bias=False,
                                                  activation=None, name='aspp_' + num + '_1')(input_)
        res1_2 = tf.compat.v1.keras.layers.LeakyReLU(alpha=0.1)(res1_2)

        res1_3 = tf.compat.v1.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), dilation_rate=(8, 8),
                                                  kernel_initializer='he_normal', padding='same', use_bias=False,
                                                  activation=None, name='aspp_' + num + '_1')(input_)
        res1_3 = tf.compat.v1.keras.layers.LeakyReLU(alpha=0.1)(res1_3)

        res1 = tf.compat.v1.concat([res1_1, res1_2, res1_3], axis=3)
        res1 = tf.compat.v1.keras.layers.Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1),
                                                kernel_initializer='he_normal', padding='same', use_bias=False,
                                                activation=None, name='aspp_1')(res1)

        res2_1 = tf.compat.v1.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), dilation_rate=(1, 1),
                                                  kernel_initializer='he_normal', padding='same', use_bias=False,
                                                  activation=None, name='aspp_' + num + '_2')(res1)
        res2_1 = tf.compat.v1.keras.layers.LeakyReLU(alpha=0.1)(res2_1)

        res2_2 = tf.compat.v1.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), dilation_rate=(4, 4),
                                                  kernel_initializer='he_normal', padding='same', use_bias=False,
                                                  activation=None, name='aspp_' + num + '_2')(res1)
        res2_2 = tf.compat.v1.keras.layers.LeakyReLU(alpha=0.1)(res2_2)

        res2_3 = tf.compat.v1.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), dilation_rate=(8, 8),
                                                  kernel_initializer='he_normal', padding='same', use_bias=False,
                                                  activation=None, name='aspp_' + num + '_2')(res1)
        res2_3 = tf.compat.v1.keras.layers.LeakyReLU(alpha=0.1)(res2_3)

        res2 = tf.compat.v1.concat([res2_1, res2_2, res2_3], axis=3)
        res2 = tf.compat.v1.keras.layers.Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1),
                                                kernel_initializer='he_normal', padding='same', use_bias=False,
                                                activation=None, name='aspp_2')(res2)

        res3_1 = tf.compat.v1.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), dilation_rate=(1, 1),
                                                  kernel_initializer='he_normal', padding='same', use_bias=False,
                                                  activation=None, name='aspp_' + num + '_3')(res2)
        res3_1 = tf.compat.v1.keras.layers.LeakyReLU(alpha=0.1)(res3_1)

        res3_2 = tf.compat.v1.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), dilation_rate=(4, 4),
                                                  kernel_initializer='he_normal', padding='same', use_bias=False,
                                                  activation=None, name='aspp_' + num + '_3')(res2)
        res3_2 = tf.compat.v1.keras.layers.LeakyReLU(alpha=0.1)(res3_2)

        res3_3 = tf.compat.v1.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), dilation_rate=(8, 8),
                                                  kernel_initializer='he_normal', padding='same', use_bias=False,
                                                  activation=None, name='aspp_' + num + '_3')(res2)
        res3_3 = tf.compat.v1.keras.layers.LeakyReLU(alpha=0.1)(res3_3)

        res3 = tf.compat.v1.concat([res3_1, res3_2, res3_3], axis=3)
        res3 = tf.compat.v1.keras.layers.Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1),
                                                kernel_initializer='he_normal', padding='same', use_bias=False,
                                                activation=None, name='aspp_3')(res3)

        output = input_ + res1 + res2 + res3
        return output

    def PAM(self, input_l, input_r):
        l1 = self.resB(input_l, '2_l')
        r1 = self.resB(input_r, '2_r')

        Q = tf.compat.v1.keras.layers.Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1),
                                             kernel_initializer='he_normal', padding='same', use_bias=True,
                                             activation=None, name='Q')(l1)
        K = tf.compat.v1.keras.layers.Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1),
                                             kernel_initializer='he_normal', padding='same', use_bias=True,
                                             activation=None, name='K')(r1)
        V = tf.compat.v1.keras.layers.Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1),
                                             kernel_initializer='he_normal', padding='same', use_bias=True,
                                             activation=None, name='V')(r1)

        K = tf.compat.v1.transpose(K, perm=[0, 1, 3, 2])
        score = tf.compat.v1.matmul(Q, K)
        M = tf.compat.v1.nn.softmax(score, axis=3)

        buffer = tf.compat.v1.matmul(M, V)

        fusion = tf.compat.v1.concat([buffer, l1], axis=3)

        output = tf.compat.v1.keras.layers.Conv2D(filter=64, kernel_size=(1, 1), strides=(1, 1),
                                                  padding='same', kernel_initializer='he_normal', use_bias=True,
                                                  activation=None, name='fusion')(fusion)

        return output

    def pixelShuffle(self, feature, scale, channel=64):
        Xc = tf.compat.v1.split(feature, channel, 3)
        output = tf.compat.v1.concat([self._pixel_shift(x, scale) for x in Xc], axis=3)

        return output

    def _pixel_shift(self, feature, scale):
        bsize, a, b, c = feature.get_shape().as_list()
        X = tf.reshape(feature, (bsize, a, b, scale, scale))
        #         X = tf.compat.v1.transpose(X, (0, 1, 2, 4, 3))

        # important step
        X = tf.compat.v1.split(X, a, 1)
        X = tf.compat.v1.concat([tf.compat.v1.squeeze(x, axis=1) for x in X], axis=2)
        X = tf.compat.v1.split(X, b, 1)
        X = tf.compat.v1.concat([tf.compat.v1.squeeze(x, axis=1) for x in X], axis=2)

        return tf.compat.v1.reshape(X, (bsize, scale * a, scale * b, 1))


