import tensorflow as tf
import numpy as np

class SegNet:
    def __init__(self, maximages):
        self.max_images = maximages


    def conv_2d(self, X, filter, stride,name="conv"):
        temp2 = X.get_shape().as_list()
        #print(temp2)
        return tf.nn.conv2d(X, filter=filter, strides=stride, padding="SAME", name=name)

    def deconv_2d(self,X,filter,stride,name="deconv"):
        temp=X.get_shape().as_list()
        batch=temp[0]
        h=temp[1]
        w=temp[2]
        temp2=filter.get_shape().as_list()
        #print(temp)

        return tf.nn.conv2d_transpose(X, filter=filter, strides=stride,output_shape=[batch,h,w,temp2[2]],padding="SAME", name=name)


    def max_pool(self, X,name="max_pool"):
        return tf.nn.max_pool(X, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", data_format="NHWC",name=name)

    def max_pool_transpose(self,x,size):
        out = tf.concat([x, tf.zeros_like(x)], 3)
        out = tf.concat([out, tf.zeros_like(out)], 2)

        sh = x.get_shape().as_list()
        if None not in sh[1:]:
            out_size = [-1, sh[1] * size, sh[2] * size, sh[3]]
            return tf.reshape(out, out_size)

        shv = tf.shape(x)
        ret = tf.reshape(out, tf.stack([-1, shv[1] * size, shv[2] * size, sh[3]]))
        ret.set_shape([None, None, None, sh[3]])
        return ret

    def relu(self, X,name="relu"):
        return tf.nn.relu(X,name=name)

    def filter(self, height, width, _in, out):
        return tf.Variable(tf.truncated_normal([height, width, _in, out], dtype=tf.float32,
                                               stddev=1e-1), name='weights')

    def biases(self, shape,name='biases'):
        return tf.Variable(tf.constant(0.0001, shape=[shape], dtype=tf.float32),
                           trainable=True, name=name)

    def inference(self, images):
        tf.summary.image('input_img', images, max_outputs=self.max_images)
        with tf.variable_scope('pool1'):

            conv = self.conv_2d(images, filter=self.filter(3, 3, 3, 64), stride=[1, 1, 1, 1])
            out = tf.nn.bias_add(conv,self.biases(64))
            conv1_1=self.relu(out,name="conv1_1")

            conv = self.conv_2d(conv1_1, filter=self.filter(3, 3, 64, 64), stride=[1, 1, 1, 1])
            out = tf.nn.bias_add(conv, self.biases(64))
            conv1_2 = self.relu(out, name="conv1_2")

            conv1_max_pool=self.max_pool(conv1_2,"max_pool_1")

        with tf.variable_scope('pool2'):

            conv = self.conv_2d(conv1_max_pool, filter=self.filter(3, 3, 64, 128), stride=[1, 1, 1, 1])
            out = tf.nn.bias_add(conv, self.biases(128))
            conv2_1 = self.relu(out, name="conv2_1")

            conv = self.conv_2d(conv2_1, filter=self.filter(3, 3, 128, 128), stride=[1, 1, 1, 1])
            out = tf.nn.bias_add(conv, self.biases(128))
            conv2_2 = self.relu(out, name="conv2_2")

            conv2_max_pool=self.max_pool(conv2_2,"max_pool_2")


        with tf.variable_scope('pool3'):

            conv = self.conv_2d(conv2_max_pool, filter=self.filter(3, 3, 128, 256), stride=[1, 1, 1, 1])
            out = tf.nn.bias_add(conv, self.biases(256))
            conv3_1 = self.relu(out, name="conv3_1")

            conv = self.conv_2d(conv3_1, filter=self.filter(3, 3, 256, 256), stride=[1, 1, 1, 1])
            out = tf.nn.bias_add(conv, self.biases(256))
            conv3_2 = self.relu(out, name="conv3_2")

            conv = self.conv_2d(conv3_2, filter=self.filter(3, 3, 256, 256), stride=[1, 1, 1, 1])
            out = tf.nn.bias_add(conv, self.biases(256))
            conv3_3 = self.relu(out, name="conv3_3")

            conv3_max_pool=self.max_pool(conv3_3,"max_pool_3")


        with tf.variable_scope('pool4'):
            conv = self.conv_2d(conv3_max_pool, filter=self.filter(3, 3, 256, 512), stride=[1, 1, 1, 1])
            out = tf.nn.bias_add(conv, self.biases(512))
            conv4_1 = self.relu(out, name="conv4_1")

            conv = self.conv_2d(conv4_1, filter=self.filter(3, 3, 512, 512), stride=[1, 1, 1, 1])
            out = tf.nn.bias_add(conv, self.biases(512))
            conv4_2 = self.relu(out, name="conv4_2")

            conv = self.conv_2d(conv4_2, filter=self.filter(3, 3, 512, 512), stride=[1, 1, 1, 1])
            out = tf.nn.bias_add(conv, self.biases(512))
            conv4_3 = self.relu(out, name="conv4_3")

            conv4_max_pool = self.max_pool(conv4_3, "max_pool_4")


        with tf.variable_scope('pool5'):
            conv = self.conv_2d(conv4_max_pool, filter=self.filter(3, 3, 512, 512), stride=[1, 1, 1, 1])
            out = tf.nn.bias_add(conv, self.biases(512))
            conv5_1 = self.relu(out, name="conv5_1")

            conv = self.conv_2d(conv5_1, filter=self.filter(3, 3, 512, 512), stride=[1, 1, 1, 1])
            out = tf.nn.bias_add(conv, self.biases(512))
            conv5_2 = self.relu(out, name="conv5_2")

            conv = self.conv_2d(conv5_2, filter=self.filter(3, 3, 512, 512), stride=[1, 1, 1, 1])
            out = tf.nn.bias_add(conv, self.biases(512))
            conv5_3 = self.relu(out, name="conv5_3")

            conv5_max_pool = self.max_pool(conv5_3, "max_pool_5")



        with tf.variable_scope('decode-pool5'):
            conv5_upsample=self.max_pool_transpose(conv5_max_pool,2)

            deconv=self.deconv_2d(conv5_upsample,filter=self.filter(3,3,512,512),stride=[1, 1, 1, 1])
            out = tf.nn.bias_add(deconv, self.biases(512))
            deconv5_1 = self.relu(out, name="deconv5_1")

            deconv = self.deconv_2d(deconv5_1, filter=self.filter(3, 3, 512, 512), stride=[1, 1, 1, 1])
            out = tf.nn.bias_add(deconv, self.biases(512))
            deconv5_2 = self.relu(out, name="deconv5_2")

            deconv = self.deconv_2d(deconv5_2, filter=self.filter(3, 3, 512, 512), stride=[1, 1, 1, 1])
            out = tf.nn.bias_add(deconv, self.biases(512))
            deconv5_3 = self.relu(out, name="deconv5_3")


        with tf.variable_scope('decode-pool4'):
            conv4_upsample = self.max_pool_transpose(deconv5_3, 2)

            deconv = self.deconv_2d(conv4_upsample, filter=self.filter(3, 3, 512, 512), stride=[1, 1, 1, 1])
            out = tf.nn.bias_add(deconv, self.biases(512))
            deconv4_1 = self.relu(out, name="deconv4_1")

            deconv = self.deconv_2d(deconv4_1, filter=self.filter(3, 3, 512, 512), stride=[1, 1, 1, 1])
            out = tf.nn.bias_add(deconv, self.biases(512))
            deconv4_2 = self.relu(out, name="deconv4_2")

            deconv = self.deconv_2d(deconv4_2, filter=self.filter(3, 3, 256, 512), stride=[1, 1, 1, 1])
            out = tf.nn.bias_add(deconv, self.biases(256))
            deconv4_3 = self.relu(out, name="deconv4_3")


        with tf.variable_scope('decode-pool3'):
            conv3_upsample = self.max_pool_transpose(deconv4_3, 2)

            deconv = self.deconv_2d(conv3_upsample, filter=self.filter(3, 3, 256, 256), stride=[1, 1, 1, 1])
            out = tf.nn.bias_add(deconv, self.biases(256))
            deconv3_1 = self.relu(out, name="deconv3_1")

            deconv = self.deconv_2d(deconv3_1, filter=self.filter(3, 3, 256, 256), stride=[1, 1, 1, 1])
            out = tf.nn.bias_add(deconv, self.biases(256))
            deconv3_2 = self.relu(out, name="deconv3_2")

            deconv = self.deconv_2d(deconv3_2, filter=self.filter(3, 3, 128, 256), stride=[1, 1, 1, 1])
            out = tf.nn.bias_add(deconv, self.biases(128,"biases3_2"))
            deconv3_3 = self.relu(out, name="deconv3_3")


        with tf.variable_scope('decode-pool2'):
            conv2_upsample = self.max_pool_transpose(deconv3_3, 2)

            deconv2_1 = self.deconv_2d(conv2_upsample, filter=self.filter(3, 3, 128, 128), stride=[1, 1, 1, 1])
            out2_1 = tf.nn.bias_add(deconv2_1, self.biases(128))
            deconv2_1 = self.relu(out2_1, name="deconv2_1")

            deconv = self.deconv_2d(deconv2_1, filter=self.filter(3, 3, 128, 128), stride=[1, 1, 1, 1])
            out = tf.nn.bias_add(deconv, self.biases(128))
            deconv2_2 = self.relu(out, name="deconv2_2")

            deconv = self.deconv_2d(deconv2_2, filter=self.filter(3, 3, 64, 128), stride=[1, 1, 1, 1])
            out = tf.nn.bias_add(deconv, self.biases(64))
            deconv2_3 = self.relu(out, name="deconv2_3")


        with tf.variable_scope('decode-pool1'):
            conv1_upsample = self.max_pool_transpose(deconv2_3, 2)

            deconv = self.deconv_2d(conv1_upsample, filter=self.filter(3, 3,64, 64), stride=[1, 1, 1, 1])
            out = tf.nn.bias_add(deconv, self.biases(64))
            deconv1_1 = self.relu(out, name="deconv1_1")

            deconv = self.deconv_2d(deconv1_1, filter=self.filter(3, 3, 64, 64), stride=[1, 1, 1, 1])
            out = tf.nn.bias_add(deconv, self.biases(64))
            deconv1_2 = self.relu(out, name="deconv1_2")

            deconv = self.deconv_2d(deconv1_2, filter=self.filter(3, 3, 2, 64), stride=[1, 1, 1, 1])
            out = tf.nn.bias_add(deconv, self.biases(2))
            deconv1_3 = self.relu(out, name="deconv1_3")



        tf.nn.softmax(deconv1_3)
        return deconv1_3





