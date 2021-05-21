import warnings
warnings.filterwarnings("ignore")
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from keras.layers import Activation, Conv2D, BatchNormalization, Lambda
from keras import backend as K
from keras.engine.topology import Layer
import numpy as np
import cv2
import pickle
from helper import DataGenerator
from helper import Huffman_Decoder
K.set_learning_phase(1) #set learning phase


#ref: https://github.com/tetrachrome/subpixel
class Subpixel(Layer):
    def __init__(self, r, **kwargs):
        self.r = r
        super(Subpixel, self).__init__(**kwargs)

    def _phase_shift(self, I, r):
        bsize, a, b, c = I.get_shape().as_list()
        bsize = tf.shape(I)[0] # Handling Dimension(None) type for undefined batch dim
        X = tf.reshape(I, [bsize, a, b, c//(r*r), r, r]) # bsize, a, b, c/(r*r), r, r
        X = tf.transpose(X, (0, 1, 2, 5, 4, 3)) # bsize, a, b, r, r, c/(r*r)
        X = tf.split(X, a, 1)  # a, [bsize, b, r, r, c/(r*r)]
        X = tf.concat([tf.squeeze(x, axis=1) for x in X], 2)  # bsize, b, a*r, r, c/(r*r)
        X = tf.split(X, b, 1)  # b, [bsize, a*r, r, c/(r*r)]
        X = tf.concat([tf.squeeze(x, axis=1) for x in X], 2)  # bsize, a*r, b*r, c/(r*r)
        return X

    def call(self, x):
        return self._phase_shift(x, self.r)

    def compute_output_shape(self, input_shape):
        bsize, a, b, c = input_shape
        return (bsize, a*self.r, b*self.r, c//(self.r*self.r))

class Decoder():
    def __init__(self, C, height, width):
        self.C = C
        self.height = height
        self.width = width
        self.x_batch = tf.placeholder(tf.float32, [None, height, width, C], name = "x_batch")
        self.output = self.build_net()
        self.decoder_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="decoder")

    def build_net(self):
        C = self.C
        height = self.height
        width = self.width

        # decoder layers
        with tf.variable_scope('decoder'):
            r = 2
            conv_4 = Conv2D(r*r*C, (3, 3), strides=(1, 1), input_shape=(height, width, C), padding='same')(self.x_batch)
            sub_4 = Subpixel(r=r)(conv_4)
            bnorm_4 = BatchNormalization()(sub_4)
            relu_4 = Activation('relu')(bnorm_4)

            conv_5 = Conv2D(r*r*C, (3, 3), strides=(1, 1), padding='same')(relu_4)
            sub_5 = Subpixel(r=r)(conv_5)
            bnorm_5 = BatchNormalization()(sub_5)
            relu_5 = Activation('relu')(bnorm_5)

            conv_6 = Conv2D(r*r*3, (3, 3), strides=(1, 1), padding='same')(relu_5)
            output = Subpixel(r=r)(conv_6)
        return output

    def load_weights(self, sess, weight_file):
        weights = np.load(weight_file,allow_pickle=True)
        assert len(weights) == len(self.decoder_params)
        for i in range(len(self.decoder_params)):
            sess.run(self.decoder_params[i].assign(weights[i]))

#original: height = 1080, width = 1920
height=1080
width = 1920
#compression ratio: 8
crr = 8
C = 32
height = int(height/crr)
width = int(width/crr)
bits_group_num = 16
path = "./data/Football"
weight_file = "./saved_model/decoder_weights.npy"
batch_size = 10

model = Decoder(C=C, height=height, width=width)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
model.load_weights(sess, weight_file)

#decode hufman
print("Decoding Huffman")
with open("./saved_model/huffman_codec", 'rb') as f:
    huffman_codec = pickle.load(f)
with open(path+"_residual.npy", 'rb') as f:
    x_encoded = f.read()

x_decoded = Huffman_Decoder(x_encoded, huffman_codec)
x_binary = x_decoded.reshape(-1, height, width, C)

data_size = len(x_binary)
batch_num = int(np.ceil(data_size/batch_size))

print("Writing Residual predicted output")
fourcc1 = cv2.VideoWriter_fourcc(*'MP4V')
out1 = cv2.VideoWriter(path+"_predicted.mp4", fourcc1, 30.0, (1920, 1080))

#predicted residual output
for i in range(data_size):
    #reconstructed residual
    x_residual = sess.run(model.output, {model.x_batch: x_binary[np.newaxis, i]})[0]
    x_residual = np.clip(x_residual, 0, 1)*255.0
    frame = (x_residual).astype(np.uint8)
    out1.write(frame)

out1.release()
print("Stored Football_predicted.mp4 in data folder")

print("Regenerating file")
com_file = path + "_compressed.mp4"
com = cv2.VideoCapture(com_file)
if(not com.isOpened()):
    raise Exception(com_file + ": read error")

fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter(path+"_decoded.mp4", fourcc, 30.0, (1920, 1080))


#residual decoder
for i in range(data_size):
    #reconstructed residual
    x_residual = sess.run(model.output, {model.x_batch: x_binary[np.newaxis, i]})[0]
    x_residual = x_residual / (255.0)
    c_ret, c_frame = com.read()
    if c_ret:
        c_frame = c_frame/255.0
        frame = (np.clip(c_frame+x_residual, 0, 1)*255.0).astype(np.uint8)
        out.write(frame)
        
    else:
        break

out.release()

print("Stored Football_decoded.mp4 in data folder")
