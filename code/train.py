import warnings
warnings.filterwarnings("ignore")
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import cv2
from auto_encoder import Model
from keras import backend as K
from helper import DataGenerator
from helper import Huffman_Encoder, Huffman_Decoder
import pickle
K.set_learning_phase(1) #set learning phase for the model

#original 1080P Football video
height = 1080
width = 1920
set_gradient_checkpointing = True #allows for easier traiing of model

#DataGenerator -- create object with residual (raw - compressed) and compressed values
path = "./data/Football"
data = DataGenerator(path, resize=False, height=height, width=width) #get residual data between raw and compressed footage
data_size = data.data_size
epochs = 10
batch_size = 10

#auto-encoder model, 32 channel for H.264
model = Model(C=32,height=height,width=width,gradient_checkpointing=set_gradient_checkpointing)
#tf sessions
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in range(epochs):
    print("Epoch %d started" %(epoch+1))
    loss = 0
    batch_num = int(np.ceil(data_size/batch_size))
    for i in range(batch_num):
        x_batch, _ = data.next_batch(batch_size)
        x_batch /= 255.
        x_batch = x_batch.astype(np.float32)
        if set_gradient_checkpointing:
            discrete = sess.run(model.discrete, {model.x_batch: x_batch})
            loss_, g_Hardtanh, _ = sess.run([model.loss, model.g_Hardtanh, model.train_decoder], {model.discrete: discrete, model.y_batch: x_batch})
            sess.run(model.train_encoder, {model.x_batch: x_batch, model.g_Hardtanh: g_Hardtanh})
        else:
            loss_, _ = sess.run([model.loss, model.train_op], {model.x_batch: x_batch, model.y_batch: x_batch})
        loss += loss_
    print("epoch %d, loss = %f" % (epoch+1, loss/batch_num))
    encoder_weights = sess.run(model.encoder_params)
    decoder_weights = sess.run(model.decoder_params)
    with open("./saved_model/encoder_weights.npy", 'wb') as f:
        np.save(f, encoder_weights)
    with open("./saved_model/decoder_weights.npy", 'wb') as f:
         np.save("./saved_model/decoder_weights.npy", decoder_weights)

    print("model saved")

print("Auto-Encoder Trained -- model saved -- weights for encoding and decoding generated")
