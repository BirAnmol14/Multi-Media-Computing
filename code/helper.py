import warnings
warnings.filterwarnings("ignore")
import numpy as np
np.random.seed(0)
import cv2
from dahuffman import HuffmanCodec

class DataGenerator(object):
    def __init__(self, path, shuffle=False, resize=False, height=None, width=None):
        if (shuffle == True):
            raise Exception("Shuffle not defined")
        #raw file
        self.raw_file = path + "_raw.mp4"
        #H.264 compressed video
        self.com_file = path + "_compressed.mp4"
        self.raw = cv2.VideoCapture(self.raw_file)
        self.com = cv2.VideoCapture(self.com_file)
        if(not self.raw.isOpened()):
            raise Exception(self.raw_file + ": read error")
        if(not self.com.isOpened()):
            raise Exception(self.com_file + ": read error")
        self._epochs_completed = 0
        self.data_size = self.raw.get(cv2.CAP_PROP_FRAME_COUNT)
        self.shuffle = shuffle
        self.resize = resize
        if(resize):
            self.height = height
            self.width = width
        else:
            self.height = self.raw.get(cv2.CAP_PROP_FRAME_HEIGHT)
            self.width = self.raw.get(cv2.CAP_PROP_FRAME_WIDTH)

    def next_batch(self, batch_size):
        """Return the next batch_size examples from this data set."""
        compressed_data = []
        residual_data = []
        index = 0
        assert batch_size <= self.data_size
        while index < batch_size:
            r_ret, r_frame = self.raw.read()
            c_ret, c_frame = self.com.read()
            if(r_ret and c_ret):
                if(self.resize):
                    r_frame = cv2.resize(r_frame, (self.width, self.height),
                                         interpolation=cv2.INTER_CUBIC)
                    c_frame = cv2.resize(c_frame, (self.width, self.height),
                                         interpolation=cv2.INTER_CUBIC)
                #cast to float, avoid overflow
                residual_frame = r_frame.astype(float) - c_frame.astype(float)
                compressed_data.append(c_frame.astype(float))
                residual_data.append(residual_frame)
                index += 1
            else:
                self._epochs_completed += 1
                self.raw.release()
                self.com.release()
                self.raw = cv2.VideoCapture(self.raw_file)
                self.com = cv2.VideoCapture(self.com_file)
                if(not self.raw.isOpened()):
                    raise Exception(self.raw_file + ": read error")
                if(not self.com.isOpened()):
                    raise Exception(self.com_file + ": read error")
        residual_data = np.asarray(residual_data)
        compressed_data = np.asarray(compressed_data)
        return residual_data, compressed_data

def float_to_binary(X):
    a = np.asarray(X, dtype=int)
    cond = (a==-1)
    a[cond] = 0
    b = np.asarray(a, dtype=bool)
    return b

def binary_to_float(X):
    b = np.asarray(X, dtype=int)
    cond = (b==0)
    b[cond] = -1
    a = np.asarray(b, dtype=float)
    return a

def Huffman_Encoder(binary_map, bits_group_num = 64):
    binary_map = float_to_binary(binary_map)
    map_length = len(binary_map)
    if map_length%bits_group_num:
        bools_list = list(binary_map[:-(map_length%bits_group_num)].reshape(-1, bits_group_num))
        bools_list.append(binary_map[-(map_length%bits_group_num):])
    else:
        bools_list = list(binary_map.reshape(-1, bits_group_num))
    bits_string = [b.tobytes() for b in bools_list]
    codec = HuffmanCodec.from_data(bits_string)
    output = codec.encode(bits_string)
    return output, codec

def Huffman_Decoder(input, codec):
    bits_string = codec.decode(input)
    bools_list = [np.frombuffer(b, dtype=np.bool) for b in bits_string]
    binary_map = [bool for bools in bools_list for bool in bools]
    return binary_to_float(np.asarray(binary_map))
