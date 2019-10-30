import struct
import numpy as np
import tensorflow as tf
from nltk.metrics.distance import edit_distance

# 1. Read phntable and phngroup and fead_train

def file_to_list(input_file):
    """
    
    :param input_file: txt file or raw file
    :return: python list
    """
    with open(input_file) as f:
        output = f.readlines()
    output = [x.strip() for x in output]
    return output    

def set_to_list(input_file):
    char_set = file_to_list(input_file)
    list = []
    for i,elem in enumerate(char_set):
        if i != 26:
            list.append(elem.split(' ')[1])
        else:
            list.append(' ')
    return list

def hash_table(input_file):
    read = file_to_list(input_file)
    dict = {}
    for elem in read:
        key = elem.split(' ')[0]
        trans = elem.split(' ')[1:]
        dict[key] = trans
    return dict

def sparse_to_list(batch_size, sparse_tensor_value):
    decoded = []
    num_of_values = len(sparse_tensor_value.values)
    elem_index = []
    index = -1
    for i in range(num_of_values):
        if sparse_tensor_value.indices[i][0] > index:
            for j in range(sparse_tensor_value.indices[i][0]-index):
                #index += 1
                elem_index.append(i)
            index = sparse_tensor_value.indices[i][0]
    for i in range(batch_size-len(elem_index)+1):
        elem_index.append(num_of_values)
    for i in range(batch_size):
        decoded.append(sparse_tensor_value.values[elem_index[i]:elem_index[i+1]])
    return decoded
    
def list_to_char(decoded_list, char_list):
    decoded_str = []
    for idx_list in decoded_list:
        str = []
        concat = ''
        for idx in idx_list:
            str.append(char_list[idx])
        decoded_str.append(concat.join(str))
    return decoded_str

def compute_batch_cer_wer(labels_list, decoded_list):
    batch_size = len(decoded_list)
    cer_list = []
    wer_list = []
    
    for i in range(batch_size):
        cer = edit_distance(labels_list[i], decoded_list[i]) / len(labels_list[i])
        cer_list.append(cer)
        wer = edit_distance(labels_list[i].split(),decoded_list[i].split())/len(labels_list[i].split())
        wer_list.append(wer)
    return np.mean(cer_list), np.mean(wer_list)

#2. Feature parsing

def parse_feat(input_file):
    with open(input_file, "rb") as f:
        header = f.read(12)
        nSamples, _, sampSize, _ = struct.unpack(">iihh", header)
        nFeatures = sampSize // 4
        data = []
        for x in range(nSamples):
            s = f.read(sampSize)
            frame = []
            for v in range(nFeatures):
                val = struct.unpack_from(">f", s, v * 4)
                frame.append(val[0])
            data.append(frame)
        return np.array(data)


def parse_char(feat,dic,char_list):
    
    key = feat.split('/')[-1].split('.')[0]
    trans = dic[key]
    trans_idx = []
    for word in trans:
        for char in word:
            trans_idx.append(char_list.index(char))
        if word != trans[-1]:
            trans_idx.append(26)
    return trans_idx

def load_save_data(feat_train_list, dic, char_list, files, SAVE_DATA):
    if SAVE_DATA:
        data_in = []
        data_trans = []
        feat_train_list= feat_train_list
        for i in range(len(feat_train_list)):
            data_in.append(parse_feat(feat_train_list[i]))
            data_trans.append(parse_char(feat_train_list[i],dic,char_list))
        data_in = np.array(data_in)
        data_trans = np.array(data_trans)
        np.save(files[0], data_in)
        np.save(files[1], data_trans)
    else:
        data_in = np.load(files[0]+'.npy')
        data_trans = np.load(files[1]+'.npy')
    return data_in,data_trans

def make_batch(batch_in_list, batch_out_list):
    feat_dim = batch_in_list[0].shape[1]
    batch_size = batch_in_list.shape[0]
    length = [batch_in_list[i].shape[0] for i in range(batch_size)]
    max_dim = np.max(length)
    length_label = [len(batch_out_list[i]) for i in range(batch_size)]
    max_label_dim = np.max(length_label)
    shape = [batch_size, max_label_dim]
    indices = []
    values = []
    for i, seq in enumerate(batch_out_list):
        indices.extend(zip([i]*len(seq), range(len(seq))))
        values.extend(seq)
    batch_in_pad = np.zeros(shape=[batch_size, max_dim, feat_dim], dtype=np.float32)
    batch_out = tf.SparseTensorValue(indices, values,shape)
    for i in range(batch_size):
        batch_in_pad[i, 0:length[i], :] = batch_in_list[i]
    return np.array(length), batch_in_pad, batch_out

def s_to_chars(s):
    s = ' '.join(s)
    s = s.replace('   ', ' <SPACE> ')
    s = s.upper()
    return s
