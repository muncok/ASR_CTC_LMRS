import tensorflow as tf
import numpy as np
import kenlm
from data_processing import file_to_list, set_to_list, hash_table, parse_feat, parse_char, load_save_data
from train import train
from evaluate import evaluate
from model.model import Model

char_list = set_to_list('char_set.txt')
key_to_trans = hash_table('train_all.trans')
feat_train_list = file_to_list('train_all.list')
test = True
if test:
    key_to_trans_test = hash_table('test_all.trans')
    feat_test_list = file_to_list('test_all.list')

####################################
###        LOAD DATA             ###
####################################

files = ['data/data_in', 'data/data_trans']
files_wsj0 = ['data/data_wsj0_in', 'data/data_wsj0_trans']
files_wsj1 = ['data/data_wsj1_in', 'data/data_wsj1_trans']
files_test = ['data/data_test_in','data/data_test_trans']
SAVE_DATA = False
print ('DATA LOADING')
if SAVE_DATA:
    data_in, data_trans = load_save_data(feat_train_list, key_to_trans, char_list, files, SAVE_DATA)
    data_wsj0_in = np.save(files_wsj0[0], data_in[0:4195])
    data_wsj0_trans = np.save(files_wsj0[1], data_trans[0:4195])
    data_wsj1_in = np.save(files_wsj1[0], data_in[4195:])
    data_wsj1_trans = np.save(files_wsj1[1], data_trans[4195:])
else:
    if test:
        data_test_in, data_test_trans = load_save_data(feat_test_list, key_to_trans_test, char_list, files_test, True)
    else:
        data_wsj0_in, data_wsj0_trans = load_save_data(feat_train_list, key_to_trans, char_list, files_wsj0, SAVE_DATA)
        #data_wsj1_in_1 , data_wsj1_trans_1 = load_save_data(feat_train_list, key_to_trans, char_list, files_wsj1_1, SAVE_DATA)

if test:
    test_in = data_test_in
    test_trans = data_test_trans
else:
    train_in = data_wsj0_in[0:3000]
    train_trans = data_wsj0_trans[0:3000]
    valid_in = data_wsj0_in[3000:3500]
    valid_trans = data_wsj0_trans[3000:3500]
    test_in = data_wsj0_in[3500:]
    test_trans = data_wsj0_trans[3500:]

####################################
###        Hyperparameters       ###
####################################

learning_rate = 0.0002
opt_name = 'ADAM'
batch_size = 40
num_of_features = 123  # input size
num_labels = 29
output_size = num_labels + 1 # output size : num_classes (num_labels + 1)
rnn_hidden_neurons = 400  # hidden layer num of features
num_of_layers = 3  # the number of Stacked LSTM layers
rnn_type = 'BiLSTM'
recurrent_dropout = False
output_dropout = False
layernorm = True
LOAD_MODEL = True
SAVE_MODEL = False 
MODEL_NAME = rnn_type+str(num_of_layers)+'_'
if layernorm:
    MODEL_NAME = MODEL_NAME + 'ln_'
if recurrent_dropout:
    MODEL_NAME = MODEL_NAME + 'RECURRENT_'
if output_dropout:
    MODEL_NAME = MODEL_NAME + 'VARIATIONAL_'
MODEL_NAME = MODEL_NAME + opt_name + '_'+str(rnn_hidden_neurons)
FILE_NAME = 'params/'+MODEL_NAME
num_epoch = 1 
clip_norm = 5.
beam_width = 50
lm_file = 'wsj_5gram.binary'
alpha = 1.25
beta = 1.5
lm_decoding_step = 100
training = False

def main():
    print(MODEL_NAME)
    with tf.variable_scope('PLACEHOLDER'):
        inputs = tf.placeholder(tf.float32, [None, None, num_of_features], 'inputs')  # (batch, max_dim, in)
        labels = tf.sparse_placeholder(tf.int32)
        seq_length = tf.placeholder(tf.int32, [None], 'seq_length')
        keep_prob = tf.placeholder(tf.float32, None, 'keep_prob')
    model = Model(batch_size, num_of_features, rnn_hidden_neurons, output_size, num_of_layers, recurrent_dropout, output_dropout, keep_prob, layernorm, rnn_type, MODEL_NAME)
    rnn_output, prob = model.get_output(inputs, seq_length) # [max_dim, batch_size, out], prob : [batch_size, max_dim, out]
    cost = model.get_loss(labels, rnn_output, seq_length)
    opt = model.optimize(opt_name, cost, learning_rate, clip_norm)
    
    ###########################
    ###      Train          ###
    ###########################

    sess = tf.Session()
    saver = tf.train.Saver()
    if training:
        train(sess, saver, FILE_NAME, inputs, labels, seq_length, keep_prob, model, lm_file, cost, prob, opt, batch_size, num_epoch, char_list, beam_width, lm_decoding_step, alpha, beta, LOAD_MODEL, SAVE_MODEL, train_in, train_trans, valid_in, valid_trans)
    evaluate(sess, saver, FILE_NAME, inputs, labels, seq_length, keep_prob, model, lm_file, cost, prob, batch_size, num_epoch, char_list, beam_width, alpha, beta, LOAD_MODEL, test_in, test_trans)

if __name__ == '__main__':
    main()

