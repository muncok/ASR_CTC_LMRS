import tensorflow as tf
import numpy as np
from data_processing import make_batch, sparse_to_list, list_to_char, compute_batch_cer_wer, s_to_chars
from decoding import decoding
import pickle

def evaluate(session, saver, file_name, inputs, labels, seq_length, keep_prob, model, lm_file, cost, prob, batch_size, num_epoch, char_list, beam_width, alpha, beta, LOAD_MODEL, test_in, test_trans):
    print("EVALUATION")
    session.run(tf.global_variables_initializer())
    if LOAD_MODEL:
        saver.restore(session, file_name)
    test_cost = 0
    test_cer = 0
    test_cer_clm = 0
    test_wer = 0
    test_wer_clm = 0
    f = open('decoded.txt','w')
    for i in range(0, test_in.shape[0] // batch_size, 1):
        batch_in = test_in[i * batch_size:(i + 1) * batch_size]
        batch_trans = test_trans[i * batch_size:(i + 1) * batch_size]
        length, batch_x, batch_y = make_batch(batch_in, batch_trans)
        temp_test_cost, temp_test_prob = session.run([cost, prob],
                                                          feed_dict={inputs: batch_x, labels: batch_y,
                                                                     seq_length: length,
                                                                     keep_prob: 1.0})
        test_cost += temp_test_cost
        decoded_argmax, decoded_clm = decoding(temp_test_prob, length, beam_width,alpha, beta, lm_file)
        labels_list = list_to_char(sparse_to_list(batch_size, batch_y), char_list)
        temp_test_cer, temp_test_wer = compute_batch_cer_wer(labels_list, decoded_argmax)
        temp_test_cer_clm, temp_test_wer_clm = compute_batch_cer_wer(labels_list, decoded_clm)
        for k in range(batch_size):
            f.write('decoding w/o lm : '+decoded_argmax[k]+'\n')
            f.write('decoding w/ lm  : '+decoded_clm[k]+'\n')
            f.write('transcription   : '+labels_list[k]+'\n')
        test_cer += temp_test_cer
        test_wer += temp_test_wer
        test_wer_clm += temp_test_wer_clm
        test_cer_clm += temp_test_cer_clm
                
    test_cer /= (test_in.shape[0] // batch_size)
    test_wer /= (test_in.shape[0] // batch_size)
    test_wer_clm /= (test_in.shape[0] // batch_size)
    test_cer_clm /= (test_in.shape[0] // batch_size)
    test_cost /= (test_in.shape[0] // batch_size)
    f.close()
    print('test ctc loss :%.4f, test CER:%.4f, test WER:%.4f, test CER CLM:%.4f, test WER CLM:%.4f' % (test_cost, test_cer, test_wer, test_cer_clm, test_wer_clm))


