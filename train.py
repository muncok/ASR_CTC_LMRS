import tensorflow as tf
import numpy as np
from data_processing import make_batch, sparse_to_list, list_to_char, compute_batch_cer_wer, s_to_chars
from decoding import decoding
import time

def train(session, saver, file_name, inputs, labels, seq_length, keep_prob, model, lm_file, cost, prob, opt, batch_size, num_epoch, char_list, beam_width, lm_decoding_step, alpha, beta, LOAD_MODEL, SAVE_MODEL, train_in, train_trans, valid_in, valid_trans):
    print("TRAINING")
    step = 1
    session.run(tf.global_variables_initializer())
    if LOAD_MODEL:
        saver.restore(session, file_name)
    best_valid_cost = 1000
    while step <= num_epoch:
        # shuffle train_data
        permutation = np.random.permutation(train_in.shape[0])
        train_x = train_in[permutation]
        train_y = train_trans[permutation]
        train_x = train_x[0:1000]
        train_y = train_y[0:1000]
        epoch_cost = 0
        epoch_cer = 0
        epoch_wer = 0
        epoch_cer_clm = 0
        epoch_wer_clm = 0
        cur_valid_cost = 0
        cur_valid_cer = 0
        cur_valid_cer_clm = 0
        cur_valid_wer = 0
        cur_valid_wer_clm = 0
        for i in range(0, train_x.shape[0] // batch_size, 1):
            print(i)
            batch_in = train_x[i * batch_size:(i + 1) * batch_size]
            batch_trans = train_y[i * batch_size:(i + 1) * batch_size]
            length, batch_x, batch_y = make_batch(batch_in, batch_trans)
            # Run Optimization and Calculate cost 
            temp_cost, temp_prob, _ = session.run([cost, prob, opt], feed_dict={inputs: batch_x, labels: batch_y, seq_length: length, keep_prob: 0.9})
            epoch_cost += temp_cost
            if step % lm_decoding_step == 0:
                decoded_argmax, decoded_clm = decoding(temp_prob, length, beam_width, alpha, beta, lm_file)
                decoded_clm = decoding(temp_prob, length, beam_width, alpha, beta, lm_file)
                labels_list = list_to_char(sparse_to_list(batch_size, batch_y), char_list)
                temp_cer, temp_wer = compute_batch_cer_wer(labels_list, decoded_argmax)
                temp_cer_clm, temp_wer_clm = compute_batch_cer_wer(labels_list,decoded_clm)
                epoch_cer += temp_cer
                epoch_wer += temp_wer
                epoch_cer_clm += temp_cer_clm
                epoch_wer_clm +=  temp_wer_clm
        epoch_cost /= (train_x.shape[0] // batch_size)
        epoch_cer /= (train_x.shape[0] // batch_size)
        epoch_wer /= (train_x.shape[0] // batch_size)
        epoch_wer_clm /= (train_x.shape[0] // batch_size)
        epoch_cer_clm /= (train_x.shape[0] // batch_size)
        for i in range(0, valid_in.shape[0] // batch_size, 1):
            batch_in = valid_in[i * batch_size:(i + 1) * batch_size]
            batch_trans = valid_trans[i * batch_size:(i + 1) * batch_size]
            length, batch_x, batch_y = make_batch(batch_in, batch_trans)
            temp_valid_cost, temp_valid_prob = session.run([cost, prob],
                                                          feed_dict={inputs: batch_x, labels: batch_y,
                                                                     seq_length: length,
                                                                     keep_prob: 1.0})
            cur_valid_cost += temp_valid_cost
            if step % lm_decoding_step == 0:
                decoded_argmax, decoded_clm = decoding(temp_valid_prob, length,beam_width, alpha, beta, lm_file)
                decoded_clm = decoding(temp_valid_prob, length, beam_width, alpha, beta, lm_file)
                labels_list = list_to_char(sparse_to_list(batch_size, batch_y), char_list)
                temp_valid_cer, temp_valid_wer = compute_batch_cer_wer(labels_list, decoded_argmax)
                temp_valid_cer_clm, temp_valid_wer_clm = compute_batch_cer_wer(labels_list, decoded_clm)
                cur_valid_cer += temp_valid_cer
                cur_valid_wer += temp_valid_wer
                cur_valid_wer_clm += temp_valid_wer_clm
                cur_valid_cer_clm += temp_valid_cer_clm
                
        cur_valid_cer /= (valid_in.shape[0] // batch_size)
        cur_valid_wer /= (valid_in.shape[0] // batch_size)
        cur_valid_wer_clm /= (valid_in.shape[0] // batch_size)
        cur_valid_cer_clm /= (valid_in.shape[0] // batch_size)
        cur_valid_cost /= (valid_in.shape[0] // batch_size)
        if best_valid_cost >= cur_valid_cost:
            best_valid_cost = cur_valid_cost
            if SAVE_MODEL:
                saver.save(session, file_name)
        print('Epoch %d, train ctc loss:%.4f, valid ctc loss:%.4f, best valid ctc loss:%.4f' % (step, epoch_cost, cur_valid_cost, best_valid_cost))
        if step % lm_decoding_step == 0:
            print('train CER:%.4f, train WER:%.4f, train CER CLM:%.4f, train WER CLM:%.4f\n valid CER:%.4f, valid WER%.4f, valid CER CLM:%.4f, valid WER CLM:%.4f' % (epoch_cer, epoch_wer, epoch_cer_clm, epoch_wer_clm, cur_valid_cer, cur_valid_wer, cur_valid_cer_clm, cur_valid_wer_clm))
        step += 1
    print("Optimization Finished!")


