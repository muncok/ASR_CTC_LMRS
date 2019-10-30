import tensorflow as tf
import numpy as np
import decoder
import time

def decoding(probs, seq_length, beam_width, alpha, beta, lm_file):
    # prob : [batch_size, max_length, 30]
    # seq_length : [batch_size]
    # output : decoded_hyp_argmax [batch_size] 
    #          decoded_hyp_lm     [batch_size]
    char_file = 'char_set_reverse.txt'
    dec_argmax = decoder.ArgmaxDecoder()
    dec_argmax.load_chars(char_file)
    
    dec_lm = decoder.BeamLMDecoder()
    dec_lm.load_chars(char_file)
    dec_lm.load_lm(lm_file)
    
    batch_size = probs.shape[0]
    batch_hyp_argmax = []
    batch_hyp_lm = []
    for k in range(batch_size):
        print(k)
        X = np.asfortranarray(np.transpose(np.log(probs[k])).astype(np.double))
        # cut steps until seq_length
        X = X[:,:seq_length[k]]
        hyp_argmax, _ = dec_argmax.decode(X)
        hyp_argmax = hyp_argmax.replace('#',' ')
        lm_s = time.time()
        hyp_lm, _ = dec_lm.decode(X)
        lm_e = time.time()
        hyp_lm = hyp_lm.replace('#',' ')
        print('decoding w/o lm : '+hyp_argmax+'\ndecoding w/ lm : '+hyp_lm+'\nlm decoding time : '+str(lm_e-lm_s)+'s')
        batch_hyp_argmax.append(hyp_argmax)
        batch_hyp_lm.append(hyp_lm)
    return batch_hyp_argmax, batch_hyp_lm


           
