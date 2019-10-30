# ASR_CTC_LMRS
End-to-End ASR using LSTM + CTC + LM Rescoring

Dataset
* wsj0

RNN Model
* 3-layer Bi-directional LSTM (Hidden dimensions: 400)
* Layer Normalization
* CTC loss
* Gradient Clipping (MaxNorm 5)
* Adam Optimizer
* Decreasing Learning Rate (Init 0.01)

Decoding ([Stanford-ctc](https://github.com/amaas/stanford-ctc))
* Greedy Decoding
* Decoding with Character Level Language Model (Maas, Andrew L., et al. "Lexicon-Free Conversational Speech Recognition with Neural Networks." HLT-NAACL. 2015)
  * Beam width: 10
  * Language Model: [Kenlm](https://github.com/kpu/kenlm) (5-gram CLM)
  * Alpha: 1.25, Beta: 1.5
 
Files
* main.py 
* model/model.py - CTC-RNN Model
* data_processing.py - manipulating dataset and compute CER and WER
* decoding.py, decoder.c, wsj_5gram.binary, char_set_reverse.txt - for LM rescoring
* tarin.py
* evaluate.py

Pre-trained Model
[Google drive](https://drive.google.com/drive/folders/1TkEZtcFRocW3cHILxtouhZNl6iLja6dB?usp=sharing)

|Error|value|
|------|---|
|CER w/o LM|12.45%|
|WER w/o LM|44.43%|
|CER w/ LM|6.28%|
|WER w/ LM|18.16%|

Usage
* dataset (wsg0)
  - put wsj0, wsj1 at root directory
* Stanford-ctc
  > python setup.py install
* KenLM
  > pip install https://github.com/kpu/kenlm/archive/master.zip
* main.py
