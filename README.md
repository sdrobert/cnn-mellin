# cnn-mellin
A simple CNN for TIMIT to test the performance of the Mellin convolution as
an acoustic model

## Instructions

1. Put this directory in _kaldi/egs/timit_
2. Install the necessary python packages
   ``` bash
   # for anaconda
   conda install -c sdrobert --file requirements.txt --yes
   conda install -c sdrobert pydrobert-kaldi
   # for PyPI
   pip install -r requirements.txt
   pip install pydrobert-kaldi
   # the local python package
   python setup.py develop
   ```
3. Run the _timit/s5_ recipe until the line
   ``` bash
   exit 0 # From this point you can run Karel's DNN : local/nnet/run_dnn.sh
   ```
   (right after _tri3_ was force-aligned to the training set). I've added
   `--snip-edges=false` to my _conf/mfcc.conf_ to make it easier to
   experiment with
   [pydrobert-speech](https://github.com/sdrobert/pydrobert-speech)
   features, but this isn't necessary as long as you're consistent (i.e. the
   number of aligned frames matches the number of feature frames).
4. From the _timit/s5_ directory, acquire force-aligned target (senone)
   sequences for the training and development sets. An easy choice is to use
   _tri3_ to align, but if you can also use DNN alignments (which I've done).
   For the former:
   ``` bash
   steps/align_fmllr.sh data/dev data/lang exp/tri3 exp/tri3_ali_dev
   ```
   and for the latter:
   ``` bash
   local/steps/run_dnn.sh  # to actually train the DNN
   steps/nnet/align.sh \
     data-fmllr-tri3/dev \
     data/lang exp/dnn4_pretrain-dbn_dnn \
     exp/dnn4_pretrain-dbn_dnn_ali_dev
   ```
5. Make symbolic links to some files in _wsj/s5_, as well as _timit/s5_
   ``` bash
   # we don't include path.sh because we use our own (don't clobber python)
   ln -s ../s5/cmd.sh .
   ln -s ../s5/local .
   ln -s ../../wsj/s5/utils .
   ln -s ../../wsj/s5/steps .
   ```
6. Generate appropriate feature files. Here's an example for
   speaker-independent f-bank features (called from this directory):
   ``` bash
   for x in train dev test; do
     mkdir -p data/kaldi/fbank/$x
     cp ../s5/data/$x/{glm,spk2gender,spk2utt,stm,text,utt2spk,wav.scp} \
        data/kaldi/fbank/$x/
     # careful! uses conf/fbank.conf
     steps/make_fbank.sh data/kaldi/fbank/$x exp/make_fbank/$x fbank
     compute-cmvn-stats \
      scp:data/kaldi/fbank/$x/feats.scp \
      ark,scp:fbank/cmvn_$x.ark,data/kaldi/fbank/$x/cmvn.scp
     utils/validate_data_dir.sh data/kaldi/fbank/$x
   done
   ```
   If you've installed _pydrobert-speech_, here's a speaker-independent
   Gammatone filter bank you could use:
   ``` bash
   for x in train dev test; do
     mkdir -p data/kaldi/tonebank/$x
     cp ../s5/data/$x/{glm,spk2gender,spk2utt,stm,text,utt2spk,wav.scp} \
        data/kaldi/tonebank/$x/
     stepsext/make_pybank.sh \
        --pybank-json conf/tonebank_41.json \
        data/kaldi/tonebank/$x \
        exp/make_tonebank/$x \
        tonebank
     compute-cmvn-stats \
        scp:data/kaldi/tonebank/$x/feats.scp \
        ark,scp:tonebank/cmvn_$x.ark,data/kaldi/tonebank/$x/cmvn.scp
     utils/validate_data_dir.sh data/kaldi/tonebank/$x
   done
   ```
   Alternatively, you can use speaker-dependent FMLLR features from the DNN
   recipe (if you called _local/steps/run_dnn.sh_):
7. Convert your kaldi feature and alignments to a format torch can use. Let's
   say you're using _tri3_ alignments and the f-bank features from above. You
   can combine them as follows:
   ``` bash
   for x in train dev test; do
     mkdir -p data/torch/fbank/$x
     write-table-to-torch-dir \
      "ark:apply-cmvn 'data/kaldi/fbank/train/cmvn.scp' 'scp:data/kaldi/fbank/$x/feats.scp' ark:- |" \
      data/torch/fbank/$x/feats
   done
   write-table-to-torch-dir \
    -i i \
    "ark:ali-to-pdf ../s5/exp/tri3/final.mdl 'ark:gunzip -c ../s5/tri3_ali/ali.*.gz |' ark:- |" \
    data/torch/fbank/train/ali
   write-table-to-torch-dir \
    -i i \
    "ark:ali-to-pdf ../s5/exp/tri3/final.mdl 'ark:gunzip -c ../s5/tri3_ali_dev/ali.*.gz |' ark:- |" \
    data/torch/fbank/dev/ali
   get-torch-spect-data-dir-info --strict data/torch/fbank/dev /dev/null
   get-torch-spect-data-dir-info --strict data/torch/fbank/test /dev/null
   get-torch-spect-data-dir-info \
    --strict data/torch/fbank/train data/torch/fbank/info.ark
   num_targets=$(hmm-info ../s5/exp/tri3/final.mdl | awk '/pdfs/ {print $4}')
   target-count-info-to-tensor \
    --num-targets ${num_targets} \
    data/torch/fbank/info.ark inv_weights data/torch/fbank/weights.pt
   target-count-info-to-tensor \
    --num-targets ${num_targets} \
    data/torch/fbank/info.ark log_prior data/torch/fbank/log_prior.pt
   echo "\
   freq_dim=${num_targets}
   target_dim=$(awk '$1 == "num_filts" {print $2}' data/torch/fbank/info.ark)
   HCLG='$(cd ../s5/exp/tri3/graph; pwd -P)/HCLG.fst'
   gmm_mdl='$(cd ../s5/exp/tri3; pwd -P)/final.mdl'
   weight_file='$(pwd -P)/data/torch/fbank/weights.pt'
   log_prior='$(pwd -P)/data/torch/fbank/log_prior.pt'
   train_data='$(pwd -P)/data/torch/train'
   dev_data='$(pwd- P)/data/torch/dev'
   test_data='$(pwd -p)/data/torch/test'
   " > data/torch/fbank/variables
   ```
   The process is the same for any combination of features and alignments
   (assuming the number of alignment frames and feature frames match).
8. Construct an experiment matrix (see below)
9. Train the models (see below)
10. Run the forward step on the test partition (see below)
11. Call `stepsext/decode_cnn_mellin.sh`
12. Call `bash RESULTS` to list phone error rates

## Acoustic modelling
The goal of the _cnn-mellin_ python package is to train and build acoustic
models. The package doesn't really care about how the features are generated or
the decoding process -- it doesn't care about Kaldi at all!

To see how to configure the acoustic model, call `print-parameters-as-ini`.
This will print the default hyperparameter values as well as any help text
they include. The only parameter values that will not work out-of-the box are
`freq_dim` and `target_dim` under the `[model]` section. These are the
input and output dimensions of the acoustic model, respectively. If you
followed the prerequisites, you can determine these values from
_data/torch/fbank/variables_.

Store your hyperparameters in a config file and feed it into your _cnn-mellin_
commands using the `--config` flag. The hyperparameters don't encode any
hard paths by default, making the models and decoding process relocatable.

After that, training a single model is fairly simple. Assuming you've put your
configuation into _conf/amodel.ini_, and you want to train on f-bank features,
you could try
``` bash
train-acoustic-model \
  --config conf/amodel.ini \
  --state-csv-file exp/amodel/state.csv \
  exp/amodel/states \
  data/torch/fbank/train \
  data/torch/fbank/dev
```
Model and optimizer states would be stored in _exp/amodel/states_, and the
experiment history would be stored in _exp/amodel/state.csv_.

When it comes time to decode, we have to get emission probabilities from the
acoustic model. This can be achieved, for example, by
``` bash
# forward step using the model state with the lowest validation error
acoustic-model-forward-pdfs \
  --config conf/amodel.ini \
  --pdfs-dir exp/amodel/pdfs \
  data/torch/fbank/log_prior.pt \
  data/torch/fbank/test \
  history exp/amodel/states exp/amodel/state.csv
```
From there, we would convert the contents of the directory `exp/amodel/pdfs`
into a kaldi table using the command `write-torch-dir-to-table`, then do the
standard TIMIT decoding and scoring, generating lattices from that table.

## Model matrix
Model matrix is a method of experimentation that simplifies the training
and decoding steps for experimentation with repeated trials and configurations.
The method assumes a specific structure of the _kaldi/egs/timit/cnn-mellin_
directory:
```
cnn-mellin/
  data/torch/
    <feats_1>/
      variables
      train
      dev
      test
    <feats_2>/
      ...
    ...
  exp/
    matrix
    <feats_1>/
      <config_1>/
        1/
          variables
          ...
        2/
          ...
        ...
      <config_2>/
        1/
          ...
        2/
          ...
    <feats_2>/
      ...
    ...
```
We set up each experimental trial in a subdirectory
_exp/feat_name/config_name/trial_number_, which contains a file called
_variables_ that contains bash variables set to experiment specifics. We can
generate this matrix using the script _stepsext/generate_matrix.sh_
