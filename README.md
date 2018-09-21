# cnn-mellin
A simple CNN for TIMIT to test the performance of the Mellin convolution

## Instructions

1. Put this directory in `kaldi/egs/timit`
2. Run the `timit/s5` recipe until the line
   ``` bash
   exit 0 # From this point you can run Karel's DNN : local/nnet/run_dnn.sh
   ```
   (right after `tri3` was force-aligned to the training set)
3. Make symbolic links to some files in `wsj/s5`, as well as `timit/s5`
   ``` bash
   ln -s ../s5/path.sh .
   ln -s ../s5/cmd.sh .
   ln -s ../../wsj/s5/utils .
   ln -s ../../wsj/s5/steps .
   ```
4. Install the local python package, preferably in a conda environment
   ``` bash
   # TODO: add conda requirements
   python setup.py develop
   ```
5. Generate feature files and get them ready for pytorch. Here's an example for
   SI filter banks
   ``` bash
   for x in train dev test; do
     mkdir -p data/$x/torch
     cp ../s5/data/$x/{glm,spk2gender,spk2utt,stm,text,utt2spk,wav.scp} data/$x
     steps/make_fbank.sh data/$x exp/make_fbank/$x fbank
     # utterance-level cmvn
     compute-cmvn-stats \
      scp:data/$x/feats.scp \
      ark,scp:fbank/cmvn_$x.ark,data/$x/cmvn.scp
     utils/validate_data_dir.sh --no-feats data/$x
     write-table-to-torch-dir \
      "scp:apply-cmvn 'data/train/cmvn.scp' 'scp:data/$x/feats.scp' |" \
      data/$x/torch/feats
   done
   write-table-to-torch-dir \
    "ark:ali-to-pdf ../s5/exp/tri3/final.mdl 'ark:gunzip -c ../s5/tri3_ali/ali.*.gz |' ark:- |" \
    data/train/torch/ali
   ```
