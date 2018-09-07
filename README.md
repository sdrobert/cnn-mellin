# cnn-mellin
A simple CNN for TIMIT to test the performance of the Mellin convolution

## Instructions

1. Put this directory in `kaldi/egs/timit`
2. Run the `timit/s5` recipe until the line
   ``` bash
   exit 0 # From this point you can run Karel's DNN : local/nnet/run_dnn.sh
   ```
   (right after `tri3` was force-aligned to the training set)
3. In the `timit/s5` directory, run
   ``` bash
   steps/align_fmllr.sh data/dev data/lang exp/tri3 exp/tri3_ali_dev
   ```
   (this generates force-alignment of the dev set, used to control the learning
   rate and early stopping of the network)
4. Make symbolic links to some files in `wsj/s5`, as well as `timit/s5`
   ``` bash
   ln -s ../s5/path.sh .
   ln -s ../s5/cmd.sh .
   ln -s ../../wsj/s5/utils .
   ln -s ../../wsj/s5/steps .
   ```
5. Generate feature files. Here's an example for filter banks
   ``` bash
   for x in train dev test; do
     mkdir -p data/$x
     cp ../s5/data/$x/{glm,spk2gender,spk2utt,stm,text,utt2spk,wav.scp} data/$x
     # you could just copy ../s5/data/$x/feats.scp if you wanted
     steps/make_fbank.sh data/$x exp/make_fbank/$x fbank
     utils/validate_data_dir.sh --no-feats data/$x
   done
   ```
6. 
