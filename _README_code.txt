# TODOs:
    -train RNN and CNN
    -eval T, R, and C
    -sample; demo-able
    -log_file to training, if time & necessary
        # use for training-speed comparison (model file size for memory comparison)
    

# Transformer, CNN, and RNN models
    ## preprocessing, training, & sampling
    

# 1. preprocess data
onmt_preprocess\
 -train_src ../../GYAFC_Corpus/CAT/train/informal_cat     \
 -train_tgt ../../GYAFC_Corpus/CAT/train/formal_cat       \
 -valid_src ../../GYAFC_Corpus/CAT/test/informal_cat      \
 -valid_tgt ../../GYAFC_Corpus/CAT/test/formal_cat        \
 -save_data demo
 
 
 # 2 TRAIN
 
 ## 2a train Transformer
 onmt_train -data demo -save_model demo-model_transformer                                       \
        -layers 6 -rnn_size 512 -word_vec_size 512 -transformer_ff 2048 -heads 8                \
        -encoder_type transformer -decoder_type transformer -position_encoding                  \
        -train_steps 200000  -max_generator_batches 2 -dropout 0.1                              \
        -batch_size 4096 -batch_type tokens -normalization tokens  -accum_count 2               \
        -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 8000 -learning_rate 2    \
        -max_grad_norm 0 -param_init 0  -param_init_glorot                                      \
        -label_smoothing 0.1 -valid_steps 10000 -save_checkpoint_steps 2000                     \
        -world_size 1                                                                           \
        -gpu_ranks 0 \
        -log_file training_logs_transformer


## 2b train CNN
onmt_train -data demo -save_model model_cnn \
    --encoder_type cnn \
    --decoder_type cnn \
    -train_steps 20000 \
    -save_checkpoint_steps 2000 \
    -world_size 1 \
    -gpu_ranks 0 \
    -log_file training_logs_cnn
    
## 2b train CNN 2
onmt_train -data demo -save_model model_cnn_2 \
    --encoder_type cnn \
    --decoder_type cnn \
    -train_steps 20000 \
    -save_checkpoint_steps 2000 \
    -world_size 1 \
    -gpu_ranks 0 \
    -log_file training_logs_cnn_2
    

## 2c train RNN
onmt_train -data demo -save_model model_rnn \
    -train_steps 20000 \
    -save_checkpoint_steps 2000 \
    -world_size 1 \
    -gpu_ranks 0 \
    -log_file training_logs_rnn


# 3 SAMPLE

##  3a sample Transformer
onmt_translate                  \
    -model demo-model_transformer_step_20000.pt  \
    -src informal_input             \
    -output formal_output_transformer_20000           \
    -replace_unk -verbose
 
 ## 3b sample CNN
 ## NOTE for this one can only sample batch size 1; ie 1 line at a time; bug in OpenNMT
 onmt_translate \
    -model model_cnn_step_20000.pt  \
    -src informal_input             \
    -output formal_output_cnn_20000 \
    -replace_unk -verbose
    
 ## 3b sample CNN_2
 ## NOTE for this one can only sample batch size 1; ie 1 line at a time; bug in OpenNMT
 onmt_translate \
    -model model_cnn_2_step_20000.pt  \
    -src informal_input             \
    -output formal_output_cnn_2_20000 \
    -replace_unk -verbose
    
    
 ## 3c sample RNN
  onmt_translate \
    -model model_rnn_step_20000.pt  \
    -src informal_input             \
    -output formal_output_rnn_20000 \
    -replace_unk -verbose
    
 ## TODO 236 BERT-TRANSFORMER for --encoder_type, in branched opennmt-py
    
# 4 EVAL

## 4a EVAL Transformer
    # acc, ppl, xent:
    # "train" from prev_model, save every 1 step

 onmt_train -data demo -save_model demo-model_transformer                                       \
        -layers 6 -rnn_size 512 -word_vec_size 512 -transformer_ff 2048 -heads 8                \
        -encoder_type transformer -decoder_type transformer -position_encoding                  \
        -max_generator_batches 2 -dropout 0.1  \
        -batch_size 4096 -batch_type tokens -normalization tokens  -accum_count 2               \
        -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 8000 -learning_rate 2    \
        -max_grad_norm 0 -param_init 0  -param_init_glorot                                      \
        -label_smoothing 0.1 \
        -valid_steps 1 -save_checkpoint_steps 2000 -train_steps 200000 \
        -world_size 1                                                                           \
        -gpu_ranks 0 \
        -train_from demo-model_transformer_step_20000.pt \
        -log_file training_logs_transformer

    # bleu, rouge # if calling from within opennmt
    onmt_translate                  \
        -model demo-model_transformer_step_20000.pt  \
        -src ../../GYAFC_Corpus/CAT/holdout/subset_informal \
        -tgt ../../GYAFC_Corpus/CAT/holdout/subset_formal \
        -replace_unk \
        -report_bleu  \
        -report_rouge \
        # don't want --report_time
        # too long? # -output formal_output_transformer_20000 \
        # unrecognized for some reason # --report_align \

    # because bleu, rouge, are called with "/../../../tools/(...)", call from within deeper folder...
    onmt_translate                  \
        -model ../demo-model_transformer_step_20000.pt  \
        -src ../../../GYAFC_Corpus/CAT/holdout/subset_informal \
        -tgt ../../../GYAFC_Corpus/CAT/holdout/subset_formal \
        -replace_unk \
        -report_bleu  \
        -report_rouge \
        # don't want --report_time
        # too long? # -output formal_output_transformer_20000 \
        # unrecognized for some reason # --report_align \
    
## 4b EVAL CNN
onmt_translate                  \
    -model model_cnn_step_20000.pt  \
    -src ../../GYAFC_Corpus/CAT/holdout/informal \
    -tgt ../../GYAFC_Corpus/CAT/holdout/formal \
    -output formal_output_cnn_20000           \
    -replace_unk  \
    -report_bleu  \
    -report_rouge \
    -report_time
    
## 4c EVAL RNN
onmt_translate \
    -model model_rnn_step_20000.pt \
    -src ../../GYAFC_Corpus/CAT/holdout/informal \
    -tgt ../../GYAFC_Corpus/CAT/holdout/formal   \
    -replace_unk  \
    -report_bleu  \
    -report_rouge \
    # -report_time
    # -output formal_output_rnn_20000              \





# RNN with pre-trained GloVe encodings and back-translation.

# PRETRAINING
# A: informal -> formal
    model_rnn_formal_to_informal_step_20000.pt

# B: formal -> informal
    model_rnn_informal_to_formal_step_20000.pt
    
    # pre-process:
    onmt_preprocess\
     -train_src ../../GYAFC_Corpus/CAT/train/formal_cat   \
     -train_tgt ../../GYAFC_Corpus/CAT/train/informal_cat \
     -valid_src ../../GYAFC_Corpus/CAT/test/formal_cat    \
     -valid_tgt ../../GYAFC_Corpus/CAT/test/informal_cat  \
     -save_data for_to_inf_data

    # train:
    onmt_train -data for_to_inf_data -save_model model_rnn_formal_to_informal \
    -train_steps 20000 \
    -save_checkpoint_steps 5000 \
    -world_size 1 \
    -gpu_ranks 0 \
    -log_file training_logs_rnn_inf_to_f
    
    
# PRETRAINING 2: w/ GloVe:
# A2: informal -> formal
    # GloVe inf->for embeddings
    ./tools/embeddings_to_torch.py -emb_file_both "glove_dir/glove.6B.100d.txt" \
    -dict_file "demo.vocab.pt" \
    -output_file "embeddings_inf_to_formal"

    # og demo data
    onmt_train -data demo -save_model model_rnn_inf_to_form_GloVe \
    -train_steps 20000 \
    -save_checkpoint_steps 5000 \
    -world_size 1 \
    -gpu_ranks 0 \
    -log_file training_logs_rnn_inf_to_form_GloVe \
    -word_vec_size 100 \
    -pre_word_vecs_enc "embeddings_inf_to_formal.enc.pt" \
    -pre_word_vecs_dec "embeddings_inf_to_formal.dec.pt"
    

# B2: formal -> informal
    # GloVe inf->for embeddings
    ./tools/embeddings_to_torch.py -emb_file_both "glove_dir/glove.6B.100d.txt" \
    -dict_file "for_to_inf_data.vocab.pt" \
    -output_file "embeddings_for_to_informal"
    
    # inf_to_form_data data
    onmt_train -data for_to_inf_data -save_model model_rnn_for_to_inf_GloVe \
    -train_steps 20000 \
    -save_checkpoint_steps 5000 \
    -world_size 1 \
    -gpu_ranks 0 \
    -log_file training_logs_rnn_for_to_inf_GloVe \
    -word_vec_size 100 \
    -pre_word_vecs_enc "embeddings_for_to_informal.enc.pt" \
    -pre_word_vecs_dec "embeddings_for_to_informal.dec.pt"
    
    
# BACK-TRANSLATION
  onmt_translate \
    -model model_rnn_for_to_inf_GloVe_step_20000.pt  \
    -src ../../GYAFC_Corpus/CAT/train/formal_cat   \
    -output ../../GYAFC_Corpus/CAT/train/formal_back \
    -replace_unk -verbose
    
    
# FINAL RNN 
# PROCESS back_data 
   onmt_preprocess\
     -train_src ../../GYAFC_Corpus/CAT/train/formal_cat_cat   \
     -train_tgt ../../GYAFC_Corpus/CAT/train/informal_back_cat_cat \
     -valid_src ../../GYAFC_Corpus/CAT/test/formal_cat    \
     -valid_tgt ../../GYAFC_Corpus/CAT/test/informal_cat  \
     -save_data for_to_inf_data_backed

#TRAIN
onmt_train -data for_to_inf_data_backed -save_model model_rnn_inf_to_form_GloVe_final \
    -enc_layers 4 \
    -dec_layers 4 \
    -train_steps 50000 \
    -save_checkpoint_steps 10000 \
    -world_size 1 \
    -gpu_ranks 0 \
    -log_file training_logs_rnn_inf_to_form_GloVe_final \
    -word_vec_size 100 \
    -pre_word_vecs_enc "embeddings_inf_to_formal.enc.pt" \
    -pre_word_vecs_dec "embeddings_inf_to_formal.dec.pt"
    
    
## 4 EVAL FINAL RNN
onmt_translate \
    -model model_rnn_inf_to_form_GloVe_final_step_50000.pt \
    -src ../../GYAFC_Corpus/CAT/holdout/informal \
    -tgt ../../GYAFC_Corpus/CAT/holdout/formal   \
    -replace_unk  \
    -report_bleu  \
    -report_rouge \
    # -report_time
    # -output formal_output_rnn_20000              \
    
    
 ## 5 SAMPLE FINAL RNN
   onmt_translate \
    -model model_rnn_inf_to_form_GloVe_final_step_40000.pt  \
    -src input_informal            \
    -output formal_output_rnn_inf_to_form_GloVe_final_step_50000 \
    -replace_unk -verbose
    
    
 ## 5 b SAMPLE GLOVE RNN FORWARD
 onmt_translate \
    -model model_rnn_inf_to_form_GloVe_step_20000.pt  \
    -src input_informal            \
    -output formal_output
    -replace_unk -verbose
    
    
 ## 5 C SAMPLE RNN
 
 ## 5 D SAMPLE TRANSFORMER 
 
 onmt_translate                  \
    -model demo-model_transformer_step_20000.pt  \
    -src input_informal             \
    -output formal_output_transformer_20000           \
    -replace_unk -verbose
    