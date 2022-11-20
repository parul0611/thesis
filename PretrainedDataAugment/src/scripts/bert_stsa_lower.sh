#!/usr/bin/env bash

SRC=thesis/PretrainedDataAugment/src
CACHE=thesis/PretrainedDataAugment/CACHE
TASK=sts

for NUMEXAMPLES in 10;
do
    for i in {0..14};
        do
        RAWDATADIR=/content/gdrive/MyDrive/datasets/${TASK}/exp_${i}_${NUMEXAMPLES}

      
 # Baseline classifier
        python $SRC/bert_aug/bert_classifier.py --task $TASK  --data_dir $RAWDATADIR --seed ${i} --learning_rate=1e-4 --cache $CACHE > $RAWDATADIR/bert_baseline.log
 
  #######################
        # GPT2 Classifier
        #######################

        GPT2DIR=$RAWDATADIR/gpt2
        mkdir $GPT2DIR
        python $SRC/bert_aug/cgpt2.py --data_dir $RAWDATADIR --output_dir $GPT2DIR --task_name $TASK  --num_train_epochs 25 --seed ${i} --top_p 0.9 --temp 1.0 --cache $CACHE
        cat $RAWDATADIR/train.tsv $GPT2DIR/cmodgpt2_aug_3.tsv > $GPT2DIR/train.tsv
        cp $RAWDATADIR/test.tsv $GPT2DIR/test.tsv
        cp $RAWDATADIR/dev.tsv $GPT2DIR/dev.tsv
        python $SRC/bert_aug/bert_classifier.py --task $TASK --data_dir $GPT2DIR --seed ${i} --cache $CACHE > $RAWDATADIR/bert_gpt2_3.log

      ##############
      ## EDA
      ##############

      EDADIR=$RAWDATADIR/eda
      mkdir $EDADIR
      python $SRC/bert_aug/eda.py --input $RAWDATADIR/train.tsv --output $EDADIR/eda_aug.tsv --num_aug=1 --alpha=0.1 --seed ${i}
      cat $RAWDATADIR/train.tsv $EDADIR/eda_aug.tsv > $EDADIR/train.tsv
      cp $RAWDATADIR/test.tsv $EDADIR/test.tsv
      cp $RAWDATADIR/dev.tsv $EDADIR/dev.tsv
      python $SRC/bert_aug/bert_classifier.py --task $TASK --data_dir $EDADIR --seed ${i} --learning_rate=1e-4 --cache $CACHE  > $RAWDATADIR/bert_eda.log

#    #######################
    #    # Backtranslation DA Classifier
    #    #######################

    # BTDIR=$RAWDATADIR/bt
    # mkdir $BTDIR
    # python $SRC/bert_aug/backtranslation.py --data_dir $RAWDATADIR --output_dir $BTDIR --task_name $TASK  --seed ${i} --cache $CACHE
    # cat $RAWDATADIR/train.tsv $BTDIR/bt_aug.tsv > $BTDIR/train.tsv
    # cp $RAWDATADIR/test.tsv $BTDIR/test.tsv
    # cp $RAWDATADIR/dev.tsv $BTDIR/dev.tsv
    python $SRC/bert_aug/bert_classifier.py --task $TASK --data_dir $BTDIR --seed ${i} --cache $CACHE  > $RAWDATADIR/bert_bt.log

#    #######################
    #    # Backtranslationtwo DA Classifier
    #    #######################

    # BTTWODIR=$RAWDATADIR/bttwo
    # mkdir $BTTWODIR
    # python $SRC/bert_aug/backtranslationtwo.py --data_dir $RAWDATADIR --output_dir $BTTWODIR --task_name $TASK  --seed ${i} --cache $CACHE
    # cat $RAWDATADIR/train.tsv $BTTWODIR/bt_augtwo.tsv > $BTTWODIR/train.tsv
    # cp $RAWDATADIR/test.tsv $BTTWODIR/test.tsv
    # cp $RAWDATADIR/dev.tsv $BTTWODIR/dev.tsv
    python $SRC/bert_aug/bert_classifier.py --task $TASK --data_dir $BTTWODIR --seed ${i} --cache $CACHE  > $RAWDATADIR/berttwo_bt.log
   # #######################
   # # CMODBERTP Classifier
   # ######################

    CMODBERTPDIR=$RAWDATADIR/cmodbertp
    mkdir $CMODBERTPDIR
    python $SRC/bert_aug/cmodbertp.py --data_dir $RAWDATADIR --output_dir $CMODBERTPDIR --task_name $TASK  --num_train_epochs 10 --seed ${i} --cache $CACHE > $RAWDATADIR/cmodbertp.log
    cat $RAWDATADIR/train.tsv $CMODBERTPDIR/cmodbertp_aug.tsv > $CMODBERTPDIR/train.tsv
    cp $RAWDATADIR/test.tsv $CMODBERTPDIR/test.tsv
    cp $RAWDATADIR/dev.tsv $CMODBERTPDIR/dev.tsv
    python $SRC/bert_aug/bert_classifier.py --task $TASK --data_dir $CMODBERTPDIR --seed ${i}  --cache $CACHE > $RAWDATADIR/bert_cmodbertp.log
 
    done
done


