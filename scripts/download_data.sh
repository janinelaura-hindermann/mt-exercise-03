#! /bin/bash

scripts=$(dirname "$0")
base=$scripts/..

data=$base/data

mkdir -p $data

# tools=$base/tools


# preprocess slightly

cat $data/familyguy/raw/family_guy_dialogues.txt | python $base/scripts/preprocess_raw.py > $data/familyguy/raw/family_guy_dialogues.cleaned.txt

# tokenize, fix vocabulary upper bound

cat $data/familyguy/raw/family_guy_dialogues.cleaned.txt | python $base/scripts/preprocess.py --vocab-size 5000 --tokenize --lang "en" --sent-tokenize > \
    $data/familyguy/raw/family_guy_dialogues.preprocessed.txt


# split into train, valid and test


# validation
head -n 995 $data/familyguy/raw/family_guy_dialogues.preprocessed.txt | tail -n 995 > $data/familyguy/valid.txt

# testing
head -n 1990 $data/familyguy/raw/family_guy_dialogues.preprocessed.txt | tail -n 995 > $data/familyguy/test.txt

# training
tail -n +1991 $data/familyguy/raw/family_guy_dialogues.preprocessed.txt > $data/familyguy/train.txt
