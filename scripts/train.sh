#! /bin/bash

scripts=$(dirname "$0")
base=$(realpath $scripts/..)

models=$base/models
data=$base/data
tools=$base/tools
logs=$base/logs

mkdir -p $models
mkdir -p $logs

num_threads=4
device=""

SECONDS=0

(
  cd $tools/pytorch-examples/word_language_model &&
    CUDA_VISIBLE_DEVICES=$device OMP_NUM_THREADS=$num_threads python3 main.py \
      --data $data/familyguy \
      --epochs 40 \
      --log-interval 10 \
      --emsize 200 --nhid 200 --dropout 0.8 --tied \
      --save $models/model.pt \
      --mps \
      --save-perplexities \
      --log-file $logs/perplexities_0.8.txt
)
echo "time taken:"
echo "$SECONDS seconds"
