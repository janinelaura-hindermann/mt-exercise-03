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
      --epochs 2 \
      --log-interval 100 \
      --emsize 200 --nhid 200 --dropout 0.3 --tied \
      --save $models/model.pt \
      --mps \
      --save-perplexities \
      --log-file $logs/perplexities_0.3.txt \
)
echo "time taken:"
echo "$SECONDS seconds"
