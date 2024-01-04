#!/bin/sh
#SBATCH -A uppmax2020-2-2
#SBATCH -p core -n 4
#SBATCH -t 03-00:00:00 
#SBATCH -M snowy
#SBATCH -J project
#SBATCH --gres=gpu:1

python3.8 run.py --text /home/yaruwu/keyphrase/Test/232.abstr \
              --StanfordCoreNLP_path /home/yaruwu/keyphrase/stanford-corenlp-full-2018-10-05/ \
              --model_path /home/yaruwu/keyphrase/bert-base-uncased \
              --beta 1 \
              --number 15 \
