# Code for the paper "Learning to Act by Predicting the Future" by Alexey Dosovitskiy and Vladlen Koltun

If you use this code or the provided environments in your research, please cite the following paper:

    @inproceedings{DK2017,
    author    = {Alexey Dosovitskiy and Vladlen Koltun},
    title     = {Learning to Act by Predicting the Future},
    booktitle = {International Conference on Learning Representations (ICLR)},
    year      = {2017}
    }

## Content

- master branch contains the algorithm implementation, example experiment configuration files and ViZDoom environment definitions (.cfg and .wad) - around 10Mb in total
- pretrained_models branch additionally contains pre-trained models for all examples, and corresponding log files - around 200 Mb in total
- Clone only the master branch to avoid extra traffic

## Dependencies:
- ViZDoom
- numpy
- tensorflow
- OpenCV python bindings
- (optionally, cuda and cudnn)

## Tested with: 
- Ubuntu 14.04
- python 3.4
- tensorflow 1.0
- ViZDoom master branch commit ed25f236ac93fbe7f667d64fe48d733506ce51f4

## Running the code:
- Adjust ViZDoom path in doom_simulator.py
- For testing, switch to the pretrained_models branch and run (using D3 as an example):

        cd examples/D3_battle
        python3 run_exp.py show

- For training, run the following (using D2 as an example):

        cd examples/D3_battle
        python3 run_exp.py train

- If you have multiple gpus, make sure that only one is visible with

        export CUDA_VISIBLE_DEVICES=NGPU

    where NGPU is the number of GPU you want to use, or "" if you do not want to use a gpu

- For speeding things up you may want to prepend "taskset -c NCORE" before the command, where NCORE is the number of the core to be used, for example:

        taskset -c 1 python3 run_exp.py train

  When training with a GPU, one core seems to perform the best. Without a GPU, you may want 4 or 8 cores.

## Remarks

- For experiments in the paper we used a slightly modified ViZDoom version which provided a post-mortem measurement. This turns out to make a difference for training. For this reason, the results with this code and the default ViZDoom version may differ slightly from the results in the paper.

- Results may vary across training runs: in our experiments, up to roughly 15%.

- In battle scenarios, the reward provided by ViZDoom is the number of frags. For training the baseline approaches we did not use this reward, but rather a weighted average of the three measurements, same as for our approach, for a fair comparison.

## Troubleshooting

Please send bug reports to Alexey Dosovitskiy <adosovitskiy@gmail.com>
