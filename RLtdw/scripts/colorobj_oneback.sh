#!/bin/bash
ENV=toys
NUM=colorobj_oneback

#Standard ablation cltt

python3 ../main.py --exp_name $ENV$NUM"_rot" $@ --focus 0 --elevation 0 --depth 0 --noise 0 --rotate 2
python3 ../main.py --exp_name $ENV$NUM"_rot_depth" $@ --focus 0 --elevation 0 --depth 0.6 --noise 0 --rotate 2
python3 ../main.py --exp_name $ENV$NUM"_rot_depth_focus" $@ --focus 1.4 --elevation 0 --depth 0.6 --noise 0 --rotate 2
python3 ../main.py --exp_name $ENV$NUM"_rot_depth_focus_ele" $@ --focus 0 --elevation 0.5 --depth 0.6 --noise 0 --rotate 2
python3 ../main.py --exp_name $ENV$NUM"_simclrtt_jitter" $@ --resize_crop False --flip False --jitter True --grayscale False --blur False --augmentations combine2 --noise 0
python3 ../main.py --exp_name $ENV$NUM"_simclrtt_jitter_gray" $@ --resize_crop False --flip False --jitter True --grayscale True --blur False --augmentations combine2 --noise 0
python3 ../main.py --exp_name $ENV$NUM"_simclrtt_jitter_gray_crop" $@ --resize_crop True --flip False --jitter True --grayscale True --blur False --augmentations combine2 --noise 0
python3 ../main.py --exp_name $ENV$NUM"_simclrtt_simclr" $@ --resize_crop True --flip True --jitter True --grayscale True --blur True --augmentations combine2 --noise 0
