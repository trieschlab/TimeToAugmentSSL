#!/bin/bash
ENV=toys
NUM=colorobj_house

#Standard ablation cltt

python3 ../main.py --exp_name $ENV$NUM"_rot" $@ --focus 0 --elevation 0 --depth 0 --noise 0 --rotate 2
python3 ../main.py --exp_name $ENV$NUM"_rot_depth" $@ --focus 0 --elevation 0 --depth 0.6 --noise 0 --rotate 2
python3 ../main.py --exp_name $ENV$NUM"_rot_depth_focus" $@ --focus 1.4 --elevation 0 --depth 0.6 --noise 0 --rotate 2
python3 ../main.py --exp_name $ENV$NUM"_rot_depth_focus_ele" $@ --focus 0 --elevation 0.5 --depth 0.6 --noise 0 --rotate 2
python3 ../main.py --exp_name $ENV$NUM"_simclrtt_jitter" $@ --resize_crop False --flip False --jitter True --grayscale False --blur False --augmentations combine2 --noise 0
python3 ../main.py --exp_name $ENV$NUM"_simclrtt_jitter_gray" $@ --resize_crop False --flip False --jitter True --grayscale True --blur False --augmentations combine2 --noise 0
python3 ../main.py --exp_name $ENV$NUM"_simclrtt_jitter_gray_crop" $@ --resize_crop True --flip False --jitter True --grayscale True --blur False --augmentations combine2 --noise 0
python3 ../main.py --exp_name $ENV$NUM"_simclrtt_simclr" $@ --resize_crop True --flip True --jitter True --grayscale True --blur True --augmentations combine2 --noise 0

#Main experiments
python3 ../main.py --exp_name $ENV$NUM"_simclrtt_1tp01_tel_noel_nofocus_rot360" $@ --agent ordturn_1_0.1 --augmentations time --noise 0 --elevation 0 --focus 0 --teleport True --max_angle_speed 360
python3 ../main.py --exp_name $ENV$NUM"_simclrtt_1tp01_tel_noel_nofocus_rot360_simclr" $@ --agent ordturn_1_0.1 --augmentations combine2 --noise 0 --elevation 0 --focus 0 --teleport True --max_angle_speed 360 --jitter True --grayscale True --resize_crop True
python3 ../main.py --exp_name $ENV$NUM"_simclrtt_bio_simclr" $@ --agent ordturn_2_0.1 --augmentations combine2 --noise 0 --elevation 0 --focus 0 --teleport False --max_angle_speed 30
python3 ../main.py --exp_name $ENV$NUM"_simclr_1tp01_tel" $@ --augmentations standard2 --blur True --resize_crop True --jitter True --grayscale True --flip True --max_angle_speed 30 --rotate 2 --focus 0 --elevation 0 --depth 0 --noise 0 --agent ordturn_1_0.1 --teleport True
python3 ../main.py --exp_name $ENV$NUM"_simclr" $@ --augmentations standard2 --blur True --resize_crop True --jitter True --grayscale True --flip True --max_angle_speed 30 --rotate 2 --focus 0 --elevation 0 --depth 0 --noise 0 --agent ord_10

#Motion speed ablation
python3 ../main.py --exp_name $ENV$NUM"_simclrtt_20tp01" $@ --agent ordturn_20_0.1 --augmentations time
python3 ../main.py --exp_name $ENV$NUM"_simclrtt_5tp01" $@ --agent ordturn_5_0.1 --augmentations time
python3 ../main.py --exp_name $ENV$NUM"_simclrtt_1tp01" $@ --agent ordturn_1_0.1 --augmentations time
python3 ../main.py --exp_name $ENV$NUM"_simclrtt_1tp01_tel" $@ --agent ordturn_1_0.1 --augmentations time --noise 0 --teleport True
python3 ../main.py --exp_name $ENV$NUM"_simclrtt_1tp01_tel_jitgrayflipblur" $@ --agent ordturn_1_0.1 --augmentations combine2 --noise 0 --teleport True --max_angle_speed 30 --jitter True --grayscale True --resize_crop False
python3 ../main.py --exp_name $ENV$NUM"_simclrtt_jitgrayflipblur" $@ --agent ord_10 --augmentations combine2 --noise 0 --teleport False --max_angle_speed 30 --jitter True --grayscale True --resize_crop False

#Saccades ablation
python3 ../main.py --exp_name $ENV$NUM"_depth_rot360_1tp01_saco_simclr_nocrop" $@ --agent ordturn_1_0.1 --augmentations combine2 --noise 0 --elevation 0 --focus 0 --teleport True --max_angle_speed 360 --jitter True --grayscale True --resize_crop False --noise -1
python3 ../main.py --exp_name $ENV$NUM"_depth_rot360_1tp01_saco_simclr_crop" $@ --agent ordturn_1_0.1 --augmentations combine2 --elevation 0 --focus 0 --teleport True --max_angle_speed 360 --jitter True --grayscale True --resize_crop True --noise -1 --teleport True
python3 ../main.py --exp_name $ENV$NUM"_depth_rot360_1tp01_nosaco_simclr_nocrop" $@ --agent ordturn_1_0.1 --augmentations combine2 --elevation 0 --focus 0 --teleport True --max_angle_speed 360 --jitter True --grayscale True --resize_crop False --noise 0 --teleport True
python3 ../main.py --exp_name $ENV$NUM"_depth_rot360_1tp01_nosaco_simclr_crop" $@ --agent ordturn_1_0.1 --augmentations combine2 --elevation 0 --focus 0 --teleport True --max_angle_speed 360 --jitter True --grayscale True --resize_crop True --noise 0 --teleport True
