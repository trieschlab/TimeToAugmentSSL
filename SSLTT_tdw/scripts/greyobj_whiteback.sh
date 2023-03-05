#!/bin/bash
ENV=toys
NUM=greyobj_whiteback

#Standard ablation cltt

python3 ../main.py --exp_name $ENV$NUM"_rot" $@ --focus 0 --elevation 0 --depth 0 --noise 0 --rotate 2
python3 ../main.py --exp_name $ENV$NUM"_rot_depth" $@ --focus 0 --elevation 0 --depth 0.6 --noise 0 --rotate 2
python3 ../main.py --exp_name $ENV$NUM"_rot_depth_focus" $@ --focus 1.4 --elevation 0 --depth 0.6 --noise 0 --rotate 2
python3 ../main.py --exp_name $ENV$NUM"_rot_depth_focus_ele" $@ --focus 0 --elevation 0.5 --depth 0.6 --noise 0 --rotate 2
python3 ../main.py --exp_name $ENV$NUM"_simclrtt_jitter" $@ --resize_crop False --flip False --jitter True --grayscale False --blur False --augmentations combine2 --noise 0
python3 ../main.py --exp_name $ENV$NUM"_simclrtt_jitter_gray" $@ --resize_crop False --flip False --jitter True --grayscale True --blur False --augmentations combine2 --noise 0
python3 ../main.py --exp_name $ENV$NUM"_simclrtt_jitter_gray_crop" $@ --resize_crop True --flip False --jitter True --grayscale True --blur False --augmentations combine2 --noise 0
python3 ../main.py --exp_name $ENV$NUM"_simclrtt_simclr_" $@ --resize_crop True --flip True --jitter True --grayscale True --blur True --augmentations combine2 --noise 0


#Saccades
python3 ../main.py --exp_name $ENV$NUM"_sacc02" $@ --focus 0 --elevation 0 --depth 0 --noise 0.2 --rotate 0
python3 ../main.py --exp_name $ENV$NUM"_sacc5" $@ --focus 0 --elevation 0 --depth 0 --noise 5 --rotate 0
python3 ../main.py --exp_name $ENV$NUM"_sacc4" $@ --focus 0 --elevation 0 --depth 0 --noise 4 --rotate 0
python3 ../main.py --exp_name $ENV$NUM"_sacc3" $@ --focus 0 --elevation 0 --depth 0 --noise 3 --rotate 0
python3 ../main.py --exp_name $ENV$NUM"_sac4_depth" $@ --focus 0 --elevation 0 --depth 0.6 --noise 4 --rotate 0

#rotation ablation
python3 ../main.py --exp_name $ENV$NUM"_s10" $@ --max_angle_speed 10
python3 ../main.py --exp_name $ENV$NUM"_s30" $@ --max_angle_speed 30
python3 ../main.py --exp_name $ENV$NUM"_s90" $@ --max_angle_speed 90
python3 ../main.py --exp_name $ENV$NUM"_s360" $@ --max_angle_speed 360

#Crop versus Depth + saccades
python3 ../main.py --exp_name $ENV$NUM"_simclrc" $@ --augmentations combine2 --blur True --resize_crop True --jitter True --grayscale True --flip True --max_angle_speed 30 --rotate 0 --focus 0 --elevation 0 --depth 0 --noise 0
python3 ../main.py --exp_name $ENV$NUM"_simclrc_nocrop" $@ --augmentations combine2 --blur True --resize_crop False --jitter True --grayscale True --flip True --max_angle_speed 30 --rotate 0 --focus 0 --elevation 0 --depth 0 --noise 0
python3 ../main.py --exp_name $ENV$NUM"_simclrc_nocrop_sacn1" $@ --augmentations combine2 --blur True --resize_crop False --jitter True --grayscale True --flip True --max_angle_speed 30 --rotate 0 --focus 0 --elevation 0 --depth 0.6 --noise -1
python3 ../main.py --exp_name $ENV$NUM"_simclrc_crop_sacn1" $@ --augmentations combine2 --blur True --resize_crop True --jitter True --grayscale True --flip True --max_angle_speed 30 --rotate 0 --focus 0 --elevation 0 --depth 0.6 --noise -1
python3 ../main.py --exp_name $ENV$NUM"_simclrc_nocrop_sac4" $@ --augmentations combine2 --blur True --resize_crop False --jitter True --grayscale True --flip True --max_angle_speed 30 --rotate 0 --focus 0 --elevation 0 --depth 0.6 --noise 4
python3 ../main.py --exp_name $ENV$NUM"_simclrc_crop_sac4" $@ --augmentations combine2 --blur True --resize_crop True --jitter True --grayscale True --flip True --max_angle_speed 30 --rotate 0 --focus 0 --elevation 0 --depth 0.6 --noise 4


#Flips versus rotation ablation
python3 ../main.py --exp_name $ENV$NUM"_rot10" $@ --rotate 2 --max_angle_speed 10
python3 ../main.py --exp_name $ENV$NUM"_rot30" $@ --rotate 2 --max_angle_speed 30
python3 ../main.py --exp_name $ENV$NUM"_rot90" $@ --rotate 2 --max_angle_speed 90
python3 ../main.py --exp_name $ENV$NUM"_rot360" $@ --rotate 2 --max_angle_speed 360
#python3 ../main.py --exp_name $ENV$NUM"_flip" $@ --augmentations standard2 --flip True
#python3 ../main.py --exp_name $ENV$NUM"_flip_rot360" $@ --augmentations combine2 --flip True --max_angle_speed 360 --rotate 2
#python3 ../main.py --exp_name $ENV$NUM"_flip_rot90" $@ --augmentations combine2 --flip True --max_angle_speed 90 --rotate 2
python3 ../main.py --exp_name $ENV$NUM"_noflip_rot360" $@ --augmentations combine2 --resize_crop 0.5 --jitter True --grayscale True --flip False --max_angle_speed 360
python3 ../main.py --exp_name $ENV$NUM"_noflip_rot90" $@ --augmentations combine2 --resize_crop 0.5 --jitter True --grayscale True --flip False --max_angle_speed 90
python3 ../main.py --exp_name $ENV$NUM"_all_rot90" $@ --augmentations combine2 --resize_crop 0.5 --jitter True --grayscale True --flip True --max_angle_speed 90
python3 ../main.py --exp_name $ENV$NUM"_all_rot360" $@ --augmentations combine2 --resize_crop 0.5 --jitter True --grayscale True --flip True --max_angle_speed 360
python3 ../main.py --exp_name $ENV$NUM"_stall_rot90" $@ --augmentations standard2 --resize_crop 0.5 --jitter True --grayscale True --flip True --max_angle_speed 90
python3 ../main.py --exp_name $ENV$NUM"_stall_rot360" $@ --augmentations standard2 --resize_crop 0.5 --jitter True --grayscale True --flip True --max_angle_speed 360
python3 ../main.py --exp_name $ENV$NUM"_noflipcrop_rot360_depth" $@ --augmentations combine2 --resize_crop 0 --jitter True --grayscale True --flip False --max_angle_speed 360 --depth 0.45
python3 ../main.py --exp_name $ENV$NUM"_all_rot360_depth" $@ --augmentations combine2 --resize_crop 0.5 --jitter True --grayscale True --flip True --max_angle_speed 360 --depth 0.45
