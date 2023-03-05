#!/bin/bash

export DATASETS_LOCATION=/home/comsee/postdoc/datasets/generated_datasets7/
#
python3 envs/full_play.py  --aperture 5 --background 5 --local True --closeness 0.8 --rotate 0 --noise 0 --num_obj 4000 --begin 0 --end 2000 --incline True --binocular True
python3 envs/full_play.py  --aperture 5 --background 5 --local True --closeness 0.8 --rotate 0 --noise 0 --num_obj 4000 --begin 2000 --end 4500 --incline True --binocular True
#
#python3 envs/full_play.py  --aperture 5 --background 7 --local True --closeness 0.8 --rotate 0 --noise 0 --num_obj 4001 --begin 0 --end 2000
#python3 envs/full_play.py  --aperture 5 --background 7 --local True --closeness 0.8 --rotate 0 --noise 0 --num_obj 4001 --begin 2000 --end 4500
#
#python3 envs/full_play.py  --aperture 5 --background 10 --local True --closeness 0.8 --rotate 0 --noise 0 --num_obj 4001 --begin 0 --end 2000
#python3 envs/full_play.py  --aperture 5 --background 10 --local True --closeness 0.8 --rotate 0 --noise 0 --num_obj 4001 --begin 2000 --end 4500
#
#python3 envs/full_play.py  --aperture 5 --background 20 --local True --closeness 0.8 --rotate 0 --noise 0 --num_obj 4001 --begin 0 --end 2000
#python3 envs/full_play.py  --aperture 5 --background 20 --local True --closeness 0.8 --rotate 0 --noise 0 --num_obj 4001 --begin 2000 --end 4500
#
#python3 envs/full_play.py  --aperture 5 --background 30 --local True --closeness 0.8 --rotate 0 --noise 0 --num_obj 4001 --begin 0 --end 2000
#python3 envs/full_play.py  --aperture 5 --background 30 --local True --closeness 0.8 --rotate 0 --noise 0 --num_obj 4001 --begin 2000 --end 4500
#
#python3 envs/full_play.py  --aperture 5 --background 40 --local True --closeness 0.8 --rotate 0 --noise 0 --num_obj 4001 --begin 0 --end 2000
#python3 envs/full_play.py  --aperture 5 --background 40 --local True --closeness 0.8 --rotate 0 --noise 0 --num_obj 4001 --begin 2000 --end 4500

#python3 envs/full_play.py  --aperture 5 --background 5 --local True --closeness 0.8 --rotate 0 --noise 0 --num_obj 4000 --begin 0 --end 2000 --roll 1
#python3 envs/full_play.py  --aperture 5 --background 5 --local True --closeness 0.8 --rotate 0 --noise 0 --num_obj 4000 --begin 2000 --end 4500 --roll 1
#
#python3 envs/full_play.py  --aperture 5 --background 7 --local True --closeness 0.8 --rotate 0 --noise 0 --num_obj 4001 --begin 0 --end 2000 --roll 1
#python3 envs/full_play.py  --aperture 5 --background 7 --local True --closeness 0.8 --rotate 0 --noise 0 --num_obj 4001 --begin 2000 --end 4500 --roll 1
#
#python3 envs/full_play.py  --aperture 5 --background 10 --local True --closeness 0.8 --rotate 0 --noise 0 --num_obj 4001 --begin 0 --end 2000 --roll 1
#python3 envs/full_play.py  --aperture 5 --background 10 --local True --closeness 0.8 --rotate 0 --noise 0 --num_obj 4001 --begin 2000 --end 4500 --roll 1

#python3 envs/full_play.py  --aperture 5 --background 5 --local True --closeness 0.8 --rotate 0 --noise 0 --num_obj 4000 --begin 0 --end 2000 --pitch 1
#python3 envs/full_play.py  --aperture 5 --background 5 --local True --closeness 0.8 --rotate 0 --noise 0 --num_obj 4000 --begin 2000 --end 4500 --pitch 1
#
#python3 envs/full_play.py  --aperture 5 --background 7 --local True --closeness 0.8 --rotate 0 --noise 0 --num_obj 4001 --begin 0 --end 2000 --pitch 1
#python3 envs/full_play.py  --aperture 5 --background 7 --local True --closeness 0.8 --rotate 0 --noise 0 --num_obj 4001 --begin 2000 --end 4500 --pitch 1
#
#python3 envs/full_play.py  --aperture 5 --background 10 --local True --closeness 0.8 --rotate 0 --noise 0 --num_obj 4001 --begin 0 --end 2000 --pitch 1
#python3 envs/full_play.py  --aperture 5 --background 10 --local True --closeness 0.8 --rotate 0 --noise 0 --num_obj 4001 --begin 2000 --end 4500 --pitch 1

rotates=('-45' '0' '-90')
for r in ${rotates[@]}
do
python3 envs/full_play.py  --aperture 5 --background 5 --local True --closeness 0.8 --rotate 0 --noise 0 --num_obj 4000 --begin 0 --end 2000 --init_rotate $r --mode rotation --incline True --binocular True
python3 envs/full_play.py  --aperture 5 --background 5 --local True --closeness 0.8 --rotate 0 --noise 0 --num_obj 4000 --begin 2000 --end 4500 --init_rotate $r --mode rotation --incline True --binocular True
done