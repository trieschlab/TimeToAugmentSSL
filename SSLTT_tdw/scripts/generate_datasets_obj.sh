#!/bin/bash

tobj=('40' '80')
for obj in ${tobj[@]}
do
  python3 envs/six_objects.py  --background 0 --num_obj $obj
  python3 envs/six_objects.py  --background 3 --num_obj $obj
  tab=('10' '0')
  for back in ${tab[@]}
  do
    python3 envs/room_toys.py  --background $back --num_obj $obj
    python3 envs/room_toys.py  --background $back --num_obj $obj
  done
done