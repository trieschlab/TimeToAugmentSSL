#!/bin/bash

python3 envs/six_objects.py  --aperture 20 --background 0 --noise 1
python3 envs/six_objects.py  --aperture 20 --background 1 --noise 1
python3 envs/six_objects.py  --aperture 20 --background 2 --noise 1
python3 envs/six_objects.py  --aperture 20 --background 3 --noise 1
python3 envs/six_objects.py  --aperture 20 --background 3 --quality True --noise 1

tab=('10' '0')
for back in ${tab[@]}
do
  python3 envs/room_toys.py  --aperture 20 --background $back --noise 1
  python3 envs/room_toys.py  --aperture 20 --background $back --closeness 0.7 --noise 1
  python3 envs/room_toys.py  --aperture 20 --background $back --foveation 1 --noise 1
  python3 envs/room_toys.py  --aperture 20 --background $back --foveation 2 --noise 1
  python3 envs/room_toys.py  --aperture 0.1 --background $back --noise 1
  python3 envs/room_toys.py  --aperture 1 --background $back --noise 1
  python3 envs/room_toys.py  --aperture 2 --background $back --noise 1
  python3 envs/room_toys.py  --aperture 3 --background $back --noise 1
done


