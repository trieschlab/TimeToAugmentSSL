#!/bin/bash

#python3 envs/six_objects.py  --aperture 20 --background 0
#python3 envs/six_objects.py  --aperture 20 --background 1
#python3 envs/six_objects.py  --aperture 20 --background 2
#python3 envs/six_objects.py  --aperture 20 --background 3
#python3 envs/six_objects.py  --aperture 20 --background 3 --quality True

#python3 envs/six_objects.py  --aperture 20 --background 1
#python3 envs/six_objects.py  --aperture 20 --background 2
#python3 envs/six_objects.py  --aperture 20 --background 3
#python3 envs/six_objects.py   --aperture 20 --background 5
#python3 envs/six_objects.py   --background 6 --aperture 20
#python3 envs/six_objects.py   --background 5 --aperture 5
#python3 envs/six_objects.py   --background 5 --aperture 3
#python3 envs/six_objects.py   --aperture 20 --background 5 --closeness 0.7
#python3 envs/six_objects.py   --aperture 20 --background 5 --foveation 1 --noise 1
#python3 envs/six_objects.py   --aperture 20 --background 5 --noise 1
#python3 envs/six_objects.py   --aperture 4 --background 5 --noise 1 --closeness 0.7 --foveation 1

#python3 envs/six_objects.py   --aperture 1 --background 0
#python3 envs/six_objects.py   --aperture 3 --background 2
#python3 envs/six_objects.py   --aperture 5 --background 0
#python3 envs/six_objects.py   --aperture 4 --background 0


tab=('10' '0')
for back in ${tab[@]}
do
  #python3 envs/room_toys.py  --aperture 20 --background $back
  #python3 envs/room_toys.py  --aperture 20 --background $back --closeness 0.7
  #python3 envs/room_toys.py  --aperture 20 --background $back --foveation 1
  #python3 envs/room_toys.py  --aperture 20 --background $back --foveation 2
  #python3 envs/room_toys.py  --aperture 0.1 --background $back
  #python3 envs/room_toys.py  --aperture 1 --background $back
  #python3 envs/room_toys.py  --aperture 2 --background $back
  #python3 envs/room_toys.py  --aperture 3 --background $back
  #python3 envs/room_toys.py  --aperture 20 --background $back --foveation 1 --noise 2
  #python3 envs/room_toys.py  --aperture 20 --background $back --foveation 1 --noise 1
  #python3 envs/room_toys.py  --aperture 20 --background $back --foveation 1 --noise 0.5
  #python3 envs/room_toys.py  --aperture 20 --background $back --closeness 0.7 --foveation 1
  #python3 envs/room_toys.py  --aperture 20 --background $back --closeness 0.7 --foveation 1 --noise 1
  #python3 envs/room_toys.py  --aperture 4 --background $back --closeness 0.7 --foveation 1 --noise 1
  #python3 envs/room_toys.py  --background $back --closeness 0.7 --foveation 1 --noise 1
  #python3 envs/room_toys.py  --aperture 4 --background $back --closeness 0.7
  #python3 envs/room_toys.py  --aperture 4 --background $back --closeness 0.7 --foveation 1 --noise 1
  #python3 envs/room_toys.py  --aperture 4 --background $back --closeness 0.7 --foveation 1 --noise 1 --num_obj 40
  #python3 envs/room_toys.py  --aperture 4 --background $back --closeness 0.7 --foveation 1 --noise 1 --num_obj 80

  ##python3 envs/room_toys.py  --aperture 4 --background $back --closeness 0.7 --foveation 1 --noise 1 --env_back 4
  #python3 envs/room_toys.py  --aperture 20 --background $back --closeness 0.7 --env_back 4
  #python3 envs/room_toys.py  --aperture 20 --background $back --foveation 1 --noise 1 --env_back 4
  #python3 envs/room_toys.py  --aperture 3 --background $back --env_back 4

done

#miss background 10 and 30
