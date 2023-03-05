# Time to augment visual self-supervised learning

This is the code for VHE and 3DShapeE experiments of this paper: https://openreview.net/pdf?id=o8xdgmwCP8l

A new cleaned up repository will be soon available.


### Download all datasets and assets

Downloads and extract TDW assets and VHE/3DShapeE evaluation datasets. They are available here:
https://zenodo.org/record/7327766#.Y_3iTYCZNH4

### Set up environment variables according to bundles/datasets positions
```
export LOCAL_BUNDLES=A_directory with ending /
export DATASETS_LOCATION=Directory that contains/will contain the datasets with ending /
export TOY_UTEX_LIBRARY_PATH=directory containing untextured tdw bundles without ending /
export TOY_TEX_LIBRARY_PATH=directory containind textured tdw bundles without ending /
```

LOCAL_BUNDLES, get_data.py is supposed to fill it and then the main will use it
DATASETS_LOCATION is the folder where you put the datasets (iclr_datasets)
TOY_UTEX_LIBRARY_PATH is the location of all untextured models (library_untex)
TOY_TEX_LIBRARY_PATH is the location of all textured models (library_tex)


### Update urls of bundles to your machine

```
python3 scripts/update_url.py --src $TOY_TEX_LIBRARY_PATH --dest $TOY_TEX_LIBRARY_PATH
python3 scripts/update_url.py --src $TOY_UTEX_LIBRARY_PATH --dest $TOY_UTEX_LIBRARY_PATH
```

Replace dest argument by binding folder if using bind functionality of singularity.

###Download scene bundles from TDW for fast simulation start in the VHE environment

```
python3 scripts/get_data.py
```

### If not using previously built dataset, Generate a validation/test dataset
```
python3 envs/full_play.py  --aperture 5 --background 5 --local True --closeness 0.8 --rotate 0 --noise 0 --num_obj 4000 
python3 envs/full_play.py  --aperture 5 --background 10 --local True --closeness 0.8 --rotate 0 --noise 0 --num_obj 4001 
```
### Install dependencies within singularity (> v3.6)

```
sudo singularity build cltt.sif cltt.def
```

### Run the code

```
singularity exec cltt.sif python3 main.py `--exp_name simclr_combine --save ../gym_results/tests/ --env_name full_play --background 10 --num_obj 4001 --agent ordturn_1_0.1 --augmentations combine2 --jitter True --grayscale True --switch False --max_angle_speed 360 --batchnorm True`
```

##### Change augmentations type
--augmentations time : TT augmentations\
--augmentations standard2 : standard augmentations\
--augmentations combine2 : TT+ augmentations
Remove the "2" to create several positive pairs per sample.

##### Change environments
--background 5 --num_obj 4000: 3DShapeE\
--background 10 --num_obj 4001: VHE environment\


##### Change loss
--method {simclr,byol,vicreg}


### Get results
Category accuracy is given by final_acc_cat_test in progress.txt
