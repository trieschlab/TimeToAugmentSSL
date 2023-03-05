# Time to augment visual self-supervised learning

The repository contains the code need to reproduce the ToyBox results of https://arxiv.org/pdf/2207.13492.pdf .

### Initialization

The following files were used to generated the datasets from the clips, you will not need them:
```
preprocess_mp4.py
preprocess_pil.py
preprocess.png.py
preprocess_simclr.py
```

Create singularity container:
```
sudo singularity build ssltt.sif ssltt.def
```

Prepare the datasets and the directory:
```
mkdir results
https://zenodo.org/record/7327766/files/augmented_toybox_dataset.zip
https://zenodo.org/record/7327766/files/toybox_dataset.zip
unzip toybox_dataset.zip   
unzip augmented_toybox_dataset.zip   
python3 build_csv.py --path toybox_dataset
python3 build_csv.py --path augmented_toybox_dataset
```


### Run the code:

```
singularity exec ssltt.sif python3 main.py --seed 0 --save_dir ./results/test_0/ --path toybox_dataset --path2 augmented_toybox_dataset/ --num_epochs 60
```

Add the method option to select applied augmentations:
```
--method time # *-TT
--method simclr2 # * with one positive samples per image
--method combine2 # *-TT+ with one positive samples per image
--method simclr # * with more positive samples per image
--method combine # *-TT+ with more positive samples per image
```

Add the loss option to select the used loss:
```
--loss {simclr,vicreg,byol}
```


