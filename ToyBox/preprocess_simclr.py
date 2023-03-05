import argparse
import os

from PIL import Image

from utils import get_augmentations

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, default=".")
parser.add_argument("--category", type=str, default="animals")
parser.add_argument("--name", type=str, default="dataset")
parser.add_argument("--crop", type=float, default=0.5)
parser.add_argument("--color_only", type=int, default=0)
parser.add_argument("--pcolor", type=float, default=0.8)
args = parser.parse_args()
path = os.path.abspath(args.path)


type=args.category
path2 = os.path.join(path,type)
augmentations = get_augmentations(args)
for subtype in os.listdir(path2):
	path3 = os.path.join(path2, subtype)
	if not os.path.isdir(path3):
		continue
	for f in os.listdir(path3):
		splits = f.split(".")
		name = splits[0]
		path4 = os.path.join(path3, name)
		if len(splits) > 1 and splits[1] == "mp4":
			if not os.path.isdir(path4):
				os.mkdir(path4)
			for d2 in os.listdir(path4):
				name = os.path.join(path4, d2)
				im = Image.open(name)
				im2 = augmentations(im)
				new_name = name.replace(args.name,"dataset_simclr")
				dir = "/".join(new_name.split("/")[:-1])
				if not os.path.isdir(dir):
					os.makedirs(dir)
				if not os.path.isfile(new_name):
					im2.save(new_name)

				im.close()


