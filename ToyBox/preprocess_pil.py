import argparse
import os

from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, default=".")
parser.add_argument("--category", type=str, default="animals")

args = parser.parse_args()
path = os.path.abspath(args.path)
type=args.category
path2 = os.path.join(path,type)
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
				try:
					im = Image.open(name)
					w, h = im.size
					if w > 500:
						im.resize((288,162)).convert('RGB').save(name)
					im.close()
				except:
					print(name)

