import argparse
import subprocess
import os

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, default=".")
parser.add_argument("--category", type=str, default="animals")

args = parser.parse_args()
path = os.path.abspath(args.path)

# for type in os.listdir(path):
type=args.category
path2 = os.path.join(path,type)
for subtype in os.listdir(path2):
	path3 = os.path.join(path2,subtype)
	if not os.path.isdir(path3):
		continue
	for f in os.listdir(path3):
		splits = f.split(".")
		name = splits[0]
		path4=os.path.join(path3,name)
		if len(splits) > 1 and splits[1] == "mp4":
			if not os.path.isdir(path4):
				os.mkdir(path4)
			if not os.path.isfile( path4+"/"+name+"_0001.png"):
				subprocess.Popen(["ffmpeg", "-i", os.path.join(path3,f), "-pix_fmt", "rgba", "-r", "2", path4+"/"+name+"_%04d.png"]).wait()
