import argparse
import csv
import os

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, default=".")
args = parser.parse_args()
path = os.path.abspath(args.path)

with open(os.path.join(path, "test_dataset.csv"), 'w') as test_set:
	with open(os.path.join(path, "dataset.csv"), 'w') as train_set:
		writer_train = csv.writer(train_set)
		writer_test = csv.writer(test_set)

		for type in os.listdir(path):
			path2 = os.path.join(path,type)
			if not os.path.isdir(path2):
				continue
			for subtype in os.listdir(path2):
				path3 = os.path.join(path2,subtype)
				for f in os.listdir(path3):
					path4 = os.path.join(path3,f)
					if os.path.isdir(path4):
						for f2 in os.listdir(path4):
							splits = f2.split("_")
							type = splits[3]
							category = splits[0]
							object = splits[1]
							view = splits[4].split(".")[0]
							last = splits[2]
							full_path = os.path.join(path4, f2)
							if int(object) <= 20:
								writer_train.writerow([full_path,  category, object, last, type, view])
							else:
								writer_test.writerow([full_path, category, object, last, type, view])
