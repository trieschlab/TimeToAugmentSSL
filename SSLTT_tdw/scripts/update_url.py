import argparse
import copy
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--src", type=str, default="/library")
parser.add_argument("--dest", type=str, default="/library")
args = parser.parse_args()

current = args.src #str(Path().cwd().resolve())
dest = args.dest
# dest = "/sps/toy_library"


js = json.load(open(str(current)+"/toys.json"))
i= 0
js_copy = copy.deepcopy(js)
for key, val in js["records"].items():
    for platform in ["Linux", "Windows", "Darwin"]:
        new_url = dest + "/" + val["wnid"] + "/" + val["name"] +"/"+ platform
        js_copy["records"][key]["urls"][platform] = "file:///" + new_url

with open(current+"/toys.json", 'w') as f:
    json.dump(js_copy, f)
