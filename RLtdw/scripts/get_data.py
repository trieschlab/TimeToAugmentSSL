import os, sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")

from json import loads
from pathlib import Path
from pkg_resources import resource_filename
from tdw.controller import Controller
import json
from tdw.librarian import SceneLibrarian
from tdw.tdw_utils import TDWUtils

from resources.rooms.simple_rooms import download_models_living_room_simple, download_materials_living_room_simple

json_object = {}

librarian = SceneLibrarian()
if "scenes.json" not in json_object:
    json_object["scenes.json"] = []
for record in librarian.records:
    if record.name[:2] == "mm" or record.name[:2] == "fl":
        json_object["scenes.json"].append(record.name)

for scene1 in ["1","2","4","5"]:
    for scene2 in ["a","b","c"]:
        for layout in ["0","1","2"]:
            scene = scene1+scene2

            floorplans = loads(Path(resource_filename("tdw.add_ons.floorplan", "floorplan_layouts.json")).read_text(encoding="utf-8"))
            objects = floorplans[scene1][layout]

            for o in objects:
                object_id = Controller.get_unique_id()
                if o["library"] not in json_object:
                    json_object[o["library"]] = []
                json_object[o["library"]].append(o["name"])

for name in download_models_living_room_simple():
    json_object["models_core.json"].append(name)


if "materials_med.json" not in json_object:
    json_object["materials_med.json"] = []

for name in download_materials_living_room_simple():
    json_object["materials_med.json"].append(name)

with open(os.environ["LOCAL_BUNDLES"]+"/local_names.json", "w") as outfile:
    json.dump(json_object, outfile)

TDWUtils.download_asset_bundles(path=os.environ["LOCAL_BUNDLES"]+"/local_asset_bundles",
                                models={"models_core.json": json_object["models_core.json"]},
                                scenes={"scenes.json": json_object["scenes.json"]},
                                materials={"materials_med.json": json_object["materials_med.json"]},
                                hdri_skyboxes={"hdri_skyboxes.json":["flower_road_4k"]}
                                )