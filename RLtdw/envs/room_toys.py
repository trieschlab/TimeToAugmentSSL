import sys, os

from tdw.tdw_utils import TDWUtils

from tools.utils import build_envname

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")
from tools.arguments import str2bool, parse_datasets

from tdw.add_ons.floorplan import Floorplan

import argparse
import copy
from envs.objects import Objects20, Objects1
from envs.six_objects import ToysDataset, ToysObjects
import random

variant1 = [{"pos": [0, 4, 5], "center": (1, ToysObjects.Y_POS, 2)},
            {"pos": [2, 3, 5], "center": (-3.6, ToysObjects.Y_POS, 2)},
            {"pos": [0, 3, 5], "center": (-7.8, ToysObjects.Y_POS, 4.4)},
            {"pos": [1, 2, 3, 4, 5], "center": (-10, ToysObjects.Y_POS, 3.6)},
            {"pos": [0, 1, 3, 5], "center": (1.7, ToysObjects.Y_POS, -1)},
            {"pos": [0, 1, 2, 3], "center": (-0.8, ToysObjects.Y_POS, -4.7)},
            {"pos": [0, 1, 2, 3], "center": (6.5, ToysObjects.Y_POS, -4.2)},
            {"pos": [0, 1, 2, 5], "center": (9.5, ToysObjects.Y_POS, -1.5)}] #30 backgrounds

# variant1 = [{"pos": [0], "center": (1, ToysObjects.Y_POS, 2)}]


# variant1= [{"pos": [0, 3, 5], "center": (-7.8, 0.2, 4.4)}]

variant2 = [{"pos": [1, 2, 3, 4, 5], "center": (-9, ToysObjects.Y_POS, 1.8)},
            {"pos": [0, 1, 2, 3, 4, 5], "center": (-8, ToysObjects.Y_POS, -0.4)},
            {"pos": [1, 2, 3, 4, 5], "center": (-4, ToysObjects.Y_POS, -0.7)},
            {"pos": [0, 1, 3, 4, 5], "center": (0, ToysObjects.Y_POS, -1.2)},
            {"pos": [0, 1, 2 , 3, 4, 5], "center": (5.5, ToysObjects.Y_POS, -2)},
            {"pos": [0, 1, 2 , 3, 4, 5], "center": (7.3, ToysObjects.Y_POS, -2.5)},
            {"pos": [1, 5], "center": (7.6, ToysObjects.Y_POS, 1.5)},
            {"pos": [2 , 5], "center": (6, ToysObjects.Y_POS, 2)},
            {"pos": [0, 1, 2], "center": (-5, ToysObjects.Y_POS, 2.7)}] #40 backgrounds

variant3 = [{"pos": [1, 2, 3, 4], "center": (-6.8, ToysObjects.Y_POS, -2)},
            {"pos": [0, 1, 2, 5], "center": (-4.3, ToysObjects.Y_POS, -2)},
            {"pos": [0, 1, 2, 3 ], "center": (-3, ToysObjects.Y_POS, 1.5)},
            {"pos": [0, 1, 2, 3], "center": (3.2, ToysObjects.Y_POS, 2)},
            {"pos": [0, 3, 4, 5], "center": (3.2, ToysObjects.Y_POS, -2)}] #20 backgrounds

variant4 = [{"pos": [0 , 1], "center": (-6.8, ToysObjects.Y_POS, -2)},
            {"pos": [0, 5], "center": (-7.8, ToysObjects.Y_POS, 1.3)},
            {"pos": [0, 3], "center": (-1.3, ToysObjects.Y_POS, -2.5)},
            {"pos": [0, 1, 2, 3], "center": (3, ToysObjects.Y_POS, -3.8)}] #10 backgrounds

# variant2 = [{"pos": [0, 1, 2 , 3, 4, 5], "center": (7.3, 0.2, -2.5)}]

class RoomToys(ToysObjects):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.above_rendering and self.args.background >= 10:
            self.c.communicate([{"$type": "set_floorplan_roof", "show": False}])
        # self.cpt_pos = 0

    def override_params(self):
        if self.args.background >= 10:
            self.oy = 0.3
            self.above_camera_position = {"x": self.cx, "y": 10.7, "z":self.cz }
            self.above_rendering = False
        # if self.above_rendering:
        #     self.img_size = 800


    def create_room(self,commands):
        if self.args.background < 10:
            return super().create_room(commands)
        floorplan = Floorplan()
        ###Start creation of the room
        background = (self.args.background - 10 ) %3
        scenes = ["1a","1b","1c","2a","2b","2c","5a","5b","5c","4a","4b","4c"]
        scene_index = (self.args.background-10)//3
        scene = scenes[scene_index]

        if scene_index < 3:
            self.available_positions = variant1
        elif scene_index < 6:
            self.available_positions = variant2
        elif scene_index < 9:
            self.available_positions = variant3
        elif scene_index < 12:
            self.available_positions = variant4
        floorplan.init_scene(scene=scene, layout=background)
        self.c.add_ons.extend([floorplan])
        self.c.communicate([])

        return commands

    def reset_center_position(self):
        self.current_center = random.randint(0, len(self.available_positions)-1)
        # self.cpt_pos += 1
        # self.current_center = self.cpt_pos%len(self.available_positions)
        return self.set_center_position(self.current_center)


class RoomsDataset(ToysDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def settup(self, args):
        self.env = RoomToys(args)
        self.num_views = max(1, int(40//args.num_obj)*(16//len(self.env.available_positions)))
        # self.num_views = int(0.05*args.num_obj)*(16//len(self.env.available_positions))
        # print(self.num_views)


    def rotate_view(self, id_obj):
        return [{"$type": "rotate_object_by", "angle": random.uniform(0, 360), "axis": "yaw", "id": id_obj}]


# def build_envname(args):
#     env_name = "tdw_room_toys"+str(args.num_obj)
#     env_name = env_name+"_back"+str(args.background)
#     env_name = env_name if args.aperture == 0 else env_name + "_app" + str(int(args.aperture))
#     env_name = env_name if args.closeness == 1. else env_name + "_clo" + str(args.closeness)
#     env_name = env_name if not args.quality else env_name + "_quality"
#     env_name = env_name if not args.foveation else env_name + "_fov"+str(args.foveation)
#     env_name = env_name if args.noise == 3 or args.noise == 0 else env_name + "_noi"+str(args.noise)
#     env_name = env_name if args.env_back == 1 else env_name + "_b"+str(args.env_back)
#
#     print("env_name:", env_name)
#     return env_name

if __name__ == "__main__":
    parser = parse_datasets()
    args = parser.parse_args()
    os.environ["DISPLAY"] = args.display
    if args.local:
        TDWUtils.set_default_libraries(model_library=os.environ["LOCAL_BUNDLES"] + "/local_asset_bundles/models.json",
                                       scene_library=os.environ["LOCAL_BUNDLES"] + "/local_asset_bundles/scenes.json",
                                       material_library=os.environ["LOCAL_BUNDLES"] + "/local_asset_bundles/materials.json",
                                       hdri_skybox_library=os.environ["LOCAL_BUNDLES"] + "/local_asset_bundles/hdri_skyboxes.json")
    env_name = build_envname(args)
    # env_name = env_name if not args.postprocess else env_name+"_pp"
    dataset = RoomsDataset(env_name, args)

