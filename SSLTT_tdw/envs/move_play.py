import csv
import math
import os
import sys
import time

from tdw.controller import Controller
from tdw.output_data import OutputData, Bounds, Collision, EnvironmentCollision

from envs.full_play import FullPlay

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")

import gym
import random
import numpy as np
from tdw.librarian import ModelLibrarian
from tdw.tdw_utils import TDWUtils
from envs.room_toys import RoomToys

from envs.six_objects import ToysObjects, ToysDataset
from tools.arguments import parse_datasets, str2bool
from tools.utils import build_envname


class MovePlay(FullPlay):

    def __init__(self, *args,**kwargs):
        super().__init__(*args, **kwargs)
        self.holded_object = None

    def override_params(self):
        self.above_rendering = False
        self.aperture = self.args.aperture if not self.above_rendering else 20
        self.oy=0.32#if self.args.background != 0 and self.args.background != 5 else -0.1
        self.c.communicate([{"$type": "set_gravity_vector", "gravity": {"x": 0, "y": -9.81, "z": 0}},{"$type": "simulate_physics", "value": True}])

        if self.args.background >= 7:
            self.above_camera_position = {"x": 0, "y": 10.7, "z":0}
            self.img_size = 128
        self.y_look_at = ToysObjects.Y_POS

    def get_object(self):
        if self.holded_object is None:
            return None
        return self.objects[self.holded_object]

    def adjust_physics(self):
        return [{"$type": "step_physics", "frames": 5}]
        # return []
        # self.foveation = 1

    def get_object_focus(self):
        return self.diagonal_init_distance - self.focus_corr

    def reset(self):
        command = []
        command.extend(self.reset_center_position())

        for obj in self.objects:
            command.append({"$type": "destroy_object", "id": obj["id"]})
        self.c.communicate(command)

        command = []
        self.objects_d = {}
        weights = self.get_object_weights()
        # for i in range(self.number_objects):
        for k in self.available_positions:
            for i in k["pos"]:
                obj, command_obj = self.generate_object(weights, i, cx=k["center"][0], cz=k["center"][2])
                command.extend(command_obj)
                self.objects_d[obj["oid"]]=obj
        command.extend(self.adjust_physics())
        self.holded_object = None
        command.append({"$type": "send_transforms", "ids": [objects_d[o]["oid"] for o in self.objects_d]})
        self.resp = self.c.communicate(command)
        img = self.render()
        return img.swapaxes(0, 2).swapaxes(1, 2)  #

    def turn(self, angle, switch=False):
        command = []
        obj = self.objects_d[self.posorder]

        if angle != 0:
            command.append({"$type": "rotate_avatar_by", "avatar_id": self.avatar_id, "euler_angles": {"y": angle,"x": self.y_look_at, "z": 0}})
            command.append({"$type": "rotate_object_by", "angle": -angle, "axis": "yaw", "id": obj["id"]})

        if switch:
            if self.holded_object is not None:
                self.holded_object = None
            else:
                pos_obj = None
                id_obj = None
                a_pos = np.asarray(self.avatar_data.get_position())
                a_for = np.asarray(self.avatar_data.get_forward())
                for i in self.transforms.get_num():
                    opos = np.asarray(self.transforms.get_position(i))
                    d = np.linalg.norm(opos, a_pos)
                    align = np.dot(a_for,opos-a_pos)
                    if align < 0.5:
                        if pos_obj is not None:
                            if d < pos_obj:
                                pos_obj = d
                                id_obj = self.transforms.get_id(i)
                if pos_obj < 1:
                    self.holded_object = id_obj
                    self.objects_d[self.holded_object]["depth"]=self.args.closeness

        return command

    def turn_body_action(self, action):
        new_turn = action[0] #if not self.default_values[0] else 0.1
        switch_f = action[7]
        command = []
        sigmoid = self._sigmoid(new_turn)
        neg_sigmoid = self._neg_sigmoid(new_turn)

        new_switch = False
        if switch_f == -1 or random.uniform(0, 1) > self._sigmoid(switch_f, -5, 0):
            new_switch = True
        if switch_f == 1:
            new_switch = False

        if new_turn == 0:
            new_switch = new_switch if not self.default_values[7] else False
            command.extend(self.turn(0, switch=new_switch))
        elif new_turn == -1 or random.uniform(0, 1) > sigmoid:
            new_switch = new_switch if not self.default_values[7] else True
            command.extend(self.turn(-1, switch=new_switch))
        elif new_turn == 1 or random.uniform(0, 1) > neg_sigmoid:
            new_switch = new_switch if not self.default_values[7] else True
            command.extend(self.turn(1, switch=new_switch))
        else:
            new_switch = new_switch if not self.default_values[7] else False
            command.extend(self.turn(0, switch=new_switch))

        forward = action[10]
        sigmoid_for = self._sigmoid(new_turn)
        neg_sigmoid_for = self._neg_sigmoid(new_turn)


        if forward != 0 and (forward == -1 or random.uniform(0, 1) > sigmoid_for):
            command.append({"$type": "move_avatar_forward_by", "magnitude": 1, "avatar_id": "a"})
        elif forward != 0 and (forward == 1 or random.uniform(0, 1) > neg_sigmoid_for):
            command.append({"$type": "move_avatar_forward_by", "magnitude": -1, "avatar_id": "a"})

        return not new_switch, command

    def change_vision(self, action):
        normalized_focus = (action[6]+ 1)/2
        new_focus = normalized_focus * (self.max_distance_focus-self.min_distance_focus) + self.min_distance_focus
        if not self.default_values[6]:
            new_focus = self.diagonal_init_distance*new_focus - self.focus_corr
        else:
            new_focus = self.get_object_focus()
        command = [{"$type": "set_focus_distance", "focus_distance": new_focus}]
        return command

    def step(self, action):
        command = []
        if self.args.max_angle_speed == -1:
            self.max_angle_speed = max(10,self.max_angle_speed/1.00001)
        if self.args.max_angle_speed == -2:
            self.max_angle_speed = min(self.max_angle_speed * 1.00001, 360)
        command.append({"$type": "reset_sensor_container_rotation"})
        self.angle_x, self.angle_y = 0, 0

        # action = [0,0,0,0,0,0,min(self.tmp,1),0]
        # self.tmp += 0.1
        # action = [-0.1,0,0,0,0,0,0,0,0,0]
        # action[4]= -1
        # action[5]= 0
        fix, command_body = self.turn_body_action(action)
        # Teleport of the object happens later in change_object_position
        command.extend(command_body)
        command.extend(self.make_objects_turn(action))
        # action = [0,0,0,1,-1,-1,0]
        command.extend(self.noisy_action(action))
        command.extend(self.change_object_position(action))
        command.extend(self.change_vision(action))


        # command.append({"$type": "set_focus_distance",  "focus_distance": math.sqrt(0.6**2 + (self.cy - self.oy)**2)})
        # command.append({"$type": "set_aperture", "aperture": 10})
        # distance = np.linalg.norm(np.asarray(self.objects[self.posorder]["position"]) - np.asarray(self.available_positions[self.posorder]["center"]))
        # command.append({"$type": "send_collisions"})
        self.resp = self.c.communicate(command)
        # for j in range(len(self.resp) - 1):
        #     r_id = OutputData.get_data_type_id(self.resp[j])
        #     if r_id == "coll":
        #         b = Collision(self.resp[j])
        #         if b.get_collidee_id() == self.objects[self.posorder]["id"] or b.get_collider_id() == self.objects[self.posorder]["id"]:
        #             print(self.current_center, self.posorder, self.objects[self.posorder]["name"])
        #
        #     if r_id == "enco":
        #         c = EnvironmentCollision(r)
        #         print("ENCO",c.get_object_id())
        #         print(self.objects[self.posorder]["name"])

        obs = self.render().swapaxes(0, 2).swapaxes(1,2)
        infos = self.get_infos()
        infos["fix"] = fix
        self.cpt_act += 1
        return {"observation": obs, "oid": infos["oid"], "category": infos["category"] }, 0, False, infos
        # return obs, 0, False, infos# {"object_names": self.list_possible_names}


if __name__ == "__main__":
    parser = parse_datasets()
    parser.add_argument("--mode",type=str, default="single")
    parser.add_argument("--init_rotate",type=int, default=-45)
    parser.add_argument("--pitch",type=int, default=0)
    parser.add_argument("--roll",type=int, default=0)
    parser.add_argument("--begin",type=int, default=0)
    parser.add_argument("--end",type=int, default=2000)

    args = parser.parse_args()
    os.environ["DISPLAY"] = args.display
    if args.local:
        TDWUtils.set_default_libraries(model_library=os.environ["LOCAL_BUNDLES"] + "/local_asset_bundles/models.json",
                                       scene_library=os.environ["LOCAL_BUNDLES"] + "/local_asset_bundles/scenes.json",
                                       material_library=os.environ["LOCAL_BUNDLES"] + "/local_asset_bundles/materials.json",
                                       hdri_skybox_library=os.environ["LOCAL_BUNDLES"] + "/local_asset_bundles/hdri_skyboxes.json")
    args.env_name = "full_play"
    env_name = build_envname(args, rotate=False)
    if args.pitch:
        env_name = env_name+"_pitch"
    if args.roll:
        env_name = env_name+"_roll"
    if args.mode == "single":
        env_name = env_name
    elif args.init_rotate == -45:
        env_name = env_name+"_r"
    else:
        env_name = env_name+"_r"+str(args.init_rotate)


    dataset = FullPlayDatasets(env_name, args)

#foveation, textures, objects rotations, backgrounds