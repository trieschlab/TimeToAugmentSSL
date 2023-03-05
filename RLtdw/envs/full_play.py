import csv
import math
import os
import sys
import time

from tdw.controller import Controller
from tdw.output_data import OutputData, Bounds, Collision, EnvironmentCollision

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


class FullPlay(RoomToys):

    def __init__(self, args, default_values =None, test_mode=False, **kwargs):
        super().__init__(args, **kwargs)
        #First: object switch;  fourth: turn left/right, second: object rotation; third: object elevation/position; fifth/sixth: eye saccades
        # Depth of focus
        #Depth of focus/Depth of field, depth of objects ?
        # 1 Discrete , 7 continuous
        #turn left/center/right, object rotation, x saccade, y saccade, object depth, object elevation, distance of focus
        self.action_space = gym.spaces.Box(np.array([-1]*10), np.array([1]*10))
        self.elevation = 0
        # self.available_positions = [{"pos": [0, 1, 2, 3, 4, 5], "center": (0, 0.4, 0)}]
        self.max_angle_saccade= self.args.noise*6 if self.args.noise >= 0 else 24
        self.init_distance = args.closeness
        self.diagonal_init_distance = math.sqrt(self.init_distance**2 + (self.cy - self.oy)**2)
        self.d_angle = 2 * math.pi / self.number_objects
        self.min_depth = args.min_depth
        self.max_depth = self.min_depth + args.depth
        self.max_distance_focus = self.min_depth + (args.focus if args.focus >= 0 else 1.4)
        self.min_distance_focus = self.min_depth
        self.max_elevation = args.elevation
        self.focus_corr = 0.13 #0.13
        self.default_values = default_values if default_values is not None else [False]*10
        self.constant_angle_y = 25 if not self.args.incline else 0
        self.max_angle_speed = self.args.max_angle_speed
        self.test_mode = test_mode
        if self.args.max_angle_speed == -1:
            self.max_angle_speed = 360
        if self.args.max_angle_speed == -2:
            self.max_angle_speed = 10

        if test_mode:
            self.cpt_objects = 0
            self.list_possible_names = self.OBJECTS.get_records_name(test=True, training=True)


        if self.args.categories_removal == 1:
            self.list_possible_names = [self.list_possible_names[i] for i in range(len(self.list_possible_names)) if self.list_possible_names[i] > "c"]

        if self.args.room_categories:
            self.index_categories = {}
            self.list_of_categories = []
            for oname in self.list_possible_names:
                category = self.lib.get_record(oname).wcategory
                if not category in self.index_categories:
                    self.index_categories[category] = []
                    self.list_of_categories.append(category)
                self.index_categories[category].append(oname)

        if self.above_rendering and self.args.background >= 10:
            self.c.communicate([{"$type": "set_floorplan_roof", "show": False}])
        # time.sleep(20)
        # self.c.communicate([{"$type": "send_bounds", "ids": [id], "frequency": "always"}])

        # commands.append({"$type": "simulate_physics", "value": False})

        # self.c.communicate([{"$type": "send_avatars", "ids": ["a"], "frequency": "always"}])
        # self.c.communicate([{"$type": "send_composite_objects", "ids": ["a"], "frequency": "always"}])
        self.tmp = -1
        self.cpt_act = 0
        self.cpt_steps = 0
        self.actual_object_angle = 0
        self.sum_rotation = 0
        self.cpt_all = 0
        self.list_possible_numbers = list(range(0, len(self.list_possible_names)))

    def override_params(self):
        self.above_rendering = False
        self.aperture = self.args.aperture if not self.above_rendering else 20
        self.oy=0.32 if not self.args.incline else ToysObjects.Y_POS
        # self.oy=0.32
        # self.oy
        #if self.args.background != 0 and self.args.background != 5 else -0.1
        self.c.communicate([{"$type": "set_gravity_vector", "gravity": {"x": 0, "y": 0, "z": 0}},{"$type": "simulate_physics", "value": False}])

        # self.oy=0.4#if self.args.background != 0 and self.args.background != 5 else -0.1
        # self.oy=0 if self.args.background != 0 and self.args.background != 5 else -0.2
        if self.args.background >= 7:
            self.above_camera_position = {"x": 0, "y": 10.7, "z":0}
            # self.above_camera_position = {"x": 0, "y": 45.7, "z":0 }
            self.img_size = 128
        # self.y_look_at = 0
        self.y_look_at = ToysObjects.Y_POS if not self.args.incline else self.incline_rot
        # self.c.communicate([{"$type": "set_target_framerate", "framerate": 1}])

    def get_object(self):
        return self.objects[self.posorder]

    def adjust_physics(self):
        # return [{"$type": "step_physics", "frames": 5}]
        return []
        # self.foveation = 1

    def set_center_position(self, pos):
        command = super().set_center_position(pos)
        # command.append({"$type": "set_focus_distance",  "focus_distance": self.init_distance*self.args.closeness})
        # command.append({"$type": "set_focus_distance",  "focus_distance": self.diagonal_init_distance - self.focus_corr})
        # self.diagonal_init_distance * new_focus - self.focus_corr
        return command

    def reset(self):
        command=[]
        command.extend(self.reset_center_position())

        for obj in self.objects:
            if obj is not None:
                command.append({"$type": "destroy_object", "id": obj["id"]})
        self.c.communicate(command)

        command=[]
        self.objects = []
        # for i in range(self.number_objects):
        for i in range(len(self.available_positions[self.current_center]["pos"])):
            if i == self.posorder:
                direction = self.available_positions[self.current_center]["pos"][self.posorder]
                obj, command_obj = self.generate_object(self.get_object_weights(), direction)
                self.objects.append(obj)
            else:
                self.objects.append(None)
        command.extend(command_obj)
        command.extend(self.adjust_physics())
        command.append({"$type": "reset_sensor_container_rotation"})
        command.append({"$type": "rotate_sensor_container_by", "angle": self.constant_angle_y, "axis": "pitch", "avatar_id": self.avatar_id})
        command.append({"$type": "set_focus_distance", "focus_distance": self.diagonal_init_distance - self.focus_corr})
        # command.append(self.get_object_focus())
        self.resp = self.c.communicate(command)
        img= self.render()

        return {"observation": img.swapaxes(0,2).swapaxes(1,2), "oid": obj["oid"], "category": obj["category"], "angle": self.actual_object_angle}

    def get_x_z_positions(self, index):
        ox = math.cos(index * self.d_angle) * self.init_distance
        oz = math.sin(index * self.d_angle) * self.init_distance
        return ox, oz

    def get_object_focus(self):
        new_focus = math.sqrt((self.get_object()["posx"] - self. cx) ** 2 + (self.get_object()["posy"] - self. cy)**2
                + (self.get_object()["posz"] - self. cz) **2) - self.focus_corr
        return new_focus

    def generate_object(self, weights, i, cx=None, cz=None):
        command = []
        ox, oz = self.get_x_z_positions(i)
        cx = cx if cx is not None else self.cx
        cz = cz if cz is not None else self.cz
        if self.test_mode:
            oname_number = self.cpt_objects%len(self.list_possible_names)
            oname = self.list_possible_names[oname_number]
            self.cpt_objects += 1
        elif self.args.room_categories:
            v = len(self.available_positions)
            num_categories = len(self.list_of_categories)
            steps = num_categories//v
            min_cat = steps*self.current_center
            max_cat = (steps)*(self.current_center+1)-1 if (steps+2)*self.current_center < num_categories-1 else num_categories-1
            category_number = random.randint(min_cat, max_cat)
            oname = random.choices(self.index_categories[self.list_of_categories[category_number]])[0]
            oname_number = self.list_possible_names.index(oname)
            #self.current_center
        elif self.args.split_obj:
            if self.args.split_obj > self.cpt_steps:
                oname_number = random.choices(list(range(0, len(self.list_possible_names)-500)), weights=weights[:-500])[0]
            else:
                oname_number = random.choices(list(range(len(self.list_possible_names)-500,len(self.list_possible_names))), weights=weights[-500:])[0]
            oname = self.list_possible_names[oname_number]
        else:
            # oname_number = random.choices(list(range(0, len(self.list_possible_names))), weights=weights)[0]
            oname_number = random.choices(self.list_possible_numbers, weights=weights)[0]
            oname = self.list_possible_names[oname_number]

        obj_id = self.c.get_unique_id()
        try:
            category = self.categories_order[self.lib.get_record(oname).wcategory]
        except:
            print(oname)
            raise Exception(oname+ " not present")
        command.append(self.c.get_add_object(oname, object_id=obj_id, library=self.OBJECTS.lib_path,
                                             position={"x": cx + ox, "y": self.oy, "z": cz + oz},
                                             rotation={"x": 0, "y": 0, "z": 0}
                                             ))
        # command.append({"$type": "set_object_drag", "drag": 100, "angular_drag": 100, "id": obj_id})
        self.actual_object_angle = 0
        canonical_angle = self.lib.get_record(oname).canonical_rotation["y"]
        angle_pitch = 0
        angle_roll = 0
        self.cpt_act = 0
        if self.args.random_orient:
            self.actual_object_angle = random.uniform(0, 360)
            # self.actual_object_angle = 10*(self.actual_object_angle//10)
            self.variation = self.actual_object_angle%45
            self.variation = self.variation if self.variation <= 22.5 else self.variation - 45
            self.basis = 45*(self.actual_object_angle//45) if self.actual_object_angle%45 <= 22.5 else (45*(self.actual_object_angle//45) + 45)
            command.append({"$type": "rotate_object_by", "angle": self.actual_object_angle + canonical_angle, "axis": "yaw", "id": obj_id})
            if self.args.pitch > 0 and self.args.init_pitrol:
                angle_pitch = random.uniform(0, 360)
                command.append({"$type": "rotate_object_by", "angle": angle_pitch, "axis": "pitch", "id": obj_id})
            if self.args.roll > 0 and self.args.init_pitrol:
                angle_roll = random.uniform(0, 360)
                command.append({"$type": "rotate_object_by", "angle": random.uniform(0, 360), "axis": "roll", "id": obj_id})
        else:
            command.append({"$type": "rotate_object_by", "angle": self.actual_object_angle + canonical_angle, "axis": "yaw", "id": obj_id})

        obj = {"posx": cx + ox,"posy": self.oy, "posz": cz + oz, "order": i, "pitch":angle_pitch, "roll":angle_roll,
               "id": obj_id, "name": oname, "oid": oname_number, "depth": self.args.closeness, "category":category}
               # "id": obj_id, "name": oname, "oid": oname_number, "depth":0.8}
        # command.append({"$type": "send_transforms", "ids": [obj_id]})

        return obj, command


    def turn(self, angle, switch=False):
        command = []
        obj = self.get_object()
        previous_pos = self.posorder
        previous_center = self.current_center
        eulerdangle = 360 / self.number_objects

        if self.args.reset_switch:
            switch=False

        if switch:
            command.append({"$type": "destroy_object", "id": obj["id"]})
            self.c.communicate(command)
            command = []

        if angle != 0:
            if self.args.teleport:
                command.extend(self.reset_center_position())
                command.extend(self.adjust_physics())
                self.objects = []
                for i in range(len(self.available_positions[self.current_center]["pos"])):
                    self.objects.append(None)

            else:
                self.posorder = (self.posorder + angle) % len(self.objects)
                self.turn_avatar_to_position(command)

            if not switch:
                if not self.args.teleport: self.objects[previous_pos] = None
                old_direction = self.available_positions[previous_center]["pos"][previous_pos]
                new_direction = self.available_positions[self.current_center]["pos"][self.posorder]
                if obj is not None:
                    command.append({"$type": "rotate_object_by", "angle": -(new_direction-old_direction) * eulerdangle, "axis": "yaw", "id": obj["id"]})
                self.objects[self.posorder] = obj
            # self.c.communicate(command)
        if switch:
            if not self.args.teleport: self.objects[previous_pos] = None
            pos_angle = self.available_positions[self.current_center]["pos"][self.posorder]
            self.objects[self.posorder], com = self.generate_object(self.get_object_weights(), pos_angle)
            command.extend(com)

            # if not switch:
        return command

    def _sigmoid(self, val, k=-12, b=0.4):
        return 1 / (1 + np.exp(k*(val+b)))

    def _neg_sigmoid(self, val,  k=-12, b=0.4):
        return 1 - 1 / (1 + np.exp(k*(val-b)))

    def compute_aperture(self, new_focus):
        # return self.aperture * (self.init_distance / (new_focus))**2
        # return self.aperture * (self.init_distance / (new_focus))
        return self.aperture

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

        return not new_switch, command

    def make_objects_turn(self, action):
        commanderot = []
        angle=0
        if self.get_object() is None:
            return commanderot

        if self.args.roll< 0 and self.objects[self.posorder]["roll"] != 0:
            commanderot.extend([{"$type": "rotate_object_by", "angle": -self.objects[self.posorder]["roll"], "axis": "roll", "id": self.get_object()["id"]}])  # good rotation
            self.objects[self.posorder]["roll"] = 0
        if self.args.pitch < 0 and self.objects[self.posorder]["pitch"] != 0:
            commanderot.extend([{"$type": "rotate_object_by", "angle": -self.objects[self.posorder]["pitch"], "axis": "pitch", "id": self.get_object()["id"]}])  # good rotation
            self.objects[self.posorder]["pitch"] = 0


        if self.args.rotate != 0:
            angle = (action[1] + 1)/2 #Between 0 and 1
            angle = (angle * (self.max_angle_speed - self.args.min_angle_speed)) + self.args.min_angle_speed
            angle = angle if not self.default_values[1] else random.uniform(self.args.min_angle_speed,self.max_angle_speed)
            if self.args.rotate_noise:
                angle = angle + random.gauss(0,self.args.rotate_noise)
            if self.args.rotate_uni:
                if angle < 0: angle = random.uniform(angle,0)
                if angle > 0: angle = random.uniform(0, angle)
            commanderot.extend([{"$type": "rotate_object_by", "angle": angle, "axis": "yaw", "id": self.get_object()["id"]}])  # good rotation

        if self.args.pitch != 0:
            angle = (action[8] + 1) / 2  # Between 0 and 1
            angle = (angle * (self.max_angle_speed - self.args.min_angle_speed)) + self.args.min_angle_speed
            angle = angle if not self.default_values[8] else random.uniform(self.args.min_angle_speed, self.max_angle_speed)
            if self.args.rotate_noise:
                angle = angle + random.gauss(0,self.args.rotate_noise)
            self.objects[self.posorder]["pitch"] = angle
            commanderot.extend([{"$type": "rotate_object_by", "angle": angle, "axis": "pitch", "id": self.get_object()["id"]}])  # good rotation

        if self.args.roll != 0:
            angle = (action[9] + 1) / 2  # Between 0 and 1
            angle = (angle * (self.max_angle_speed - self.args.min_angle_speed)) + self.args.min_angle_speed
            angle = angle if not self.default_values[9] else random.uniform(self.args.min_angle_speed, self.max_angle_speed)
            if self.args.rotate_noise:
                angle = angle + random.gauss(0,self.args.rotate_noise)
            commanderot.extend([{"$type": "rotate_object_by", "angle": angle, "axis": "roll", "id": self.get_object()["id"]}])  # good rotation
        self.actual_object_angle = (self.actual_object_angle+angle)%360

        return commanderot


    def noisy_action(self, action):
        x_sacc = action[2] if not self.default_values[2] else 0
        y_sacc = action[3] if not self.default_values[3] else 0


        command=[]
        normalize_x = (x_sacc+ 1)/2
        normalize_y = (y_sacc + 1)/2

        self.angle_x = (normalize_x * (2*self.max_angle_saccade)) - self.max_angle_saccade
        self.angle_y = (normalize_y * (2*self.max_angle_saccade)) - self.max_angle_saccade
        # if self.args.noise == -1 and not self.default_values[2] and not self.default_values[3]:
        #     if random.random() < 0.5 :
        #         self.angle_x, self.angle_y = 0,0
        #     else:
        #         angle = random.uniform(0,2*math.pi)
        #         self.angle_x, self.angle_y = math.cos(angle)*24, math.sin(angle)*24
        command.append({"$type": "rotate_sensor_container_by", "angle": self.angle_x, "axis": "yaw", "avatar_id": self.avatar_id})
        command.append({"$type": "rotate_sensor_container_by", "angle": self.angle_y + self.constant_angle_y, "axis": "pitch", "avatar_id": self.avatar_id})
        return command

    def change_object_position(self,action):
        #Compute new depth of the object
        if self.get_object() is None:
            return []
        normalize_d = (action[4] + 1)/2
        # gap = 0.1 # 0.075
        gap = self.args.depth_changes # 0.075
        depth_change = (normalize_d * 2*gap) - gap
        # new_depth = min(1, max(0.7, self.get_object()["depth"]+depth_change))
        new_depth = min(self.max_depth, max(self.min_depth, self.get_object()["depth"]+depth_change))
        new_depth = new_depth if not self.default_values[4] else self.args.closeness
        self.get_object()["depth"]=new_depth
        self.get_object()["posx"] = self.cx + math.cos(self.available_positions[self.current_center]["pos"][self.posorder] * self.d_angle) * new_depth
        self.get_object()["posz"] = self.cz + math.sin(self.available_positions[self.current_center]["pos"][self.posorder] * self.d_angle) * new_depth

        new_el = action[5] if not self.default_values[5] else 0
        normalize_ele = (new_el + 1)/2
        elevation_change = (normalize_ele * 2*0.1) - 0.1
        self.get_object()["posy"] = self.oy+min(self.max_elevation, max(0., self.get_object()["posy"]-self.oy +elevation_change))

        command = [{"$type": "teleport_object", "id": self.get_object()["id"],
                    "position": {"x": self.get_object()["posx"], "y": self.get_object()["posy"], "z": self.get_object()["posz"]}}]
        return command
        #Compute new elevation of the object

    def change_vision(self, action):
        normalized_focus = (action[6]+ 1)/2
        new_focus = normalized_focus * (self.max_distance_focus-self.min_distance_focus) + self.min_distance_focus
        if not self.default_values[6] and action[6] != -2:
            new_focus = self.diagonal_init_distance*new_focus - self.focus_corr
        else:
            new_focus = self.get_object_focus()
        command = [{"$type": "set_focus_distance", "focus_distance": new_focus}]
        # command.append({"$type": "set_aperture", "aperture": self.compute_aperture(new_focus)})#0.83

        # aperture = 6 - 3 * new_focus
        # command.append({"$type": "set_aperture", "aperture": aperture})
        return command

    def step(self, action):
        done=False
        command = []
        prev_angle =  self.actual_object_angle
        if self.args.aug_meth == "one_aug":
            self.resp = self.c.communicate(self.reset_augmentations())
            prev_obs = self.render().swapaxes(0, 2).swapaxes(1, 2)

        command.append({"$type": "reset_sensor_container_rotation"})
        self.angle_x, self.angle_y = 0, 0
        fix, command_body = self.turn_body_action(action)
        # Teleport of the object happens later in change_object_position
        command.extend(command_body)
        command.extend(self.make_objects_turn(action))

        command.extend(self.noisy_action(action))
        command.extend(self.change_object_position(action))
        command.extend(self.change_vision(action))


        # command.append({"$type": "set_focus_distance",  "focus_distance": math.sqrt(0.6**2 + (self.cy - self.oy)**2)})
        # command.append({"$type": "set_aperture", "aperture": 10})
        # distance = np.linalg.norm(np.asarray(self.get_object()["position"]) - np.asarray(self.available_positions[self.posorder]["center"]))
        # command.append({"$type": "send_collisions"})
        self.resp = self.c.communicate(command)
        # for j in range(len(self.resp) - 1):
        #     r_id = OutputData.get_data_type_id(self.resp[j])
        #     if r_id == "coll":
        #         b = Collision(self.resp[j])
        #         if b.get_collidee_id() == self.get_object()["id"] or b.get_collider_id() == self.get_object()["id"]:
        #             print(self.current_center, self.posorder, self.get_object()["name"])
        #
        #     if r_id == "enco":
        #         c = EnvironmentCollision(r)
        #         print("ENCO",c.get_object_id())
        #         print(self.get_object()["name"])

        obs = self.render().swapaxes(0, 2).swapaxes(1,2)
        infos = self.get_infos()
        infos["fix"] = fix
        infos["angle"] = self.actual_object_angle
        infos["prev_angle"] = prev_angle
        infos["prev_obs"] = prev_obs if self.args.aug_meth =="one_aug" else None


        if self.test_mode:
            infos["finish"] = False
            if self.cpt_objects == len(self.list_possible_names)+1:
                done = True
        self.cpt_act += 1
        self.cpt_steps += 1
        return {"observation": obs, "oid": infos["oid"], "category": infos["category"], "angle": infos["angle"]}, 0, done, infos
        # return obs, 0, False, infos# {"object_names": self.list_possible_names}

    def reset_augmentations(self):
        command = []
        command.append({"$type": "reset_sensor_container_rotation"})
        command.append({"$type": "set_focus_distance", "focus_distance": self.get_object_focus()})
        command.append({"$type": "rotate_sensor_container_by", "angle": self.constant_angle_y, "axis": "pitch","avatar_id": self.avatar_id})
        posx = self.cx + math.cos(self.available_positions[self.current_center]["pos"][self.posorder] * self.d_angle) * self.args.closeness
        posz = self.cz + math.sin(self.available_positions[self.current_center]["pos"][self.posorder] * self.d_angle) * self.args.closeness
        posy = self.get_object()["posy"]
        command.append({"$type": "teleport_object", "id": self.get_object()["id"], "position": {"x": posx, "y":posy, "z": posz}})
        return command

class FullPlayDatasets(ToysDataset):

    def settup(self, args):
        self.env = FullPlay(args, training=False)
        self.num_backgrounds = 1
        if self.env.args.mode == "single":
            self.num_views = 3
            if self.env.args.background >= 10:
                self.num_views = 1
                self.num_backgrounds = 5
        elif self.env.args.mode == "rotation":
            self.num_views = 1
            if self.env.args.background >= 10:
                self.num_backgrounds = 3
        elif self.env.args.mode == "all":
            self.num_views = 36





    def rotate_view(self, id_obj):
        if self.env.args.mode == "all":
            return [{"$type": "rotate_object_by", "angle": 10, "axis": "yaw", "id": id_obj}]

        command = []
        command.append({"$type": "rotate_object_by", "angle": random.uniform(0, 360), "axis": "yaw", "id": id_obj})
        if self.env.args.pitch:
            command.append({"$type": "rotate_object_by", "angle": random.uniform(0, 360), "axis": "pitch", "id": id_obj})
        if self.env.args.roll:
            command.append({"$type": "rotate_object_by", "angle": random.uniform(0, 360), "axis": "pitch", "id": id_obj})
        return command

    def vision_view(self):
        new_focus = math.sqrt((self.env.objects[0]["posx"] - self.env.cx) ** 2 + (self.env.objects[0]["posy"] - self.env.cy) ** 2
                              + (self.env.objects[0]["posz"] - self.env.cz) ** 2) - self.env.focus_corr

        command = [{"$type": "set_focus_distance", "focus_distance": new_focus}]
        command.append({"$type": "set_aperture", "aperture": self.env.compute_aperture(new_focus)})
        return command

    def init_rotate(self, id_obj, canonical_angle):
        if self.env.args.mode == "rotation" or self.env.args.mode == "all":
            return {"$type": "rotate_object_by", "angle": args.init_rotate+canonical_angle, "axis": "yaw", "id": id_obj}
        return {"$type": "rotate_object_by", "angle": random.uniform(0, 360)+canonical_angle, "axis": "yaw", "id": id_obj}

    def generate_dataset(self):
        root_dir = os.environ["DATASETS_LOCATION"]+self.env_name+"_dataset/"
        if not os.path.exists(root_dir):
            try:
                os.makedirs(root_dir)
            except:
                pass

        lib = ModelLibrarian(self.env.OBJECTS.lib_path)
        cpt_obj = 0

        with open(root_dir+"dataset.csv", 'a') as f:
            writer = csv.writer(f)
            for i in range(len(self.list_possible_names)):
                obj_num = int("".join(list(self.list_possible_names[i])[-3:]))
                if obj_num > self.env.categories_cpt[lib.get_record(self.list_possible_names[i]).wcategory] * 0.66:
                    continue
                cpt_obj += 1
                if cpt_obj > args.end or cpt_obj < args.begin:
                    continue
                if os.path.isdir(root_dir + self.list_possible_names[i]):
                    continue
                pos_indexes = random.choices(range(len(self.env.available_positions)),k=self.num_backgrounds)
                # for pos_index in range(len(self.env.available_positions)):
                id_obj = None
                for pos_index in pos_indexes:
                    command = self.env.set_center_position(pos_index)
                    command.extend(self.env.adjust_physics())
                    self.c.communicate(command)

                    possible_orientations = self.env.available_positions[pos_index]["pos"]
                    possible_orientations = random.choices(possible_orientations,k=1)
                    id_obj = self.write_position(writer, root_dir, possible_orientations, self.list_possible_names[i], id_obj=id_obj)
                command = [{"$type": "destroy_object", "id": id_obj}]
                if i %500 == 0:
                    command.append({"$type": "unload_unused_assets"})
                self.env.objects = []
                self.c.communicate(command)

        with open(root_dir+"dataset_test.csv", 'a') as f:
            writer = csv.writer(f)

            for i in range(len(self.list_possible_names)):
                obj_num = int("".join(list(self.list_possible_names[i])[-3:]))
                if obj_num <= self.env.categories_cpt[lib.get_record(self.list_possible_names[i]).wcategory] * 0.66:
                    continue
                cpt_obj += 1
                if cpt_obj > args.end or cpt_obj < args.begin:
                    continue
                if os.path.isdir(root_dir + "/" + self.list_possible_names[i]):
                    continue

                pos_indexes = random.choices(range(len(self.env.available_positions)),k=self.num_backgrounds)
                # for pos_index in range(len(self.env.available_positions)):
                id_obj=None
                for pos_index in pos_indexes:
                    command = self.env.set_center_position(pos_index)
                    command.extend(self.env.adjust_physics())
                    self.c.communicate(command)
                    possible_orientations = self.env.available_positions[pos_index]["pos"]
                    possible_orientations = random.choices(possible_orientations,k=1)
                    id_obj = self.write_position(writer, root_dir, possible_orientations,self.list_possible_names[i], id_obj=id_obj)
                command = [{"$type": "destroy_object", "id": id_obj}]
                if i %500 == 0:
                    command.append({"$type": "unload_unused_assets"})
                self.env.objects = []
                self.c.communicate(command)


    def write_position(self, writer, root_dir, position, oname, id_obj=None):
        dangle = (2 * math.pi) / self.env.number_objects
        eulerdangle = 360 / self.env.number_objects
        ###Starting
        # t= time.time()
        cpt = 0
        command = []


        ox, oz = self.env.init_distance, 0
        if id_obj is None:
            id_obj = self.c.get_unique_id()
            r = self.lib.get_record(oname)
            command.extend([{"$type": "add_object",
                    "name": oname,
                    "url": r.get_url(),
                    "scale_factor": r.scale_factor,
                    "position":{"x": self.env.cx + ox, "y": self.env.oy,"z": self.env.cz + oz},
                    "id": id_obj}])

                # self.c.get_add_object(oname, object_id=id_obj,
                # position={"x": self.env.cx + ox, "y": self.env.oy,"z": self.env.cz + oz}, library=self.env.OBJECTS.lib_path)])
            self.env.objects.append({"x": 360 / self.num_views, "y": 0, "z": 2, "posx": self.env.cx + ox, "posy": self.env.oy,
                                     "posz": self.env.cz + oz, "order": 0, "id": id_obj})
            command.append({"$type": "set_object_drag", "drag": 100, "angular_drag": 100, "id": id_obj})
            # self.resp = self.c.communicate(command)
            self.c.communicate(command)
        for num in position:
            ox = math.cos(num * dangle) * self.env.init_distance
            oz = math.sin(num * dangle) * self.env.init_distance
            command = []
            command.append({"$type": "teleport_object","id": id_obj,
                            "position": {"x": self.env.cx + ox, "y": self.env.oy, "z": self.env.cz + oz}})
            self.env.objects[0] = {"x": 360 / self.num_views, "y": 0, "z": 2, "posx": self.env.cx + ox, "posy": self.env.oy,"posz": self.env.cz + oz, "order": 0, "id": id_obj}
            command.append(self.init_rotate(id_obj, int(r.canonical_rotation["y"])))
            # command.extend(self.vision_view())
            # if self.env.args.rotate == 3:
            #     command.append({"$type": "rotate_object_by", "angle": 45, "axis": "yaw", "id": id_obj})
            command.append({"$type": "reset_sensor_container_rotation"})
            command = self.env.turn_avatar_to_position(command, num)

            # command.extend([{"$type": "rotate_avatar_to_euler_angles", "avatar_id": self.env.avatar_id,
            #                  "euler_angles": {"y": self.env.compute_constant_angle() - num * eulerdangle, "x": self.env.y_look_at,
            #                                   "z": 0}}])
            command.extend(self.env.noisy_action(self.env.action_space.sample()))
            command.append({"$type": "set_focus_distance",  "focus_distance": self.env.diagonal_init_distance - self.env.focus_corr})
            command.append({"$type": "set_aperture", "aperture": self.env.compute_aperture(self.env.diagonal_init_distance)})

            cpt += 1
            self.resp = self.c.communicate(command)

            filename = self.save_img(root_dir + oname + "/", oname, str(cpt) + "_" + str(num), str(self.env.current_center))
            writer.writerow([filename+".png", str(id_obj), oname, str(num)+"_"+str(self.env.current_center), str(self.env.current_center), "".join(list(str(oname))[:-4])])
            for j in range(self.num_views - 1):
                command = []
                command.extend(self.rotate_view(id_obj))
                command.append({"$type": "teleport_object", "id": id_obj,
                                "position": {"x": self.env.cx + ox, "y": self.env.oy, "z": self.env.cz + oz}})
                command.append({"$type": "reset_sensor_container_rotation"})
                command = self.env.turn_avatar_to_position(command, num)
                command.extend(self.env.noisy_action(self.env.action_space.sample()))
                cpt += 1
                self.resp = self.c.communicate(command)

                filename = self.save_img(root_dir + oname + "/", oname, str(cpt) + "_" + str(num), str(self.env.current_center))
                writer.writerow([filename+".png", str(id_obj), oname, str(num)+"_"+str(self.env.current_center), str(self.env.current_center), "".join(list(str(oname))[:-4])])
        return id_obj

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