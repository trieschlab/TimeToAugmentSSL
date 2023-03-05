import argparse
import copy
import csv
import math
import sys, os
import time

from tdw.add_ons.embodied_avatar import EmbodiedAvatar

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")
from tools.arguments import str2bool, parse_datasets

from envs.foveated_vision import FoveatedVision
from envs.objects import Objects20, Objects1
import random
from subprocess import Popen
import numpy as np
import gym

from tdw.add_ons.third_person_camera import ThirdPersonCamera
from tdw.controller import Controller
from tdw.output_data import OutputData, Images, Transforms
from tdw.release.build import Build
from tdw.tdw_utils import TDWUtils
from tdw.librarian import ModelLibrarian
import socket
from contextlib import closing


from tools.utils import build_envname, get_env_object
from resources.rooms.simple_rooms import living_room_simple


class ToysObjects(gym.Env):

    # OBJECTS = Objects20
    Y_POS = 0.6508

    def __init__(self, args, training = True):
        super().__init__()
        self.OBJECTS = get_env_object(args)
        # if not hasattr(self,"list_possible_names"):
        # self.available_positions = [{"pos": [0, 1, 2, 3, 4, 5], "center": (0, 0.4, 0)}]
        self.available_positions = [{"pos": [0, 1, 2, 3, 4, 5], "center": (0, ToysObjects.Y_POS, 0)}]
        # self.available_positions = [{"pos": [0, 1, 2, 3, 4, 5], "center": (0, 0.7, 0)}]
        self.list_possible_names = self.OBJECTS.get_records_name(training=training)
        self.avatar_id = "a"
        self.action_space = gym.spaces.Discrete(args.num_actions)
        self.img_size = 128
        self.args=args
        self.binocular_radius = 0.1
        self.incline_rot = 0
        # self.img_size = 512

        self.observation_space = gym.spaces.Box(
            low=0.,
            high=1.,
            shape=(3 if not self.args.binocular else 6, self.img_size, self.img_size) if not self.args.batch_mode else (3, 160, 120),
            dtype='uint8'#torch.uint8
        )
        # build_info = RemoteBuildLauncher.launch_build(5556, "127.0.0.1", "127.0.0.1")
        # self.c = Controller(check_version=False, port=build_info["build_port"], launch_build=False)
        port = self._find_free_port()
        if self.args.batch_mode:
            Popen([str(Build.BUILD_PATH.resolve()), "-port "+str(port), "-force-glcore42", "-batchmode"])
            # Popen([str(Build.BUILD_PATH.resolve()), "-port "+str(port), "-batchmode"])
        # else:
        elif not args.cluster_mode:
            Popen([str(Build.BUILD_PATH.resolve()), "-port "+str(port), "-force-glcore42", "-screen-height 128", "-screen-width 128"])
        else:
            # Popen([str(Build.BUILD_PATH.resolve()), "-port "+str(port), "-force-glcore42", "-screen-height 128", "-screen-width 128"])
            # import subprocess
            # print("before")
            # bash_command = "VGL_DISPLAY=:1 xvfb-run -a -s '-screen 0 256x256x24:32 +extension GLX +extension RANDR +render -noreset' vglrun ~/tdw_build/TDW/TDW.x86_64 -port="+str(port)+" -force-glcore42 -screen-height=128 -screen-width=128"
            bash_command = "xvfb-run -a -s '-screen 0 256x256x24:32 +extension GLX +extension RANDR +render -noreset' ~/tdw_build/TDW/TDW.x86_64 -port="+str(port)+" -force-glcore42 -screen-height=128 -screen-width=128"
            Popen([bash_command],shell=True)

        self.c = Controller(check_version=False, port=port, launch_build=False)
        # self.c = Controller(check_version=False, port=1075, launch_build=False)
        self.init_distance = 1#math.sqrt(3)*0.6*args.closeness #1
        # self.y_look_at = -15 if background == 0 else 20/closeness
        self.y_look_at = 20/args.closeness if not self.args.incline else self.incline_rot
        self.number_objects = 6
        self.objects = []
        # self.oy=0.2 if mode == 0 else 0.3 #0.2
        self.oy=0.3 if args.background != 0 and args.background != 5 else -0.2
        # self.oy=0.3
        self.above_rendering = False
        self.above_camera_position = {"x": -2, "y": 8.7, "z":0 }
        self.rendering = True
        self.current_center = 0
        self.angle_x, self.angle_y = 0, 0
        self.aperture = self.args.aperture
        self.foveation = self.args.foveation
        self.lib = ModelLibrarian(self.OBJECTS.lib_path)
        self.categories_cpt = {}
        self.categories_order = {}

        cpt_cat=0
        for r in self.lib.records:
            if r.wcategory not in self.categories_cpt:
                self.categories_cpt[r.wcategory] = 0
                self.categories_order[r.wcategory] = cpt_cat
                cpt_cat+=1
            self.categories_cpt[r.wcategory] += 1
        with open("number_to_cat.csv", 'a') as f:
            writer = csv.writer(f)
            for k, a in self.categories_order.items():
                writer.writerow([k, a])
        # import csv
        # with open("objects_weights_color.csv", 'a') as f:
        #     writer = csv.writer(f)
        #     ar = [0]*cpt_cat
        #     for k in range(len(self.list_possible_names)):
        #         ar[self.categories_order[k]]=self.categories_cpt[k]
        #     writer.writerow(ar)

        self.number_categories = len(self.categories_cpt)
        self.cx, self.cy, self.cz = self.available_positions[self.current_center]["center"]
        self.override_params()

        commands =[]
        commands = self.create_room(commands)
        commands = self.create_avatars(commands)


        commands.extend([{"$type": "enable_reflection_probes", "enable": True},{"$type": "set_render_quality", "render_quality": 2 if not self.args.quality else 10}])
        if not self.args.batch_mode:
            commands.extend([{"$type": "set_screen_size", "width": self.img_size, "height": self.img_size}])
        # commands.extend([{"$type": "set_screen_size", "width": 640, "height": 480}])
        # self.observation_space = gym.spaces.Box(
        #     low=0.,high=1.,shape=(3, 640, 480),dtype='uint8'#torch.uint8
        # )

        self.resp = self.c.communicate(commands)
        print("End loading the scene")

    def override_params(self):
        pass

    def create_room(self,commands):
        size = 6 if self.args.background != 3 else 10

        ###Start creation of the room
        if self.args.background == 1:
            commands.append(TDWUtils.create_empty_room(size, size))
            commands.extend([self.c.get_add_material("parquet_long_horizontal_clean"),{"$type": "set_proc_gen_floor_material", "name": "parquet_long_horizontal_clean"}])
        elif self.args.background == 2:
            commands.append(TDWUtils.create_empty_room(size, size))
        elif self.args.background == 3 or self.args.background == 4 or self.args.background == 6:
            commands = living_room_simple(self.c, commands)
            if self.args.background == 4:
                self.available_positions = [{"pos": [0,1], "center": (0, ToysObjects.Y_POS, 0)}]
            if self.args.background == 6:
                self.number_objects = 12
                self.available_positions = [{"pos": [0,1,2,3,4,5,6,7,8,9,10,11], "center": (0, ToysObjects.Y_POS, 0)}]
        elif self.args.background == 0 or self.args.background == 5:
            commands.append({"$type": "create_empty_environment"})
            commands.append({"$type": "simulate_physics","value": False})
            if self.args.background == 5:
                self.available_positions = [{"pos": [0], "center": (0,ToysObjects.Y_POS, 0)}]
        elif self.args.background == 7:
            self.available_positions = [{"pos": [0], "center": (0, ToysObjects.Y_POS, 0)}]
            commands.append(self.c.get_add_scene(scene_name="mm_craftroom_1a"))#,
            commands.append(self.c.get_add_hdri_skybox(skybox_name="flower_road_4k"))

        else:
            raise Exception("This background does not exist for this environment")
        return commands

    def create_avatars(self, commands):
        # commands.extend(TDWUtils.create_avatar(avatar_type="A_Img_Caps", avatar_id=self.avatar_id,position={"x": self.cx, "y": self.cy, "z": self.cz}))
        # commands.extend(TDWUtils.create_avatar(avatar_type="A_Simple_Body", avatar_id=self.avatar_id,position={"x": self.cx, "y": self.cy, "z": self.cz}))
        self.c.add_ons.append(EmbodiedAvatar(avatar_id=self.avatar_id,position={"x": self.cx, "y": self.cy, "z": self.cz},scale_factor= {"x": 1, "y": 0, "z": 0.7}))
        # self.c.add_ons.append(ThirdPersonCamera(avatar_id="b", position=self.above_camera_position, look_at={"x": self.cx, "y": self.cy, "z": self.cz})
        # commands.extend([{"$type": "change_avatar_body", "body_type": "Capsule", "avatar_id": self.avatar_id}])
        self.c.communicate(commands)
        commands = []
        commands.extend([{"$type": "set_avatar_drag","drag": 10,"angular_drag": 20,"avatar_id": self.avatar_id}])
        commands.append({"$type": "set_pass_masks", "avatar_id": "a", "pass_masks": ["_img"]})

        commands.append({"$type": "send_rigidbodies","frequency": "never"})
        if not self.args.binocular:
            commands.extend([{"$type": "send_images", "ids": ["a","b"], "frequency": "always"},
                             {"$type": "send_image_sensors", "ids": [], "frequency": "always"},
                             {"$type": "enable_image_sensor", "enable": True, "sensor_name": "SensorContainer",
                              "avatar_id": "a"}])
        if self.args.aperture == 0.:# and not self.postprocess:
            commands.append({"$type": "set_post_process", "value": False})
        if self.args.aperture != 0.:
            # commands.append({"$type": "set_focus_distance",  "focus_distance": self.init_distance})
            # commands.append({"$type": "set_focus_distance",  "focus_distance": self.init_distance*math.pow(self.args.closeness,0.4)})
            # commands.append({"$type": "set_focus_distance",  "focus_distance": self.init_distance*math.pow(self.args.closeness,0.7)})
            commands.append({"$type": "set_focus_distance",  "focus_distance": self.init_distance*math.pow(self.args.closeness,0.4)})
            commands.append({"$type": "set_aperture", "aperture": self.aperture})
            # commands.append({"$type": "set_contrast", "contrast": self.contrast})
        if self.foveation:
            self.foveated_vision = FoveatedVision()
        if self.above_rendering:
            self.camera = ThirdPersonCamera(avatar_id="b", position=self.above_camera_position, look_at={"x": self.cx, "y": self.cy, "z": self.cz})
            # camera = ThirdPersonCamera(avatar_id="b", position={"x": -1, "y": 8.7, "z": -1}, look_at={"x": self.cx, "y": self.cy, "z": self.cz})
            self.c.add_ons.append(self.camera)
        if self.args.binocular:
            self.second_camera = ThirdPersonCamera(avatar_id="c", position={"x": self.cx+20, "y": 0, "z": self.cz})
            self.third_camera = ThirdPersonCamera(avatar_id="d", position={"x": self.cx-20, "y": 0, "z": self.cz})

            self.c.add_ons.append(self.second_camera)
            self.c.add_ons.append(self.third_camera)
            commands.extend([{"$type": "send_images", "ids": ["c", "d"], "frequency": "always"},
                             # {"$type": "send_image_sensors", "ids": [], "frequency": "always"},
                             #                  {"$type": "enable_image_sensor", "enable": True, "sensor_name": "SensorContainer",
                             #                   "avatar_id": "a"}])
                             # {"$type": "send_image_sensors", "ids": [], "frequency": "always"},
                             # {"$type": "enable_image_sensor", "enable": True, "sensor_name": "SensorContainer","avatar_id": "c"},
                             # {"$type": "enable_image_sensor", "enable": True, "sensor_name": "SensorContainer","avatar_id": "d"}
                             ])
        return commands

    @classmethod
    def get_records_name(cls):
        raise Exception()

    def get_object_weights(self):
        if self.args.obj_mode == 0:
            #Same weight for all objects
            return [1]*len(self.list_possible_names)
        elif self.args.obj_mode == 1:
            # Differently weighted probabilities according to the object
            return [1 + int(i * 0.12) for i in range(1, len(self.list_possible_names) + 1)]
        elif self.args.obj_mode == 2:
            #One unique object per episode
            number = random.randint(0, len(self.list_possible_names) - 1)
            t = [0]*len(self.list_possible_names)
            t[number]= 1
            return t

    def compute_constant_angle(self):
        return 90

    def turn_avatar_to_position(self, command, direction = None):
        eulerdangle = 360 / self.number_objects

        if direction is None:
            direction = self.available_positions[self.current_center]["pos"][self.posorder]
        # look = self.y_look_at
        command.append({"$type": "rotate_avatar_to_euler_angles", "avatar_id": self.avatar_id, "euler_angles": {
            "y": self.compute_constant_angle() - direction * eulerdangle, "x": self.y_look_at, "z": 0}})
        if self.args.binocular:
            # ox = math.cos(i * dangle) * self.init_distance
            # oz = math.sin(i * dangle) * self.init_distance
            cam_angle = math.pi * (self.compute_constant_angle() - direction * eulerdangle) / 180
            mod_x_c = -self.binocular_radius * math.cos(cam_angle)
            mod_z_c = self.binocular_radius * math.sin(cam_angle)
            mod_x_d = -self.binocular_radius * math.cos(cam_angle + math.pi)
            mod_z_d = self.binocular_radius * math.sin(cam_angle + math.pi)
            # self.second_camera.teleport({"x": self.cx, "y": 0.325, "z": self.cz})
            # self.third_camera.teleport({"x": self.cx, "y": 0.325, "z": self.cz })
            self.second_camera.teleport({"x": self.cx +mod_x_c, "y": self.cy, "z": self.cz+mod_z_c})
            self.third_camera.teleport({"x": self.cx+mod_x_d, "y": self.cy, "z": self.cz+mod_z_d })
            c_angle = 90 - 180 * math.atan((self.diagonal_init_distance - self.focus_corr) / self.binocular_radius) / math.pi
            # c_angle = 0
            look = 25.65 if not self.args.incline else self.incline_rot
            command.extend([{"$type": "rotate_avatar_to_euler_angles", "avatar_id": "c",
                             "euler_angles": {"y": self.compute_constant_angle() - direction * eulerdangle + c_angle,"x": look, "z": 0}}])
            command.extend([{"$type": "rotate_avatar_to_euler_angles", "avatar_id": "d",
                             "euler_angles": {"y": self.compute_constant_angle() - direction * eulerdangle - c_angle,"x":look, "z": 0}}])
            # command.extend([{"$type": "rotate_avatar_to_euler_angles", "avatar_id": "c",
            #                  "euler_angles": {"y":self.compute_constant_angle() - direction * eulerdangle - c_angle,"x": 25.65, "z": 0}}])
        return command


    def reset_center_position(self):
        return self.set_center_position(self.current_center)

    def set_center_position(self, pos):
        self.current_center = pos
        self.cx, self.cy, self.cz = self.available_positions[self.current_center]["center"]
        if self.above_rendering:
            self.above_camera_position["x"] = self.cx
            self.above_camera_position["z"] = self.cz
            self.camera.teleport(self.above_camera_position)
            self.camera.look_at({"x": self.cx, "y": self.cy, "z": self.cz})
        ###Avatar position
        # command =[]
        command = [{"$type": "teleport_avatar_to", "avatar_id": self.avatar_id,"position": {"x": self.cx, "y": self.cy, "z": self.cz}}]
        ### Avatar rotation
        # self.posorder = random.choices(self.available_positions[self.current_center]["pos"],k=1)[0]
        self.posorder = random.randint(0,len(self.available_positions[self.current_center]["pos"])-1)
        self.turn_avatar_to_position(command)
        # Make avatar on the floor
        return command



    def reset(self):
        command=[]
        command.extend(self.reset_center_position())

        for obj in self.objects:
            command.append({"$type": "destroy_object", "id": obj["id"]})
        self.c.communicate(command)

        command=[]
        self.objects = []
        weights = self.get_object_weights()
        # for i in range(self.number_objects):
        for i in self.available_positions[self.current_center]["pos"]:
            obj, command_obj = self.generate_object(weights, i)
            # self.objects[i]=obj
            self.objects.append(obj)
            command.extend(command_obj)
        command.append({"$type": "step_physics", "frames": 5})
        self.resp = self.c.communicate(command)
        img= self.render()

        return img.swapaxes(0,2).swapaxes(1,2)#
        # return img.transpose()

    def generate_object(self, weights, i):
        command = []
        dangle = (2 * math.pi) / self.number_objects
        ox = math.cos(i * dangle) * self.init_distance
        oz = math.sin(i * dangle) * self.init_distance
        oname_number = random.choices(list(range(0, len(self.list_possible_names))), weights=weights)[0]
        oname = self.list_possible_names[oname_number]
        obj_id = self.c.get_unique_id()
        category = self.categories_order[self.lib.get_record(oname).wcategory]
        command.append(self.c.get_add_object(oname, object_id=obj_id, library=self.OBJECTS.lib_path,
                                             position={"x": self.cx + ox, "y": self.oy, "z": self.cz + oz},
                                             rotation={"x": 0, "y": 0, "z": 0}))

        command.append({"$type": "set_object_drag", "drag": 100, "angular_drag": 100, "id": obj_id})
        if self.args.random_orient:
            command.append({"$type": "rotate_object_by", "angle": random.uniform(0, 360), "axis": "yaw", "id": obj_id})

        obj = {"x": random.uniform(self.args.min_angle_speed, self.args.max_angle_speed), "y": 0, "z": 0, "posx": self.cx + ox,
                 "posy": self.oy, "posz": self.cz + oz, "order": i,"id": obj_id, "name": oname, "oid": oname_number, "category": category}
        return obj, command

    def turn(self, angle):
        command = []
        self.posorder = (self.posorder + angle) % len(self.objects)
        obj = self.objects[self.posorder]
        if self.args.switch and angle != 0:
            command.append({"$type": "destroy_object", "id": obj["id"]})
            self.c.communicate(command)
            weights = self.get_object_weights()
            pos_angle = self.available_positions[self.current_center]["pos"][self.posorder]
            self.objects[self.posorder], command = self.generate_object(weights, pos_angle)
            obj = self.objects[self.posorder]
        eulerdangle = 360 / self.number_objects
        command.append({"$type": "rotate_avatar_to_euler_angles", "avatar_id": self.avatar_id,
                         "euler_angles": {"y": self.compute_constant_angle() - obj["order"] * eulerdangle,
                                          "x": self.y_look_at if not self.args.include else self.incline_rot, "z": 0}})
        return command

    def render(self, mode="human", resp = None, output_dir = None, file_name =None):
        resp = resp if resp is not None else self.resp
        for r in resp[:-1]:
            r_id = OutputData.get_data_type_id(r)
            if r_id == "imag":
                images = Images(r)
                if images.get_avatar_id()=="a":
                    # index = 0 if self.above_rendering or not self.rendering else 1
                    if self.args.batch_mode:
                        pil_image = TDWUtils.get_pil_image(images=images, index=0)
                        img = np.array(pil_image)
                        img = img[::4,::4,:]
                    else:
                        pil_image = TDWUtils.get_pil_image(images=images, index=0)
                        img = np.array(pil_image)
                    if output_dir is not None:
                        TDWUtils.save_images(images=images,output_directory=output_dir,filename=file_name, append_pass=False)
                if self.args.binocular and images.get_avatar_id()=="c":
                    pil_image = TDWUtils.get_pil_image(images=images, index=0)
                    img_c = np.array(pil_image)
                    if output_dir is not None:
                        TDWUtils.save_images(images=images, output_directory=output_dir, filename=file_name+"_vc",append_pass=False)
                if self.args.binocular and images.get_avatar_id()=="d":
                    pil_image = TDWUtils.get_pil_image(images=images, index=0)
                    img_d = np.array(pil_image)
                    if output_dir is not None:
                        TDWUtils.save_images(images=images, output_directory=output_dir, filename=file_name + "_vd",append_pass=False)

            # print(r_id)
            # if r_id== "avsb":
            #     a = AvatarSimpleBody(r)
            #     self.avatar_data = a
            #     print("simple_body")
            #
            # if r_id == "avki":
            #     a = AvatarKinematic(r)
            #     self.avatar_data = a
            #     print("kinematic")
            # if r_id == "avnk":
            #     a = AvatarNonKinematic(r)
            #     self.avatar_data = a
            #     print("non kinematic")

            if r_id == "tran":
                self.transforms = Transforms(r)


            # if r_id == "coll":
            #     c = Collision(r)
            #     print("COLLISION", c.get_collider_id())
            # if r_id == "enco":
            #     c = EnvironmentCollision(r)
            #     print("ENCO",c.get_object_id())

                        # sleep(1)
        if self.foveation:
            x_fov, y_fov = img.shape[0] // 2, img.shape[1] // 2
            img = self.foveated_vision.foveat_img(img, [(x_fov, y_fov)], p=7.5, k=3, alpha=1)
        if self.args.binocular:
            img = np.concatenate((img_c,img_d),axis=2)

        return img

    def make_objects_turn(self, action):
        commanderot = []
        if self.args.rotate == 1:
            for obj in self.objects:
                commanderot.extend([{"$type": "rotate_object_by", "angle": obj["x"], "axis": "yaw", "id": obj["id"]}])  # good rotation
        if self.args.rotate == 2:
            commanderot.extend([{"$type": "rotate_object_by", "angle":  random.uniform(self.args.min_angle_speed, self.args.max_angle_speed), "axis": "yaw", "id": self.objects[self.posorder]["id"]}])  # good rotation

            # commanderot.extend([{"$type": "rotate_object_by","angle":rot["z"],"axis": "pitch", "id":ob}])
                # commanderot.extend([{"$type": "rotate_object_by","angle":rot["z"],"axis": "roll", "id":ob}])
        return commanderot

    def noisy_action(self, action):
        command=[]
        if self.args.noise:
            self.angle_x, self.angle_y = self.args.noise*random.gauss(0, 2), self.args.noise*random.gauss(0, 2)
            command.append({"$type": "rotate_sensor_container_by", "angle": self.angle_x, "axis": "yaw", "avatar_id": self.avatar_id})
            command.append({"$type": "rotate_sensor_container_by", "angle": self.angle_y, "axis": "pitch", "avatar_id": self.avatar_id})
        # if self.args.noise == 2:
        #     new_angle_x = self.angle_x + random.gauss(0, 1)
        #     new_angle_y = self.angle_y + random.gauss(0, 1)
        #     self.angle_y = 3*random.gauss(0, 2), 3*random.gauss(0, 2)
        #     command.append({"$type": "rotate_sensor_container_by", "angle": self.angle_x, "axis": "yaw", "avatar_id": self.avatar_id})
        #     command.append({"$type": "rotate_sensor_container_by", "angle": self.angle_y, "axis": "pitch", "avatar_id": self.avatar_id})
        return command

    def step(self, action):
        command = []
        command.append({"$type": "reset_sensor_container_rotation"})
        self.angle_x, self.angle_y = 0, 0


        if action == 0:
            command.extend(self.turn(-1))
        elif action == 1:
            command.extend(self.turn(1))
        elif action == 2:
            command.extend(self.turn(0))
        elif action == 3:
            command.extend(self.turn(2))
        elif action == 4:
            command.extend(self.turn(-2))
        elif action == 5:
            command.extend(self.turn(3))
        command.extend(self.make_objects_turn(action))
        command.extend(self.noisy_action(action))
        self.resp = self.c.communicate(command)
        # self.c.add_ons.append(StepPhysics(num_frames=1))
        obs = self.render().swapaxes(0, 2).swapaxes(1,2)#.swapaxes(1,2)
        # obs = self.render().transpose()
        infos = self.get_infos()
        infos["fix"] = (action == 2)
        return obs, 0, False, infos# {"object_names": self.list_possible_names}

    def get_infos(self):
        infos = self.objects[self.posorder].copy()
        infos["position"] = self.available_positions[self.current_center]["pos"][self.posorder]
        infos["total_num_objects"] = len(self.list_possible_names)
        infos["total_num_positions"] = 6
        infos["total_num_categories"] = self.number_categories
        return infos

    @staticmethod
    def _find_free_port():
        """
        Returns a free port as a string.
        """
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.bind(("", 0))
            return int(s.getsockname()[1])

    def close(self):
        self.c.communicate([{"$type": "terminate"}])


class ContinuousToyObjects(ToysObjects):

    def __init__(self, args):
        super().__init__(args)
        self.action_space = gym.spaces.Box(np.array([-1]*2), np.array([1]*2))


    def make_objects_turn(self, action):
        commanderot = []
        angle = (action[1] + 1)/2 #Between 0 and 1
        angle = (angle * (self.args.max_angle_speed - self.args.min_angle_speed)) + self.args.min_angle_speed
        commanderot.extend([{"$type": "rotate_object_by", "angle": angle, "axis": "yaw", "id": self.objects[self.posorder]["id"]}])  # good rotation
        return commanderot

    def _sigmoid(self, val):
        return 1 / (1 + np.exp(-12*(val+0.4)))

    def _neg_sigmoid(self, val):
        return 1 - 1 / (1 + np.exp(-12*(val-0.4)))

    def step(self, action):
        command = []
        command.append({"$type": "reset_sensor_container_rotation"})
        self.angle_x, self.angle_y = 0, 0
        sigmoid = self._sigmoid(action[0])
        neg_sigmoid = self._neg_sigmoid(action[0])
        fix = False
        if random.uniform(0,1) > sigmoid:
            command.extend(self.turn(-1))
        elif random.uniform(0,1) > neg_sigmoid:
            command.extend(self.turn(1))
        else:
            command.extend(self.turn(0))
            fix = True

        command.extend(self.make_objects_turn(action))
        command.extend(self.noisy_action(action))
        self.resp = self.c.communicate(command)
        # self.c.add_ons.append(StepPhysics(num_frames=1))
        obs = self.render().swapaxes(0, 2).swapaxes(1,2)
        infos = self.get_infos()
        infos["fix"] = fix
        return obs, 0, False, infos# {"object_names": self.list_possible_names}

def get_label_names(args):
    return get_env_object(args).get_records_name()

class ToysDataset():

    def __init__(self, env_name, args):
        self.env_name = env_name
        self.settup(args)
        self.c = self.env.c
        self.lib = ModelLibrarian(self.env.OBJECTS.lib_path)
        self.list_possible_names = self.env.OBJECTS.get_records_name(training=False)
        self.generate_dataset()
        self.c.communicate({"$type": "terminate"})

    def settup(self, args):
        self.env = ToysObjects(args, training =False)
        # self.num_views = max(1, int(200//args.num_obj)*(6//len(self.env.available_positions[0]["pos"])))
        self.num_views = max(1, int(200//args.num_obj))#*(6//len(self.env.available_positions[0]["pos"])))
        # self.c.communicate([{"$type": "send_bounds", "ids": [id], "frequency": "always"}])

    def generate_dataset(self):
        root_dir = os.environ["DATASETS_LOCATION"]+self.env_name+"_dataset/"
        if not os.path.exists(root_dir):
            try:
                os.makedirs(root_dir)
            except:
                pass
        with open(root_dir+"dataset.csv", 'w') as f:
            writer = csv.writer(f)
            for pos_index in range(len(self.env.available_positions)):
                self.c.communicate(self.env.set_center_position(pos_index))
                self.c.communicate({"$type": "step_physics", "frames": 5})
                for i in range(len(self.list_possible_names)):
                    id_obj = self.write_position(writer, root_dir, self.env.available_positions[pos_index]["pos"], self.env.list_possible_names[i])
                    command = [{"$type": "destroy_object", "id": id_obj}]
                    self.env.objects = []
                    self.resp = self.c.communicate(command)


    def rotate_view(self, id_obj):
        return self.env.make_objects_turn(None)

    def vision_view(self):
        return []

    def init_rotate(self, id_obj):
        if self.env.args.rotate:
            return {"$type": "rotate_object_by", "angle": random.uniform(0, 360), "axis": "yaw", "id": id_obj}
        return {}

    def step_physics(self):
        return [{"$type": "step_physics", "frames": 5}]

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
            self.env.objects.append({"x": 360 / self.num_views, "y": 0, "z": 2, "posx": self.env.cx + ox, "posy": 0,
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
            command.extend(self.step_physics())

            command.append(self.init_rotate(id_obj))
            command.extend(self.vision_view())
            # if self.env.args.rotate == 3:
            #     command.append({"$type": "rotate_object_by", "angle": 45, "axis": "yaw", "id": id_obj})
            command.append({"$type": "reset_sensor_container_rotation"})
            command.extend([{"$type": "rotate_avatar_to_euler_angles", "avatar_id": self.env.avatar_id,
                             "euler_angles": {"y": self.env.compute_constant_angle() - num * eulerdangle, "x": self.env.y_look_at,
                                              "z": 0}}])
            command.extend(self.env.noisy_action(self.env.action_space.sample()))
            cpt += 1
            self.resp = self.c.communicate(command)

            filename = self.save_img(root_dir + oname + "/", oname, str(cpt) + "_" + str(num), str(self.env.current_center))
            writer.writerow([filename+".png", str(id_obj), oname, str(num)+"_"+str(self.env.current_center), str(self.env.current_center), "".join(list(str(oname))[:-4])])
            # time.sleep(5)

            for j in range(self.num_views - 1):
                command = []
                command.extend(self.rotate_view(id_obj))
                command.append({"$type": "reset_sensor_container_rotation"})
                command.extend([{"$type": "rotate_avatar_to_euler_angles", "avatar_id": self.env.avatar_id,
                                 "euler_angles": {"y": self.env.compute_constant_angle() - num * eulerdangle,"x": self.env.y_look_at, "z": 0}}])
                command.extend(self.env.noisy_action(self.env.action_space.sample()))
                cpt += 1
                self.resp = self.c.communicate(command)

                filename = self.save_img(root_dir + oname + "/", oname, str(cpt) + "_" + str(num), str(self.env.current_center))
                writer.writerow([filename+".png", str(id_obj), oname, str(num)+"_"+str(self.env.current_center), str(self.env.current_center), "".join(list(str(oname))[:-4])])
        return id_obj

    def save_img(self,directory,name,count, position):
        # Save the image.
        filename = f"{name}_{count}_{position}"
        if not os.path.exists(directory):
            try:
                os.makedirs(directory)
            except:
                pass
        # path = os.path.join(directory, filename)
        self.env.render(resp=self.resp, output_dir=directory, file_name=filename)
        # Image.fromarray(img).save(path)

        return filename

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
    dataset = ToysDataset(env_name, args)