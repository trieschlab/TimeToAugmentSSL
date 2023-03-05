from tdw.tdw_utils import TDWUtils
import sys, os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")


def living_room_simple(c, commands):
    ids = -10000
    commands.append(TDWUtils.create_empty_room(6, 10))
    commands.extend([c.get_add_material("parquet_long_horizontal_clean"),
                     {"$type": "set_proc_gen_floor_material", "name": "parquet_long_horizontal_clean"}])
    commands.extend([c.get_add_material("anodic_powder_paint", library="materials_med.json"),
                     {"$type": "set_proc_gen_walls_material", "name": "anodic_powder_paint"}])

    commands.append(c.get_add_object("4ft_wood_shelving", object_id=ids+1, library="models_core.json",
                                          position={"x": 2, "y": 0, "z": 4}))
    commands.append(c.get_add_object("4ft_wood_shelving", object_id=ids+2, library="models_core.json",
                                          position={"x": 2, "y": 0, "z": 3}))
    commands.append(c.get_add_object("4ft_wood_shelving", object_id=ids+3, library="models_core.json",
                                          position={"x": 2, "y": 0, "z": 2}))
    commands.append(c.get_add_object("arflex_hollywood_sofa", object_id=ids+4, library="models_core.json",
                                          position={"x": 0, "y": 0, "z": -3}))
    commands.append(c.get_add_object("marble_table", object_id=ids+5, library="models_core.json",
                                          position={"x": 0, "y": 0, "z": -2.3}, rotation={"x": 0, "y": 0, "z": 0}))
    commands.append(c.get_add_object("b04_ramlosa_bottle_2015_vray", object_id=ids+6, library="models_core.json",
                              position={"x": 0, "y": 1, "z": -2.3}))
    commands.append(c.get_add_object("bakerparisfloorlamp03", object_id=ids+7, library="models_core.json",
                                          position={"x": -2, "y": 0, "z": -0.8}))
    commands.append(c.get_add_object("glass_table", object_id=ids+8, library="models_core.json",
                                          position={"x": -0.5, "y": 0, "z": 2.9}))
    commands.append(c.get_add_object("chair_willisau_riale", object_id=ids+9, library="models_core.json",
                                          position={"x": -1, "y": 0, "z": 3.3}, rotation={"x": 0, "y": 180, "z": 0}))
    commands.append(c.get_add_object("chair_willisau_riale", object_id=ids+10, library="models_core.json",
                                          position={"x": 0, "y": 0, "z": 3.3}, rotation={"x": 0, "y": 180, "z": 0}))
    commands.append(c.get_add_object("chair_willisau_riale", object_id=ids+11, library="models_core.json",
                                          position={"x": -1, "y": 0, "z": 2.4}))
    commands.append(c.get_add_object("chair_willisau_riale", object_id=ids+12, library="models_core.json",
                                          position={"x": 0, "y": 0, "z": 2.4}))
    commands.append(c.get_add_object("rh1", object_id=ids+13, library="models_core.json",
                                          position={"x": -2, "y": 0, "z": 1.5}, rotation={"x": 0, "y": 37, "z": 0}))
    commands.append(c.get_add_object("puzzle_box_composite", object_id=ids+14, library="models_core.json",
                                          position={"x": -2, "y": 0, "z": 0.5}))
    # commands.append(self.c.get_add_object("backpack", object_id=self.init_ids, library="models_core.json",
    #                                       position={"x": 2, "y": 0, "z": -1.5}))
    # self.init_ids+=1
    # commands.append(self.c.get_add_object("carpet_rug", object_id=self.init_ids, library="models_core.json",
    #                                       position={"x": 0, "y": 0.1, "z": 0}))
    return commands
    # self.init_ids+=1

def download_models_living_room_simple():
    return ["4ft_wood_shelving", "arflex_hollywood_sofa", "marble_table",
           "b04_ramlosa_bottle_2015_vray", "bakerparisfloorlamp03","glass_table", "chair_willisau_riale", "rh1","puzzle_box_composite"]

def download_materials_living_room_simple():
    return ["parquet_long_horizontal_clean", "anodic_powder_paint"]

#
#
#
# if __name__ == "__main__":
#     download_data()