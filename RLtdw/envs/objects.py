import csv
import os

from tdw.librarian import ModelLibrarian


class SixObjects:
    lib_path = "models_core.json"

    @classmethod
    def get_records_name(cls):
        return ["balance_double_doghouse", "easter4", "bag_3", "b04_ramlosa_bottle_2015_vray",
                                "buddah", "blue_satchal","basket_18inx18inx12iin_wood_mesh","baton_table_lamp_dark"]
        # return ["buddah"]


class Objects4k_untex_small:
    lib_path = os.environ["TOY_UTEX_LIBRARY_PATH"] + "/toys.json"

    @classmethod
    def get_records_name(cls, training=False, test=False, **kwargs):
        lib = ModelLibrarian(library=Objects4k_untex.lib_path)
        if not training:
            list_possible_names = [record.name for record in lib.records]
        else:
            list_possible_names = []
            name = "dataset" if not test else "dataset_test"
            with open(os.environ["DATASETS_LOCATION"]+"full_play4000_back5_app5_clo0.8_dataset/"+name+".csv") as f:
                csv_f = csv.reader(f, delimiter=",")
                prev_name = None
                for r in csv_f:
                    if r[2] != prev_name:
                        list_possible_names.append(r[2])
                    prev_name = r[2]
        # list_possible_names = ["drum_007"]
        return list_possible_names


class Objects4k_untex:
    lib_path = os.environ["TOY_UTEX_LIBRARY_PATH"] + "/toys.json"

    @classmethod
    def get_records_name(cls, training=False, test=False, **kwargs):
        # return ["dolphin_002"]
        lib = ModelLibrarian(library=Objects4k_untex.lib_path)
        if not training:
            list_possible_names = [record.name for record in lib.records]
        else:
            list_possible_names = []
            name = "dataset" if not test else "dataset_test"
            with open(os.environ["DATASETS_LOCATION"]+"full_play4000_back5_app5_clo0.8_dataset/"+name+".csv") as f:
                csv_f = csv.reader(f, delimiter=",")
                prev_name = None
                for r in csv_f:
                    if r[2] != prev_name:
                        list_possible_names.append(r[2])
                    prev_name = r[2]
        # list_possible_names = ["airplane_000"]
        return list_possible_names

    @classmethod
    def get_categories(cls):
        lib = ModelLibrarian(library=Objects4k_untex.lib_path)
        categories_order={}
        number_to_category = {}
        cpt_cat=0
        for r in lib.records:
            if r.wcategory not in categories_order:
                categories_order[r.wcategory] = cpt_cat
                number_to_category[cpt_cat] = r.wcategory
                cpt_cat+=1
        return categories_order, number_to_category

class Objects4k_tex:
    # self.lib_path = str(Path().resolve().joinpath("resources/toy_library/toys.json"))
    lib_path = os.environ["TOY_TEX_LIBRARY_PATH"] + "/toys.json"

    @classmethod
    def get_records_name(cls, training=False, **kwargs):
        lib = ModelLibrarian(library=Objects4k_tex.lib_path)
        if not training:
            list_possible_names = [record.name for record in lib.records]
        else:
            list_possible_names = []
            with open(os.environ["DATASETS_LOCATION"]+"full_play4001_back10_app5_clo0.8_dataset/dataset.csv") as f:
                csv_f = csv.reader(f, delimiter=",")
                prev_name = None
                for r in csv_f:
                    if r[2] != prev_name:
                        list_possible_names.append(r[2])
                    prev_name = r[2]
        # return ["stove_011"]
        return list_possible_names
        # return ["hat_014","hat_023","hat_006"]



        # return ["pen_000"]
        # return ["stove_000"]
        # return ["key_022", "laptop_000"], places 0 0 and 0 2  and 5 0, 2 1, 6 3

    @classmethod
    def get_categories(cls):
        lib = ModelLibrarian(library=Objects4k_tex.lib_path)
        categories_order={}
        number_to_category = {}
        cpt_cat=0
        for r in lib.records:
            if r.wcategory not in categories_order:
                categories_order[r.wcategory] = cpt_cat
                number_to_category[cpt_cat] = r.wcategory
                cpt_cat+=1
        return categories_order, number_to_category

class Objects100:
    # self.lib_path = str(Path().resolve().joinpath("resources/toy_library/toys.json"))
    lib_path = os.environ["TOY_LIBRARY_PATH"]+"/toys.json"

    @classmethod
    def get_records_name(cls, **kwargs):
        lib = ModelLibrarian(library=Objects100.lib_path)
        list_possible_names = [record.name for record in lib.records]
        return list_possible_names

class Objects80:
    # self.lib_path = str(Path().resolve().joinpath("resources/toy_library/toys.json"))
    lib_path = os.environ["TOY_LIBRARY_PATH"]+"/toys.json"

    @classmethod
    def get_records_name(cls, **kwargs):
        lib = ModelLibrarian(library=Objects100.lib_path)
        list_possible_names = [record.name for record in lib.records][0:90]
        #We remove toys that take a too long processing time
        to_removes = ["laptop_012","octopus_014","shark_004","frog_009","bottle_093","chicken_008","dragon_036",
                      "pen_028","grapes_012","guitar_041"]
        list_possible_names = [o for o in list_possible_names if o not in to_removes]
        return list_possible_names

class Objects40:
    # self.lib_path = str(Path().resolve().joinpath("resources/toy_library/toys.json"))
    lib_path = os.environ["TOY_LIBRARY_PATH"]+"/toys.json"

    @classmethod
    def get_records_name(cls, **kwargs):
        list_possible_names = Objects80.get_records_name()
        return list_possible_names[0:40]

class Objects20:
    lib_path = os.environ["TOY_LIBRARY_PATH"]+"/toys.json"

    @classmethod
    def get_records_name(cls, **kwargs):
        # lib = ModelLibrarian(library=Objects20.lib_path)
        # list_possible_names = [record.name for record in lib.records]
        # return ['submarine_007', 'trashcan_033', 'violin_009', 'tree_029', 'cupcake_003', 'lizard_001',
        # 'truck_002', 'train_020', 'robot_102', 'dolphin_010', 'monkey_001',  "dragon_036", "cookie_009",
        # 'bus_010', 'donut_032', 'microwave_014', 'airplane_021', 'sofa_000', 'panda_020', 'sink_007']
        # return ["sheep_027"]
        # return ["piano_037"]*20
        return ['submarine_007', 'trashcan_033', 'violin_009', 'tree_029', 'cupcake_003', 'lizard_001',
        'truck_002', 'train_020', 'robot_102', 'dolphin_010', 'monkey_001',  "sheep_027", "cookie_009",
        'bus_010', 'donut_032', 'microwave_014', 'airplane_021', 'sofa_000', 'panda_020', 'sink_007']
        #12 cookie 009, 15 microwaves , 16 airplane, 3 tree039, 4 cupcake_003, 2 violin, 18 panda
        #15 microwaves, panda, cookie,
        #s90: o11, o17, sofa 000,
        # return ["submarine_007"], we do not make [:20] to avoid the spade (too large) and replace it with the cookie
        #also removed the piano_037

class Objects1:
    lib_path = os.environ["TOY_LIBRARY_PATH"]+"/toys.json"

    @classmethod
    def get_records_name(cls, **kwargs):
        # lib = ModelLibrarian(library=Objects20.lib_path)
        # list_possible_names = [record.name for record in lib.records]
        return ["piano_037"]
        # return ["submarine_007"], we do not make [:20] to avoid the spade (too large) and replace it with the cookie


class Objects5:
    lib_path = os.environ["TOY_LIBRARY_PATH"]+"/toys.json"
    @classmethod
    def get_records_name(cls, **kwargs):
        lib = ModelLibrarian(library=Objects100.lib_path)
        list_possible_names = [record.name for record in lib.records]
        return list_possible_names[0:5]
