from pathlib import Path
import json
from tdw.librarian import ModelLibrarian, ModelRecord
from tdw.asset_bundle_creator import AssetBundleCreator
from typing import Union, List, Optional
from argparse import ArgumentParser
import csv
from tqdm import tqdm

from tdw.librarian import ModelLibrarian


class LocalObject:
    @staticmethod
    def run():
        # Create the asset bundle and the record.
        asset_bundle_paths, record_path = AssetBundleCreator().create_asset_bundle("cube.fbx", True, 123, "", 1)
        # Get the name of the bundle for this platform. For example, Windows -> "StandaloneWindows64"
        bundle = SYSTEM_TO_UNITY[system()]
        # Get the correct asset bundle path.
        for p in asset_bundle_paths:
            # Get the path to the asset bundle.
            if bundle in str(p.parent.resolve()):
                url = "file:///" + str(p.resolve())

                # Launch the controller.
                c = Controller()
                c.start()
                # Create the environment.
                # Add the object.
                commands = [{"$type": "create_empty_environment"},
                            {"$type": "add_object",
                             "name": "cube",
                             "url": url,
                             "scale_factor": 1,
                             "id": c.get_unique_id()}]
                # Create the avatar.
                commands.extend(TDWUtils.create_avatar(position={"x": 0, "y": 0, "z": -3.6}))
                c.communicate(commands)
                return

    @staticmethod
    def run_all():
        src = "../../datasets/generate_scene_CRIB.fbx"
        dest = "../resources/toy_library"

        if not isinstance(src, Path):
            src = Path(src)
        dest = dest
        if not isinstance(dest, Path):
            dest = Path(dest)

        if not dest.exists():
            dest.mkdir(parents=True)

        library_path = dest.joinpath("records.json")
        records = ModelLibrarian(library=str(library_path.resolve())).records
        path = "../resources/tdw_toy_library.json"
        ModelLibrarian.create_library(description="Toy library", path=path)
        lib = ModelLibrarian(library=path)


class _ShapeNet:
    def __init__(self, src: Union[str, Path], dest: Union[str, Path]):
        """
        :param src: The source path or directory.
        :param dest: The root destination directory for the library file and asset bundles.
        """

        self.src = src
        if not isinstance(self.src, Path):
            self.src = Path(self.src)
        self.dest = dest
        if not isinstance(self.dest, Path):
            self.dest = Path(self.dest)
        if not self.dest.exists():
            self.dest.mkdir(parents=True)

        self.library_path = self.dest.joinpath("records.json")

    def run(self, batch_size: int = 1000, vhacd_resolution: int = 8000000, first_batch_only: bool = False) -> None:
        """
        Create a library file if one doesn't exist yet. Then generate asset bundles.

        :param batch_size: The number of models per batch.
        :param vhacd_resolution: Higher value=better-fitting colliders and slower build process.
        :param first_batch_only: If true, output only the first batch. Useful for testing purposes.
        """

        if not self.library_path.exists():
            self.create_library()
        self.create_asset_bundles(batch_size=batch_size, vhacd_resolution=vhacd_resolution,
                                  first_batch_only=first_batch_only)

    def create_library(self) -> ModelLibrarian:
        raise Exception()

    def _get_librarian(self, description: str) -> ModelLibrarian:
        ModelLibrarian.create_library(description, self.library_path)
        print("Adding records to the library...")
        return ModelLibrarian(str(self.library_path.resolve()))

    def _get_url(self, wnid: str, name: str, platform: str) -> str:
        dest = self.dest.joinpath(wnid + "/" + name + "/" + platform)
        return "file:///" + str(dest.resolve()).replace("\\", "/")

    def create_asset_bundles(self, batch_size: int = 1000, vhacd_resolution: int = 8000000,
                             first_batch_only: bool = False) -> None:
        """
        Convert all .obj files into asset bundles.

        :param batch_size: The number of models per batch.
        :param vhacd_resolution: Higher value=better-fitting colliders and slower build process.
        :param first_batch_only: If true, output only the first batch. Useful for testing purposes.
        """

        records = ModelLibrarian(library=str(self.library_path.resolve())).records
        a = AssetBundleCreator(quiet=True)

        pbar = tqdm(total=len(records))
        while len(records) > 0:
            # Get the next batch.
            batch: List[ModelRecord] = records[:batch_size]
            records = records[batch_size:]

                # Process the .obj
                obj_path = self._get_obj(record)
                # Move the files and remove junk.
                a.move_files_to_unity_project(None, model_path=obj_path, sub_directory=f"models/{record.name}")
            # Creating the asset bundles.
            a.create_many_asset_bundles(str(self.library_path.resolve()), cleanup=True,
                                        vhacd_resolution=vhacd_resolution)
            pbar.update(len(batch))

            # Process only the first batch of models.
            if first_batch_only:
                break
        pbar.close()

class ToyDataset(_ShapeNet):
    """
    Generate asset bundles from Toy-200 dataset.
    """
    def _get_obj(self, record: ModelRecord) -> Path:
        return self.src.joinpath(f"models/{record.name}.obj")

    def create_library(self) -> ModelLibrarian:
        lib = self._get_librarian("Toy200")
        first_time_only = True
        for i in range(200):
            record = ModelRecord()
            record.name = "toy"+str(i)
            record.wcategory = "wcategory"
            record.wnid = "wnid"
            record.urls[platform] = self._get_url(record.wnid, record.name, platform)
            lib.add_or_update_record(record, overwrite=False, write=False)
        # Write to disk.
        lib.write(pretty=False)
        return lib

if __name__ == "__main__":
    LocalObject.run()
