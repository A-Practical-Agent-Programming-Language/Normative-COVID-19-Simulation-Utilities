#!/bin/python3
import glob
import os.path
import shutil
from typing import Iterable, List, Tuple

import click


@click.command()
@click.option("-v", "verbose", flag_value="v", default=False)
@click.argument(
    "src",
    nargs=-1,
)
@click.argument(
    "dest",
    nargs=1,
    type=click.Path(file_okay=True, dir_okay=True, exists=False)
)
class SelectCopy(object):
    """
    Recursively copy epicurve.sim2apl.csv and tick-averages.csv
    files from simulation output directories, maintaining directory hierarchy
    """
    target_files = ["epicurve.sim2apl.csv", "tick-averages.csv"]
    overwrite_ok = False

    def __init__(self, verbose: bool, src: Iterable[str], dest: str):
        self.verbose = verbose
        self.src = src
        self.dest = dest

        self.files = self.find_files()
        self.copy_files()

    def find_files(self) -> Iterable[Tuple[str, str]]:
        files: List[Tuple[str, str]] = list()
        for src in self.src:
            all_files = glob.glob(src)
            if not len(all_files):
                print("No such file:", src)
            for path in glob.glob(src):
                files.append((path, self.get_target(src, path)))

        return files

    def get_target(self, src: str, path: str) -> str:
        if os.getcwd() in src:
            return os.path.join(self.dest, src[len(os.getcwd()):], path)
        else:
            return os.path.join(self.dest, path[src.index("*"):])

    def copy_files(self):
        for (src, dest) in self.files:
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            if self.__can_write(dest):
                if self.verbose:
                    print(dest)
                shutil.copy2(src, dest)

    def __can_write(self, dest: str) -> bool:
        if os.path.exists(dest):
            if not self.overwrite_ok:
                print(f"File {dest} exists.")
                prompt = click.prompt("Overwrite? [y/N/a] (a = overwrite this and all future files)")
                if not prompt.lower() in ['y', 'n', 'a']:
                    return self.__can_write(dest)
                elif prompt.lower() == 'y':
                    return True
                elif prompt.lower() == 'a':
                    self.overwrite_ok = True
                    return True
                else:
                    return False
        return True


if __name__ == "__main__":
    SelectCopy()
