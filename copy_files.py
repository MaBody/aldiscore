import shutil
import pathlib
import os
import time
from tqdm import tqdm

SOURCE = pathlib.Path("/hits/fast/cme/bodynems/data/output/treebase")
DESTINATION = pathlib.Path("/home/bodynems/data/output/treebase")


def copy_files():
    subdirs = os.listdir(SOURCE)
    subdirs.sort()

    if not os.path.exists(DESTINATION):
        os.makedirs(DESTINATION, exist_ok=True)

    for i in tqdm(range(len(subdirs))):
        subdir = subdirs[i]
        if (i < len(subdirs) - 1) and os.path.exists(DESTINATION / subdirs[i + 1]):
            continue
        try:
            if os.path.isfile(SOURCE / subdir):
                if os.path.exists(DESTINATION / subdir):
                    os.remove(DESTINATION / subdir)
                shutil.copy(SOURCE / subdir, DESTINATION / subdir)
            elif os.path.isdir(SOURCE / subdir):
                if os.path.exists(DESTINATION / subdir):
                    shutil.rmtree(DESTINATION / subdir)
                shutil.copytree(SOURCE / subdir, DESTINATION / subdir)
            else:
                print(f"Unknown file type for {SOURCE / subdir}")
        except Exception as e:
            pass
            # print(f"Exception in dir '{subdir}'!")


if __name__ == "__main__":
    copy_files()
