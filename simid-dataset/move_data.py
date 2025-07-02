from pathlib import Path
import os
import shutil

# the path to the input dataset
# part1 scene1
PART1_SCENE1_WIFI = "F:\\dataset\\xrf55\\part1\\Scene1\\Scene1\\WiFi"
# part2 scene1
PART2_SCENE1_WIFI = "F:\\archive\\Scene1_part2\\WiFi"
# part1 scene2
PART1_SCENE2_WIFI = "F:\\dataset\\xrf55\\part1\\Scene2\\Scene2\\WiFi"
# part1 scene3
PART1_SCENE3_WIFI = "F:\\dataset\\xrf55\\part1\\Scene3\\Scene3\\WiFi"
# part1 scene4
PART1_SCENE4_WIFI = "F:\\dataset\\xrf55\\part1\\Scene4\\Scene4\\WiFi"

# the path to the output of the dataset
SCENE1 = "F:\\dataset\\xrf55_for_simid\\scene1\\"
SCENE2 = "F:\\dataset\\xrf55_for_simid\\scene2\\"
SCENE3 = "F:\\dataset\\xrf55_for_simid\\scene3\\"
SCENE4 = "F:\\dataset\\xrf55_for_simid\\scene4\\"

SCENE_DIRS = [
    Path(SCENE1),
    Path(SCENE2),
    Path(SCENE3),
    Path(SCENE4)
]

for dir_path in SCENE_DIRS:
    try:
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"The directory already exists or has been successfully created: {dir_path}")
    except OSError as e:
        print(f"Failed to create directory: {dir_path} message: {e}")

def copy_contents(src, dst):
    """
    Copies all contents of the source directory src to the destination directory dst.
    If the destination directory does not exist, it is created.
    """
    try:
        if not os.path.exists(dst):
            os.makedirs(dst)
            print(f"Target directory created: {dst}")
        else:
            print(f"The target directory already exists, the contents will be copied to: {dst}")

        for item in os.listdir(src):
            s = os.path.join(src, item)
            d = os.path.join(dst, item)

            try:
                if os.path.isdir(s):
                    shutil.copytree(s, d, dirs_exist_ok=True)
                    print(f"Copied directories: {s} -> {d}")
                else:
                    shutil.copy2(s, d)
                    print(f"Copied files: {s} -> {d}")
            except Exception as e:
                print(f"failed to copy {s} to {d} : {e}")

    except Exception as e:
        print(f"failed when copy {src} to {dst} : {e}")

# Move the two parts of scene1 data
# Define a list of source folders
source_dirs = [PART1_SCENE1_WIFI, PART2_SCENE1_WIFI]

# Iterate through each source folder and copy the contents to the target folder
for src_dir in source_dirs:
    if os.path.exists(src_dir):
        print(f"\nstart to copy the content in {src_dir} ...")
        copy_contents(src_dir, SCENE1)
    else:
        print(f"\nSource directory does not exist, skipping: {src_dir}")


# Move scene 2 3 4 data
mission_dirs = [(PART1_SCENE2_WIFI, SCENE2), (PART1_SCENE3_WIFI, SCENE3), (PART1_SCENE4_WIFI, SCENE4)]

# Traversal
for md in mission_dirs:
    if os.path.exists(md[0]):
        print(f"\nstart to copy the content from {md} ...")
        copy_contents(md[0], md[1])
    else:
        print(f"\nSource directory does not exist, skipping: {md}")
