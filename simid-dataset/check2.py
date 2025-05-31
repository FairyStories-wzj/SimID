"""
This program is aiming to check if the seperated CA, CP, CACP and CPCS are correct.
"""
import os
import filecmp

CA_PATH = "F:\\dataset\\xrf55_for_siamid\\CA"
CP_PATH = "F:\\dataset\\xrf55_for_siamid\\CP"
CACP_PATH = "F:\\dataset\\xrf55_for_siamid\\CACP"
CPCS_PATH = "F:\\dataset\\xrf55_for_siamid\\CPCS"
SCENE1_PATH = "F:\\dataset\\xrf55_for_siamid\\scene1_filtered"
SCENE2_PATH = "F:\\dataset\\xrf55_for_siamid\\scene2_filtered"
SCENE3_PATH = "F:\\dataset\\xrf55_for_siamid\\scene3_filtered"
SCENE4_PATH = "F:\\dataset\\xrf55_for_siamid\\scene4_filtered"
CA = 0
CP = 1
CACP = 2
CPCS = 3
TRAIN = 0
TEST = 1

def check_dataset(path, dataset, part):
    """
    Check if the samples in the dataset are from the right persons and actions, and if they include all the trails.
    """
    print(f"Checking dataset {path}, {dataset}, {part}...")

    global CA, CP, CACP, CPCS, TRAIN, TEST

    if part == TRAIN:
        path = os.path.join(path, 'train')
    else:
        path = os.path.join(path, 'test')

    expected_persons = [[set() for _ in range(2)] for _ in range(4)]
    expected_persons[CA][TRAIN] = set(f"{i:02}" for i in range(1, 31))
    expected_persons[CA][TEST] = expected_persons[CA][TRAIN]
    expected_persons[CP][TRAIN] = set(f"{i:02}" for i in range(1, 21))
    expected_persons[CP][TEST] = set(f"{i:02}" for i in range(21, 31))
    expected_persons[CACP][TRAIN] = expected_persons[CP][TRAIN]
    expected_persons[CACP][TEST] = expected_persons[CP][TEST]
    expected_persons[CPCS][TRAIN] = set(f"{i:02}" for i in range(1, 31)) - set(f"{i:02}" for i in range(3, 8)) - {"13", "23", "24", "31"}
    expected_persons[CPCS][TEST] = set(f"{i:02}" for i in range(3, 8)) | {"13", "23", "24", "31"}

    expected_actions = [[set() for _ in range(2)] for _ in range(4)]
    expected_actions[CA][TRAIN] = (set(f"{i:02}" for i in range(1, 12)) | set(f"{i:02}" for i in range(23, 27)) |
                                   set(f"{i:02}" for i in range(31, 41)) | set(f"{i:02}" for i in range(45, 52)))
    expected_actions[CA][TEST] = (set(f"{i:02}" for i in range(12, 16)) | set(f"{i:02}" for i in range(27, 31)) |
                                   set(f"{i:02}" for i in range(41, 45)) | set(f"{i:02}" for i in range(52, 56)))
    expected_actions[CP][TRAIN] = set(f"{i:02}" for i in range(1, 16)) | set(f"{i:02}" for i in range(23, 56))
    expected_actions[CP][TEST] = expected_actions[CP][TRAIN]
    expected_actions[CACP][TRAIN] = expected_actions[CA][TRAIN]
    expected_actions[CACP][TEST] = expected_actions[CA][TEST]
    expected_actions[CPCS][TRAIN] = expected_actions[CP][TRAIN]
    expected_actions[CPCS][TEST] = expected_actions[CP][TEST]

    expected_sample_number = [[0 for _ in range(2)] for _ in range(4)]
    expected_sample_number[CA][TRAIN] = 19200
    expected_sample_number[CA][TEST] = 9600
    expected_sample_number[CP][TRAIN] = expected_sample_number[CA][TRAIN]
    expected_sample_number[CP][TEST] = expected_sample_number[CA][TEST]
    expected_sample_number[CACP][TRAIN] = 12800
    expected_sample_number[CACP][TEST] = 3200
    expected_sample_number[CPCS][TRAIN] = 21120
    expected_sample_number[CPCS][TEST] = 8640

    expected_trails = set(f"{i:02}" for i in range(1, 21))

    actual_persons = set()
    actual_actions = [set() for _ in range(32)]
    actual_trails = [[set() for _ in range(56)] for _ in range(32)]
    if len(os.listdir(path)) != expected_sample_number[dataset][part]:
        print(f"Error: the sample number doesn't math the expected number in dataset {dataset}-{part}.")

    for file in os.listdir(path):
        # print("Processing: ", file)
        person, action, trial = file.split('_')[0], file.split('_')[1], file.split('_')[2][:-4]
        actual_persons.add(person)
        actual_actions[int(person)].add(action)
        actual_trails[int(person)][int(action)].add(trial)

    missing_persons = expected_persons[dataset][part] - actual_persons
    extra_persons = actual_persons - expected_persons[dataset][part]
    if missing_persons or extra_persons:
        print(f"Error: missing or extra person in dataset {dataset}-{part}.")

    for person in actual_persons:
        missing_actions = expected_actions[dataset][part] - actual_actions[int(person)]
        extra_actions = actual_actions[int(person)] - expected_actions[dataset][part]
        if missing_actions or extra_actions:
            print(f"Error: missing or extra action in dataset {dataset}-{part}.")

        for action in actual_actions[int(person)]:
            missing_trails = expected_trails - actual_trails[int(person)][int(action)]
            extra_trails = actual_trails[int(person)][int(action)] - expected_trails
            if missing_trails or extra_trails:
                print(f"Error: missing or extra trails in dataset {dataset}-{part}.")


def check_scene(person, dataset_path, part, scene_path):
     """
     Check if the samples of person in dataset-part are from the right scene.
     """
     print(f"Checking scene {person}, {dataset_path}, {part}, {scene_path}...")

     if part == TRAIN:
         dataset_path = os.path.join(dataset_path, 'train')
     else:
         dataset_path = os.path.join(dataset_path, 'test')

     files = set([file for file in os.listdir(dataset_path) if file.split('_')[0] == person])
     files_in_scene = set([file for file in os.listdir(scene_path) if file.split('_')[0] == person])
     files, files_in_scene = set(files) & set(files_in_scene), set(files_in_scene) & set(files)
     # Just compare between the two with the same file name

     for file, file_in_scene in zip(files, files_in_scene):
         if not filecmp.cmp(os.path.join(dataset_path, file), os.path.join(scene_path, file_in_scene), shallow=False):
             # They have the same filename but the different content
             print(file, file_in_scene, dataset_path, scene_path)
             print(f"Error: person {person} from the wrong scene.")
         # else:
         #    break


check_dataset(CA_PATH, CA, TRAIN)
check_dataset(CA_PATH, CA, TEST)
check_dataset(CP_PATH, CP, TRAIN)
check_dataset(CP_PATH, CP, TEST)
check_dataset(CACP_PATH, CACP, TRAIN)
check_dataset(CACP_PATH, CACP, TEST)
check_dataset(CPCS_PATH, CPCS, TRAIN)
check_dataset(CPCS_PATH, CPCS, TEST)

multi_scene_persons = ["03", "04", "05", "06", "07", "13", "23", "24", "31"]
scene2_persons = ["05", "24", "31"]
scene3_persons = ["06", "07", "23"]
scene4_persons = ["03", "04", "13"]
for person in multi_scene_persons:
    check_scene(person, CA_PATH, TRAIN, SCENE1_PATH)
    check_scene(person, CA_PATH, TEST, SCENE1_PATH)
    check_scene(person, CP_PATH, TRAIN, SCENE1_PATH)
    check_scene(person, CP_PATH, TEST, SCENE1_PATH)
    check_scene(person, CACP_PATH, TRAIN, SCENE1_PATH)
    check_scene(person, CACP_PATH, TEST, SCENE1_PATH)
    if person in scene2_persons:
        check_scene(person, CPCS_PATH, TRAIN, SCENE1_PATH)
        check_scene(person, CPCS_PATH, TEST, SCENE2_PATH)
    if person in scene3_persons:
        check_scene(person, CPCS_PATH, TRAIN, SCENE1_PATH)
        check_scene(person, CPCS_PATH, TEST, SCENE3_PATH)
    if person in scene4_persons:
        check_scene(person, CPCS_PATH, TRAIN, SCENE1_PATH)
        check_scene(person, CPCS_PATH, TEST, SCENE4_PATH)

print('Check completed.')