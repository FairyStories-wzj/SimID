"""
This program is aiming to check if the original XRF55 dataset is intact and correct
"""
import os

SCENE1_PATH = "F:\\dataset\\xrf55_for_simid\\scene1_filtered"
SCENE2_PATH = "F:\\dataset\\xrf55_for_simid\\scene2_filtered"
SCENE3_PATH = "F:\\dataset\\xrf55_for_simid\\scene3_filtered"
SCENE4_PATH = "F:\\dataset\\xrf55_for_simid\\scene4_filtered"


def check_intact(path, scene):
    expected_actions = set(f"{i:02}" for i in range(1, 56))
    expected_trails = set(f"{i:02}" for i in range(1, 21))
    if scene == 1:
        expected_persons = set(f"{i:02}" for i in range(1, 31))
    elif scene == 2:
        expected_persons = {'05', '24', '31'}
    elif scene == 3:
        expected_persons = {'06', '07', '23'}
    elif scene == 4:
        expected_persons = {'03', '04', '13'}

    actual_persons = set()
    actual_actions = [set() for _ in range(32)]
    actual_trails = [[set() for _ in range(56)] for _ in range(32)]
    for file in os.listdir(path):
        # print("Processing: ", file)
        person, action, trial = file.split('_')[0], file.split('_')[1], file.split('_')[2][:-4]
        actual_persons.add(person)
        actual_actions[int(person)].add(action)
        actual_trails[int(person)][int(action)].add(trial)

    missing_persons = expected_persons - actual_persons
    extra_persons = actual_persons - expected_persons
    if missing_persons or extra_persons:
        print(f"Error: missing or extra person in scene {scene}.")

    for person in actual_persons:
        missing_actions = expected_actions - actual_actions[int(person)]
        extra_actions = actual_actions[int(person)] - expected_actions
        if missing_actions or extra_actions:
            print(f"Error: missing or extra action in person {person}, scene {scene}.")

        for action in actual_actions[int(person)]:
            missing_trails = expected_trails - actual_trails[int(person)][int(action)]
            extra_trails = actual_trails[int(person)][int(action)] - expected_trails
            if missing_trails or extra_trails:
                print(f"Error: missing or extra trails in action {action}, person {person}, scene {scene}.")

check_intact(SCENE1_PATH, 1)
check_intact(SCENE2_PATH, 2)
check_intact(SCENE3_PATH, 3)
check_intact(SCENE4_PATH, 4)

print("Check completed.")