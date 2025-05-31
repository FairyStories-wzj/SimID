import os
import shutil

# split the dataset

# the sets
# all users
U = set([f"{i:02d}" for i in range(1, 32)])
# all actions
A = set([f"{i:02d}" for i in range(1, 56)]) - set([f"{i:02d}" for i in range(16, 23)])

# the path to the data
DATA_ROOT = "F:\\dataset\\xrf55_for_siamid"
scene1_filtered_path = os.path.join(DATA_ROOT, "scene1_filtered")
scene2_filtered_path = os.path.join(DATA_ROOT, "scene2_filtered")
scene3_filtered_path = os.path.join(DATA_ROOT, "scene3_filtered")
scene4_filtered_path = os.path.join(DATA_ROOT, "scene4_filtered")


# the function to process the data
def copy_files_based_on_conditions(input_paths, output_train, output_test, user_train_set, user_test_set,
                                   action_train_set, action_test_set):
    """
    Copies files from folders in the input path list to the output path for the training set or the test set, depending on the conditions of the user and action set.

    params:
    input_paths (list): Input folder path list
    output_train (str): Training set output folder path
    output_test (str): Test suite output folder path
    user_train_set (set): The set of users that match the training set
    user_test_set (set): The set of users that match the test set
    action_train_set (set): Action set that matches the training set
    action_test_set (set): Action set that matches the test set
    """
    # Make sure the output folder exists
    os.makedirs(output_train, exist_ok=True)
    os.makedirs(output_test, exist_ok=True)

    # Iterate through the input folder list
    for input_path in input_paths:
        # Iterate over all files in the input folder
        for file in os.listdir(input_path):
            if file.endswith(".npy"):
                user, action, _ = file.split("_")
                if user in user_train_set and action in action_train_set:
                    # Copy to the training set folder
                    shutil.copy(os.path.join(input_path, file), output_train)
                    print(f"Copying {input_path}, {file} to {output_train}")
                elif user in user_test_set and action in action_test_set:
                    # Copy to the test set folder
                    shutil.copy(os.path.join(input_path, file), output_test)
                    print(f"Copying {input_path}, {file} to {output_test}")


# Functions that handle CPCS separately
def copy_files_for_cpcs(train_input_paths, test_input_paths, train_output, test_output, user_train_set, user_test_set, action_train_set, action_test_set):
    """
    File partitioning function specifically for CPCS dataset.

    params:
    train_input_paths (list): Training set input folder path list
    test_input_paths (list): Test set input folder path list
    train_output (str): Training set output folder path
    test_output (str): Test suite output folder path
    user_train_set (set): The set of users that match the training set
    user_test_set (set): The set of users that match the test set
    action_train_set (set): Action set that matches the training set
    action_test_set (set): Action set that matches the test set
    """
    # Make sure the output folder exists
    os.makedirs(train_output, exist_ok=True)
    os.makedirs(test_output, exist_ok=True)

    # Processing training set data
    for input_path in train_input_paths:
        for file in os.listdir(input_path):
            if file.endswith(".npy"):
                user, action, _ = file.split("_")
                if user in user_train_set and action in action_train_set:
                    shutil.copy(os.path.join(input_path, file), train_output)
                    print(f"Copying {input_path}, {file} to {train_output}")

    # Processing test set data
    for input_path in test_input_paths:
        for file in os.listdir(input_path):
            if file.endswith(".npy"):
                user, action, _ = file.split("_")
                if user in user_test_set and action in action_test_set:
                    shutil.copy(os.path.join(input_path, file), test_output)
                    print(f"Copying {input_path}, {file} to {test_output}")

# Start dividing the data
# CPCS
USER_CPCS_TRAIN = U - {"03", "04", "05", "06", "07", "13", "23", "24", "31"}
USER_CPCS_TEST = {"03", "04", "05", "06", "07", "13", "23", "24", "31"}
ACTION_CPCS_TRAIN = A
ACTION_CPCS_TEST = A

CPCS_TRAIN = os.path.join(DATA_ROOT, "CPCS", "train")
CPCS_TEST = os.path.join(DATA_ROOT, "CPCS", "test")

# Create a destination folder
os.makedirs(CPCS_TRAIN, exist_ok=True)
os.makedirs(CPCS_TEST, exist_ok=True)

# Processing CPCS dataset
copy_files_for_cpcs(
    [scene1_filtered_path],
    [scene2_filtered_path, scene3_filtered_path, scene4_filtered_path],
    CPCS_TRAIN,
    CPCS_TEST,
    USER_CPCS_TRAIN, USER_CPCS_TEST, ACTION_CPCS_TRAIN, ACTION_CPCS_TEST
)

print("CPCS finished")

# CA
USER_CA = U - {"31"}
ACTION_CA_TRAIN = A - set([f"{i:02d}" for i in range(12, 16)] + [f"{i:02d}" for i in range(27, 31)] +
                          [f"{i:02d}" for i in range(41, 45)] + [f"{i:02d}" for i in range(52, 56)])
ACTION_CA_TEST = set([f"{i:02d}" for i in range(12, 16)] + [f"{i:02d}" for i in range(27, 31)] +
                     [f"{i:02d}" for i in range(41, 45)] + [f"{i:02d}" for i in range(52, 56)])


# destination
CA_TRAIN = os.path.join(DATA_ROOT, "CA", "train")
CA_TEST = os.path.join(DATA_ROOT, "CA", "test")

# Create a destination folder
os.makedirs(CA_TRAIN, exist_ok=True)
os.makedirs(CA_TEST, exist_ok=True)

# Processing CA dataset
copy_files_based_on_conditions([scene1_filtered_path], CA_TRAIN, CA_TEST, USER_CA, USER_CA, ACTION_CA_TRAIN,
                               ACTION_CA_TEST)
print("CA finished")

# CP
USER_CP_TEST = set([f"{i:02d}" for i in range(21, 31)])
USER_CP_TRAIN = U - set([f"{i:02d}" for i in range(21, 31)] + ["31"])
ACTION_CP = A

CP_TRAIN = os.path.join(DATA_ROOT, "CP", "train")
CP_TEST = os.path.join(DATA_ROOT, "CP", "test")

os.makedirs(CP_TRAIN, exist_ok=True)
os.makedirs(CP_TRAIN, exist_ok=True)

# Processing CP dataset
copy_files_based_on_conditions([scene1_filtered_path], CP_TRAIN, CP_TEST, USER_CP_TRAIN, USER_CP_TEST, ACTION_CP,
                               ACTION_CP)

print("CP finished")

# CACP
USER_CACP_TRAIN = U - set([f"{i:02d}" for i in range(21, 31)] + ["31"])
USER_CACP_TEST = set([f"{i:02d}" for i in range(21, 31)])
ACTION_CACP_TEST = set([f"{i:02d}" for i in range(12, 16)] + [f"{i:02d}" for i in range(27, 31)] +
                           [f"{i:02d}" for i in range(41, 45)] + [f"{i:02d}" for i in range(52, 56)])
ACTION_CACP_TRAIN = A - ACTION_CACP_TEST

CACP_TRAIN = os.path.join(DATA_ROOT, "CACP", "train")
CACP_TEST = os.path.join(DATA_ROOT, "CACP", "test")

os.makedirs(CACP_TRAIN, exist_ok=True)
os.makedirs(CACP_TEST, exist_ok=True)

# Processing CACP dataset
copy_files_based_on_conditions([scene1_filtered_path], CACP_TRAIN, CACP_TEST,
                               USER_CACP_TRAIN, USER_CACP_TEST, ACTION_CACP_TRAIN, ACTION_CACP_TEST)

print("CACP finished")
