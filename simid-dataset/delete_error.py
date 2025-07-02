import os

def delete_temp_filtered_files(data_root):
    """
    delete files that ends with .temp_filtered.npy

    params:
    data_root (str): the path of the root
    """

    for root, dirs, files in os.walk(data_root):
        for file in files:
            # check if the file is end with .temp_filtered.npy
            if file.endswith('.temp_filtered.npy'):
                file_path = os.path.join(root, file)
                # delete
                os.remove(file_path)
                print(f"deleted: {file_path}")

DATA_ROOT = "F:\\dataset\\xrf55_for_simid"

delete_temp_filtered_files(DATA_ROOT)