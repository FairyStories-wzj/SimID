"""
CSI Data Batch Filtering Script
--------------------------------
This script applies a Butterworth low-pass filter to all .npy files (CSI data) in a given folder (and its subfolders),
and saves the filtered results to a new folder with the suffix '_filtered'.

Usage:
    1. Set the input_folder variable (at the bottom) to your own data root path.
    2. Optionally, set the sub_folders list to the subdirectories you want to process.
    3. Run this script directly: python filter_data.py
    4. The filtered files will be saved in a new folder named <original_folder>_filtered for each subfolder.

Note:
    - Please replace the example path (e.g., r"F:/dataset/xrf55_for_siamid") with your actual data path.
    - The script uses multi-threading for acceleration.
    - Only .npy files will be processed.
"""

import numpy as np
from scipy import signal
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing


def filter_csi_file(file_path, output_folder):
    """
    Apply a Butterworth filter to a single CSI file and save the result to a specified output folder.

    Parameters:
    file_path (str): The path to the CSI data file.
    output_folder (str): The folder where the filtered file will be saved.

    Returns:
    tuple: (file_name, processing_status, processing_time, error_message (if failed))
    """
    try:
        start_time = time.time()

        base_name = os.path.basename(file_path)
        file_name = os.path.splitext(base_name)[0]

        output_file_path = os.path.join(output_folder, base_name)

        ori_csi = np.load(file_path)

        flt_csi = np.zeros_like(ori_csi)

        # Butterworth low-pass filter, order=2, cutoff=2Hz (sampling rate assumed 100Hz)
        b, a = signal.butter(2, 2 / 100, btype='low')

        for i in range(ori_csi.shape[0]):
            flt_csi[i, :] = signal.filtfilt(b, a, ori_csi[i, :], padlen=3 * max(len(b), len(a)) - 1)

        np.save(output_file_path, flt_csi)

        end_time = time.time()
        processing_time = end_time - start_time

        return file_name, "success", processing_time, None

    except Exception as e:
        end_time = time.time()
        processing_time = end_time - start_time
        return os.path.basename(file_path), "failed", processing_time, str(e)


def get_all_npy_files(folder_path):
    """
    Recursively obtain all .npy files within a folder and its subfolders.

    Parameters:
    folder_path (str): The path to the root folder to be searched.

    Returns:
    list: A list of the full paths of all found .npy files.
    """
    npy_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.npy'):
                npy_files.append(os.path.join(root, file))
    return npy_files


def process_folder(input_folder, output_folder, num_threads=None):
    """
    Process all NPY files in a folder and its subfolders using multithreading, and save the results to a specified output folder.

    Parameters:
    input_folder (str): The path to the root folder containing the CSI data files.
    output_folder (str): The path to the folder where the filtered files will be saved.
    num_threads (int, optional): The number of threads to use. Defaults to the number of CPU cores.
    """
    
    npy_files = get_all_npy_files(input_folder)

    if not npy_files:
        print(f"Warning: No npy files were found in {input_folder} and its subfolders.")
        return

    print(f"{len(npy_files)} NPY files are to be processed.")

    if num_threads is None:
        try:
            num_threads = multiprocessing.cpu_count()
        except Exception:
            num_threads = 4  # Default number of threads

    print(f"Number of threads: {num_threads}")

    total_start_time = time.time()
    results = []

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        
        future_to_file = {executor.submit(filter_csi_file, file, output_folder): file for file in npy_files}

        for future in as_completed(future_to_file):
            file = future_to_file[future]
            try:
                file_name, status, proc_time, error = future.result()
                results.append((file_name, status, proc_time, error))
                print(f"File {file_name} finished, state: {status}, time consumed: {proc_time:.2f} seconds.")
            except Exception as exc:
                file_name = os.path.basename(file)
                results.append((file_name, "error", 0, str(exc)))
                print(f"File {file_name} errored: {str(exc)}")

    total_time = time.time() - total_start_time
    success_count = sum(1 for _, status, _, _ in results if status == "success")
    failure_count = len(results) - success_count

    print("\nSummary of the processing:")
    print(f"Total files: {len(npy_files)}")
    print(f"Succeeded: {success_count}")
    print(f"Failed: {failure_count}")
    print(f"Total time: {total_time:.2f} sec")
    print(f"Average per file: {total_time / len(npy_files):.2f} sec")

    if failure_count > 0:
        print("\nFailed files:")
        for file_name, status, proc_time, error in results:
            if status != "success":
                print(f"- {file_name}: {status} ({'message: ' + error if error else ''})")

    return results


def run_filter(input_folder, num_threads=None):
    """
    Run the CSI data processing flow and save the filtered files to a new folder.

    Parameters:
    input_folder (str): Path to the root folder containing NPY files.
    num_threads (int, optional): The number of threads to use, defaults to the number of CPU cores.
    """

    output_folder = input_folder + "_filtered"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    print(f"Start processing: {input_folder} ...")
    results = process_folder(input_folder, output_folder, num_threads)
    print("\nFinished.")


if __name__ == "__main__":
    # Please set your own data root path here. Example:
    # input_folder = r"F:/dataset/xrf55_for_siamid"
    input_folder = r"<your_data_root_path>"  # <-- Change this to your own data path
    sub_folders = ["scene1", "scene2", "scene3", "scene4"]  # <-- Change or extend as needed

    for sub_folder in sub_folders:
        full_input_folder = os.path.join(input_folder, sub_folder)
        run_filter(full_input_folder)