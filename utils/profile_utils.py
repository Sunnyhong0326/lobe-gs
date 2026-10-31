import csv
import os
from utils.general_utils import format_seconds

def write_optimization_profile(dataset, max_memory_allocated, elapsed_time):
    csv_file_path = os.path.join(dataset.profile_path, "optimization_time.csv")
    file_exists = os.path.isfile(csv_file_path)

    with open(csv_file_path, "a", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        if not file_exists:
            csv_writer.writerow(["Block_id", "Max Memory Allocated (GB)", "Elapsed Time (Seconds)", "Elapsed Time (HH:MM:SS)"])
        elapsed_time_seconds = elapsed_time 
        elapsed_time_formatted = format_seconds(elapsed_time)
        csv_writer.writerow([dataset.block_id, max_memory_allocated, elapsed_time_seconds, elapsed_time_formatted])

def write_per_block_splat_count_profile(dataset, iteration, total_splat_count, block_splat_count=None):
    csv_file_path = os.path.join(dataset.profile_path, "splat_count.csv")
    file_exists = os.path.isfile(csv_file_path)

    with open(csv_file_path, "a", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        if not file_exists:
            csv_writer.writerow(["Block_id", "Iteration", "Total Splat Count", "Block Splat Count"])
        csv_writer.writerow([dataset.block_id, iteration, total_splat_count, block_splat_count])

def write_splat_count_profile(dataset, iteration, total_splat_count):
    csv_file_path = os.path.join(dataset.profile_path, "splat_count.csv")
    file_exists = os.path.isfile(csv_file_path)

    with open(csv_file_path, "a", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        if not file_exists:
            csv_writer.writerow(["Iteration", "Total Splat Count"])
        csv_writer.writerow([iteration, total_splat_count])

def write_ply_save_time_profile(dataset, iteration, elapsed_time):
    csv_file_path = os.path.join(dataset.profile_path, "ply_save_time.csv")
    file_exists = os.path.isfile(csv_file_path)
    ply_path = os.path.join(
        dataset.model_path,
        "point_cloud",
        f"iteration_{iteration}",
        "point_cloud.ply",
    )

    with open(csv_file_path, "a", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        if not file_exists:
            csv_writer.writerow(["Block_id", "Iteration", "Elapsed Time (Seconds)", "Elapsed Time (HH:MM:SS)", "PLY Path"])
        csv_writer.writerow([dataset.block_id, iteration, elapsed_time, format_seconds(elapsed_time), ply_path])
