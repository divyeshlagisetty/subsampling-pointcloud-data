

import os
import sys
from tqdm import tqdm as tqdm
import numpy as np
import random as rand

sys.path.append("data3/ai/div/sampling-scripts")
#import dialogs as dia
import point_cloud_utils as pcu
import argparse
from multiprocessing import Pool
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Script to perform random sampling to a set percentage of loss level.")
parser.add_argument("--input_directory", type=str, help="The directory where the uncompressed dataset is stored.")
parser.add_argument("--output_directory", type=str, help="The directory where the compressed dataset is to be stored.")
parser.add_argument("--loss_level", type=int, help="Percentage loss level required. (eg : 65, 75, 85, 95). Actual loss will be -2.5 percentage of the value entered.")

args = parser.parse_args()
input_dir = args.input_directory
out_dir = args.output_directory
compression_level = args.loss_level
# Select the input directory : where the uncompressed clouds are stored

"""
print("Select the INPUT DIRectory...")
in_dir = dia.selectFolder()
files_list = pcu.listFiles(in_dir)
print("Select the OUTPUT DIRectory...")
out_dir = dia.selectFolder()

compression_level = int(input("Please enter the compression % (eg : 65, 75, 85, 95)"))
"""

# Perform the compression of every file and store in the output directory
def process_file(file):
    global input_dir, out_dir

    file_name = file.split("/")[-1]
    out_path_ply = out_dir + "/" + file_name
    out_path_csv = ".csv".join(out_path_ply.split(".ply"))

    #os.system('flatpak run org.cloudcompare.CloudCompare -SILENT -O '+file+' -AUTO_SAVE OFF -C_EXPORT_FMT PLY -SS RANDOM  -SAVE_CLOUDS FILE "'+out_path)
    file_format = file_name.split(".")[-1]
    if file_format == "ply":
        in_cloud = pcu.loadCloudFromPLY(os.path.join(input_dir,file), ["x", "y", "z", "scalar_Scalar_field", "scalar_Scalar_field_#2"]) # for dales
    elif file_format == "csv":
        in_cloud = np.loadtxt(os.path.join(input_dir, file), delimiter=",")
    in_size = len(in_cloud)
    
    # Select random indices
    #print("Generating random indices...\n\n")
    rand_indices = rand.sample(list(range(in_size)), int(((100-compression_level+2.5)/100)*in_size))

    # Append the points corresponding to the random indices to the output cloud array
    #print("Generating output cloud...\n\n")
    out_cloud = []
    for index in rand_indices:    
        out_cloud.append(list(in_cloud[index]))

    # Save the output cloud array as csv
    #print("Saving output cloud ...\n\n")
    out_cloud = np.array(out_cloud)
    pcu.saveCloud2CSV(out_cloud, out_path_csv)

    # Conver the output CSV array to PLY using Cloud Compare and save, also delete the CSV file
    #os.system('flatpak run org.cloudcompare.CloudCompare -SILENT -O '+out_path_csv+' -AUTO_SAVE OFF -C_EXPORT_FMT PLY -SAVE_CLOUDS FILE "'+out_path_ply+'" && rm ' + out_path_csv)

if __name__ == "__main__":
    files_list = os.listdir(input_dir)
    print("Processing " + str(len(files_list)) + " files...")
    
   
    with Pool() as p:
            r = list(tqdm(p.imap(process_file, files_list), total=len(files_list), colour="red"))