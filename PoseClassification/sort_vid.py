from matplotlib import pyplot as plt
import numpy as np
import os

def sort_video (root_dir, images_dir) :
    # Get the list of all files and directories
    # in the root directory
    root = os.path.join(root_dir, images_dir)
    file_list = os.listdir(root)
    # Iterate over all the entries
    for entry in file_list:
        # Create full path
        full_path = os.path.join(images_dir, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(full_path):
            sort_video(full_path)
        # assign the file to "docs" folder ./docs
        elif full_path.endswith(".mp4"):
            destination= os.path.join(root_dir,"docs/videos", entry)
            os.rename(full_path, destination)
        else:
            pass
            #os.rename(full_path, os.path.join("docs/images", entry))
            print(f"Moved {full_path} to ./docs")





sort_video('/home/khaleb.dabakuyo@Digital-Grenoble.local/Documents/ACV/Panther_trainer','docs/video_dataset/Correct sequence')