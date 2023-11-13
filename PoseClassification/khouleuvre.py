from matplotlib import pyplot as plt
import numpy as np
import os

def sort_images (root_dir, images_dir) :
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
            sort_images(full_path)
        # assign the file to "docs" folder ./docs
        elif full_path.endswith(".txt"):
            with open(full_path, 'r') as f:
                data = f.read()
                values= data.split(' ')
                classe = values[0]
                image_path = ".".join(entry.split('.')[:-1])+".jpg"
                source=os.path.join(root,image_path)
                if classe == '0':
                    destination= os.path.join(root_dir,"docs/images/pushups_down", image_path)
                    os.rename(source , destination)
                elif classe == '1':
                    destination= os.path.join(root_dir,"docs/images/pushups_up", image_path)
                    os.rename(source, destination)
                else:
                    pass
            #os.rename(full_path, os.path.join("docs/images", entry))
            print(f"Moved {full_path} to ./docs")
        else:
            pass


            
#   else:
#         if full_path.endswith(".jpg"):
#             print(full_path)
#             # print(os.path.basename(full_path))
#             # print(os.path.dirname(full_path))
#             # print(os.path.split(full_path))
#             # print(os.path.splitext(full_path))
#             # print(os.path.join(os.path.dirname(full_path), "pushups_up", os.path.basename(full_path)))
#             # print(os.path.join(os.path.dirname(full_path), "pushups_down", os.path.basename(full_path)))
#             if full_path.startswith(1):
#                 os.rename(full_path, os.path.join(os.path.dirname(full_path), "pushups_up", os.path.basename(full_path)))
#             elif full_path.startswith(0):
#                 os.rename(full_path, os.path.join(os.path.dirname(full_path), "pushups_down", os.path.basename(full_path)))
#             else:
#                 pass
#         else:
#             pass

sort_images('/home/khaleb.dabakuyo@Digital-Grenoble.local/Documents/ACV/Panther_trainer','docs/pushup_dataset/obj')
          