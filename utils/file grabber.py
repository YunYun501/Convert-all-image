
import os
import pandas as pd 
import json 







"""

The folder should contain the following functions to call: 

Input single file directory 

Output List of sub folders and files in that directory 

Should store the output as a dataframe 





"""



def get_directory_heierarchy(directory_path : str) -> dict:

    folder_heirearchy = {}

    for directory_root, directory_names, filenames in os.walk(directory_path): 
        
        sub_folder_list = [] 
        file_list =[] 

        for dirnames in directory_names: 
            sub_folder_list.append(dirnames) 
            #folder_heirearchy[dirnames] = os.path.join(directory_root, dirnames)
        for file in filenames:
            file_list.append(file)
            #folder_heirearchy[file] = os.path.join(directory_root, file)

        folder_heirearchy[directory_root] = {"sub_folders" : sub_folder_list, "files" : file_list}
    
    print(folder_heirearchy)
    
    with open("folder_heirearchy.json", "w") as f:
        json.dump(folder_heirearchy, f, indent=4)
            
    return folder_heirearchy



if __name__ == "__main__":
    directory_path = "C:\Local"
    get_directory_heierarchy(directory_path)