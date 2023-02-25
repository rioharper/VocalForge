from pydub import AudioSegment
import os

def split_files(folder: str, dir: str, duration: int):
    '''This function splits audio files in the .wav format located in the
       specified folder and saves the clips in the same folder. 

        Inputs:
        - folder: a string representing the name of the folder containing the audio files.
        - dir: a string representing the directory path containing the folder.
        - duration: ms of the duration of sample clips.
        
        Outputs:
        - None, but audio clips are saved to disk in the .wav format.
        
        Dependencies:
        - "os" library
        - "pydub" library and its component "AudioSegment"'''
        
    folder_dir = os.path.join(dir, folder)
    for file in get_files(folder_dir):
        file_dir = os.path.join(dir, folder_dir, file)
        print(file_dir)
        raw = AudioSegment.from_file(file_dir, format="wav")
        raw = raw[::duration]
        for index, clip in raw:
            clip_dir = os.path.join(folder_dir, file.split(".")[0], file.split(".")[0]+str(index)+".wav")
            clip = clip.export(clip_dir, format="wav")

def get_files(dir:str, ext=None) -> list:
    '''This function returns a list of files in a specified directory with a specified file extension. 

    Inputs:
    - dir: a string representing the directory path containing the files.
    - ext (optional): a string representing the file extension to filter the files. 
      If not specified, all files will be returned.
    
    Outputs:
    - A list of sorted filenames.
    
    Dependencies:
    - "os" library
    - "natsort" library'''

    import natsort
    files = []
    for file in os.listdir(dir):
        if ext!=None:
            if file.endswith(ext):
                files.append(file)
        else: files.append(file)
    files = natsort.natsorted(files)
    return files


def create_core_folders(folders: list, workdir: str):
    for folder in folders:
        folderdir = os.path.join(workdir, folder)
        if os.path.exists(folderdir) == False:
            os.makedirs(folderdir)