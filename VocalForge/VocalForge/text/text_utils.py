from pydub import AudioSegment
import os
import natsort
import pandas as pd

def split_files(folder: str, dir: str, duration: int):
    '''This function splits audio files in the .wav format located in the
       specified folder and saves the clips in the same folder. 

        Inputs:
        - folder: a string representing the name of the folder containing the audio files.
        - dir: a string representing the directory path containing the folder.
        - duration: ms of the duration of sample clips.
        
        Outputs:
        - None, but audio clips are saved to disk in the .wav format.
        
    '''

    folder_dir = os.path.join(dir, folder)
    for file in get_files(folder_dir, ext=".wav"):
        file_dir = os.path.join(folder_dir, file)
        print(file_dir)
        raw = AudioSegment.from_file(file_dir, format="wav")
        for index, clip in enumerate(raw[::duration]):
            clip_dir = os.path.join(folder_dir, file.split(".")[0], f"{file.split('.')[0]}_{index}.wav")
            clip.export(clip_dir, format="wav")

def get_files(dir: str, full_dir: bool = False, ext: str = None) -> list:
    """
    Retrieves a list of files in a directory, sorted in natural order.

    Parameters:
        dir (str): A string representing the directory path to search for files.
        full_dir (bool): A boolean indicating whether to return the full directory path or just the file name. Default is False.
        ext (str): A string representing the file extension to filter by. If None, all files are returned. Default is None.

    Returns:
        files (list): A list of file names sorted in natural order.

    Example:
        get_files('/home/user/documents', True, '.txt')
        Returns:
        ['/home/user/documents/file1.txt', '/home/user/documents/file2.txt']
    """
    files = []
    for file in os.listdir(dir):
        if ext is not None:
            if file.endswith(ext):
                if full_dir:
                    files.append(os.path.join(dir, file))
                else:
                    files.append(file)
        else:
            files.append(file)
    files = natsort.natsorted(files)
    return files


def create_core_folders(folders: list, workdir: str):
    """
    This function creates a list of folders in a specified directory if they do not already exist.

    Parameters:
        folders (list): A list of folder names to be created.
        workdir (str): A string representing the directory path where the folders will be created.

    Returns:
        None

    Example:
        create_core_folders(['raw', 'processed'], '/home/user/documents')
        Creates the folders 'raw' and 'processed' in the directory '/home/user/documents'.
    """
    for folder in folders:
        folderdir = os.path.join(workdir, folder)
        if not os.path.exists(folderdir):
            os.makedirs(folderdir)


def remove_small_files(dataset_dir: str):
    """
    This function removes audio files that are below 100kb in size from the specified directory and removes their corresponding entries in the metadata.csv file.

    Parameters:
        dataset_dir (str): A string representing the directory path containing the audio files and metadata.csv file. Must be in the LJSpeech format.

    Returns:
        None
    """
    bad_files = []
    for file in os.listdir(dataset_dir+'/wavs/'):
        #if file is below 100kb remove it and append it to the list
        if os.stat(dataset_dir +'/wavs/'+file).st_size < 100000:
            bad_files.append(file)

    df = pd.read_csv(dataset_dir+'/metadata.csv', sep='|', on_bad_lines='skip')
    for index, row in df.iterrows():
        if row[0]+'.wav' in bad_files:
            df.drop(index, inplace=True)
            os.remove(dataset_dir + '/wavs/' + row[0]+'.wav')   
            df.to_csv(dataset_dir+'/metadata.csv', sep='|', index=False)


def remove_extra_audio_files(dataset_dir: str):
    """
    This function removes audio files that are not listed in the metadata.csv file from the specified directory and removes their corresponding entries in the metadata.csv file.

    Parameters:
        dataset_dir (str): A string representing the directory path containing the audio files and metadata.csv file. Must be in the LJSpeech format.

    Returns:
        None
    """
    df = pd.read_csv(dataset_dir+'/metadata.csv', sep='|', on_bad_lines='skip')
    wav_files = os.listdir(dataset_dir + '/wavs/')
    good_files = []
    for index, row in df.iterrows():
        good_files.append(row[0]+'.wav')

    for file in wav_files:
        if file not in good_files:
            os.remove(dataset_dir + '/wavs/' + file)
            print(file + ' removed')


def remove_extra_metadata(dataset_dir: str):
    """
    This function removes entries in the metadata.csv file that do not have corresponding audio files in the specified directory.
    
    Parameters:
        dataset_dir (str): A string representing the directory path containing the audio files and metadata.csv file. Must be in the LJSpeech format.
    """
    wav_files = os.listdir(dataset_dir + '/wavs/')
    df = pd.read_csv(dataset_dir+'/metadata.csv', sep='|', on_bad_lines='skip')

    for index, row in df.iterrows():
        if row[0]+'.wav' not in wav_files:
            df.drop(index, inplace=True)
            print(row[0])
    df.to_csv(dataset_dir+'/metadata.csv', sep='|', index=False)