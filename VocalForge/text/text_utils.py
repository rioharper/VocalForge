from pydub import AudioSegment
from pathlib import Path
import natsort
import pandas as pd


def split_files(folder: str, dir: str, duration: int):
    """This function splits audio files in the .wav format located in the
    specified folder and saves the clips in the same folder.

     Inputs:
     - folder: a string representing the name of the folder containing the audio files.
     - dir: a string representing the directory path containing the folder.
     - duration: ms of the duration of sample clips.

     Outputs:
     - None, but audio clips are saved to disk in the .wav format.

    """

    dir_path = Path(dir) / folder
    for file in get_files(dir_path, ext=".wav"):
        file_path = dir_path / file
        print(file_path)
        raw = AudioSegment.from_file(file_path, format="wav")
        clips_folder = file_path.with_name(file_path.stem)
        clips_folder.mkdir(parents=True, exist_ok=True)
        for index, clip in enumerate(raw[::duration]):
            clip_dir = clips_folder / f"{file_path.stem}_{index}.wav"
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
    files = list(dir.glob(f"*{ext}")) if ext else list(dir.iterdir())
    files = natsort.natsorted(files, key=lambda x: x.name)
    return [file.name for file in files]


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
        folder_path = Path(workdir) / folder
        folder_path.mkdir(parents=True, exist_ok=True)


def remove_small_files(dataset_dir: str):
    """
    This function removes audio files that are below 100kb in size from the specified directory and removes their corresponding entries in the metadata.csv file.

    Parameters:
        dataset_dir (str): A string representing the directory path containing the audio files and metadata.csv file. Must be in the LJSpeech format.

    Returns:
        None
    """
    dataset_path = Path(dataset_dir) / "wavs"
    bad_files = [
        file for file in dataset_path.iterdir() if file.stat().st_size < 100000
    ]
    df = pd.read_csv(dataset_dir + "/metadata.csv", sep="|", on_bad_lines="skip")
    for file in bad_files:
        row_to_remove = df[df["wav_filename"] == file.name]
        if not row_to_remove.empty:
            df = df.drop(row_to_remove.index)
            file.unlink()
    df.to_csv(dataset_dir + "/metadata.csv", sep="|", index=False)


def remove_extra_audio_files(dataset_dir: str):
    """
    This function removes audio files that are not listed in the metadata.csv file from the specified directory and removes their corresponding entries in the metadata.csv file.

    Parameters:
        dataset_dir (str): A string representing the directory path containing the audio files and metadata.csv file. Must be in the LJSpeech format.

    Returns:
        None
    """
    dataset_path = Path(dataset_dir) / "wavs"
    df = pd.read_csv(dataset_dir + "/metadata.csv", sep="|", on_bad_lines="skip")
    good_files = [f"{row[0]}.wav" for _, row in df.iterrows()]
    for file in dataset_path.iterdir():
        if file.name not in good_files:
            file.unlink()
            print(file.name + " removed")


def remove_extra_metadata(dataset_dir: str):
    """
    This function removes entries in the metadata.csv file that do not have corresponding audio files in the specified directory.

    Parameters:
        dataset_dir (str): A string representing the directory path containing the audio files and metadata.csv file. Must be in the LJSpeech format.
    """
    dataset_path = Path(dataset_dir) / "wavs"
    df = pd.read_csv(dataset_dir + "/metadata.csv", sep="|", on_bad_lines="skip")
    wav_files = [file.name for file in dataset_path.iterdir()]
    for _, row in df.iterrows():
        if f"{row[0]}.wav" not in wav_files:
            df = df.drop(row.name)
            print(row[0])
    df.to_csv(dataset_dir + "/metadata.csv", sep="|", index=False)
