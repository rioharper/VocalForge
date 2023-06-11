import os
import pandas as pd
from .text_utils import get_files
import shutil

class GenerateDataset():
    """
    Generates a dataset by processing audio files and corresponding metadata.

    TODO: 
        Add the ability for user to delete unwanted files, and have an autoupdating metadata when called
        List the dataset length (in seconds)

    Parameters:
        threshold (float): Threshold value used to filter audio data.

    Returns:
        None
    """
    def __init__(self, segment_dir, sliced_aud_dir, output_dir, threshold=2.5):
        self.Segment_Dir = segment_dir
        self.Sliced_Aud_Dir = sliced_aud_dir
        self.Out_Dir = output_dir
        self.Threshold = threshold
        self.Dataset = []


    # Create directories for storing processed data

    
    def create_metadata(self, file_path: str, thres: float):
        """
        Creates metadata for the audio data.

        Parameters:
            file_path (str): Path to audio file.
            thres (float): Threshold value used to filter audio data.

        Returns:
            pd.DataFrame: DataFrame object containing audio metadata.
        """
        thres = -thres
        name = []
        regular = []
        normalized = []
        punct = []
        with open(file_path, "r", encoding='UTF-8') as f:
            next(f)
            for index, line in enumerate(f):
                line = line.split("|")
                values = line[0].split()
                strings = line[1:]
                if float(values[2]) > thres:
                    index = format(index, '04')
                    file_name = file_path.split('/')[-1:]
                    file_name = file_name[0].split('seg')[0]
                    name.append(file_name+index)
                    regular.append(strings[0].strip())
                    normalized.append(strings[1].strip())
                    punct.append(strings[2].strip())

        metadata = {'name': name, 'regular': regular, 'normalized': normalized,
                    'punct': punct}
        df = pd.DataFrame(metadata)
        return df

    def create_dataset(self, metadata: pd.DataFrame):
        """
        Creates a dataset by copying audio files and saving metadata.

        Parameters:
            metadata (pd.DataFrame): Metadata for the audio data.
            dataset_dir (str): Directory where the dataset is to be saved.

        Returns:
            None
        """
        wav_dir = os.path.join(self.Out_Dir, 'wavs')
        try:
            os.mkdir(wav_dir)
        except:
            pass
        metadata.to_csv(os.path.join(self.Out_Dir, "metadata.csv"),
                        index=False, header=False, sep='|', encoding='utf-8-sig')
        
        for folder in get_files(self.Sliced_Aud_Dir):
            #TODO: check if file is entered on metadata.csv, currently copies regardless of validity
            aud_clips_dir = os.path.join(self.Sliced_Aud_Dir, folder)
            destination = shutil.copytree(aud_clips_dir, wav_dir, dirs_exist_ok=True)
    
    def run_dataset_generation(self):
        if os.listdir(self.Out_Dir) != []:
            print("Dataset has already been created! Skipping...")
            return
        for file in get_files(self.Segment_Dir, '.txt'):
            file_dir = os.path.join(self.Segment_Dir, file)
            self.Dataset.append(self.create_metadata(file_dir, self.Threshold))
        metadata = pd.concat(self.Dataset)
        self.create_dataset(metadata)
        self.Dataset = []
        print("Dataset has been created!")