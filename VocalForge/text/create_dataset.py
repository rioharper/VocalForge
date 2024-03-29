from pathlib import Path
import pandas as pd
from .text_utils import get_files
import shutil


class GenerateDataset:
    """
    Generates a dataset by processing audio files and corresponding metadata.

    Parameters:
        segment_dir (str): Directory path of the segmented audio files.
        sliced_aud_dir (str): Directory path of the sliced audio files.
        output_dir (str): Directory path of the dataset.
        threshold (float): Threshold value of confidence from CTC segmentation. The closer to 0, the more selective the clips will be. (cannot be > 0)

    Generates:
        Dataset (list): List of metadata for each audio file.

    TODO:
        Add the ability for user to delete unwanted files, and have an autoupdating metadata when called
        List the dataset length (in seconds)
    """

    def __init__(self, segment_dir, sliced_aud_dir, output_dir, threshold=2.5):
        self.Segment_Dir = Path(segment_dir)
        self.Sliced_Aud_Dir = Path(sliced_aud_dir)
        self.Output_Dir = Path(output_dir)
        self.Threshold = threshold
        self.Dataset = []

    def create_metadata(self, file_path: Path, thres: float):
        """
        Creates metadata for the audio data.

        Parameters:
            file_path (Path): Path to audio file.
            thres (float): Threshold value used to filter audio data.

        Returns:
            pd.DataFrame: DataFrame object containing audio metadata.
        """
        thres = -thres
        name = []
        regular = []
        normalized = []
        punct = []

        with file_path.open("r", encoding="UTF-8") as f:
            next(f)
            for index, line in enumerate(f):
                line = line.split("|")
                values = line[0].split()
                strings = line[1:]
                if float(values[2]) > thres:
                    index = format(index, "04")
                    file_name = file_path.name.split("seg")[0]
                    name.append(file_name + index)
                    regular.append(strings[0].strip())
                    normalized.append(strings[1].strip())
                    punct.append(strings[2].strip())

        metadata = {
            "name": name,
            "regular": regular,
            "normalized": normalized,
            "punct": punct,
        }
        df = pd.DataFrame(metadata)
        return df

    def create_dataset(self, metadata: pd.DataFrame):
        """
        Creates a dataset by copying audio files and saving metadata.

        Parameters:
            metadata (pd.DataFrame): Metadata for the audio data.
        """
        wav_dir = self.Output_Dir / "wavs"
        wav_dir.mkdir(parents=True, exist_ok=True)

        metadata.to_csv(
            self.Output_Dir / "metadata.csv",
            index=False,
            header=False,
            sep="|",
            encoding="utf-8-sig",
        )

        for folder in get_files(str(self.Sliced_Aud_Dir)):
            shutil.copytree(self.Sliced_Aud_Dir / folder, wav_dir, dirs_exist_ok=True)

    def run(self):
        """
        Runs the dataset generation process.
        """
        for file in get_files(str(self.Segment_Dir), ".txt"):
            file_dir = self.Segment_Dir / file
            self.Dataset.append(self.create_metadata(file_dir, self.Threshold))
        metadata = pd.concat(self.Dataset)
        self.create_dataset(metadata)
        self.Dataset = []
