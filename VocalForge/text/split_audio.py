from .ctc_utils import process_alignment
from pathlib import Path
from .text_utils import get_files


class SplitAudio:
    """
    This function splits the audio into clips using the timestamps in the
    segmented audio folder.

    Args:
    1. input_dir (str): Directory path of the segmented audio files.
    2. output_dir (str): Directory path of the clips.
    3. offsets (list): List of offsets for each file, i.e [0, 0.5, 1.0] for 3 files.
    4. paddings (list): List of paddings for each file, i.e [0, 0.5, 1.0] for 3 files.
    5. max_duration (int): Max duration of the clips in seconds.
    6. threshold (float): Confidence threshold for the timing of the clips. The closer to 0, the more selective the clips will be. (cannot be > 0)
    """

    def __init__(
        self,
        input_dir,
        output_dir,
        offsets=[0],
        paddings=[0],
        max_duration=40,
        threshold=2.5,
    ):
        self.Input_Dir = Path(input_dir)
        self.Output_Dir = Path(output_dir)
        self.Offset = offsets
        self.Padding = paddings
        self.Max_Duration = max_duration
        self.Threshold = -threshold

    def run(self):
        for index, file in enumerate(get_files(str(self.Input_Dir), ".txt")):
            try:
                offset = self.Offset[index]
            except:
                offset = self.Offset[0]
            try:
                padding = self.Padding[index]
            except:
                padding = self.Padding[0]
            clips_folder_dir = self.Output_Dir / file.name.split("_segmented")[0]
            clips_folder_dir.mkdir(parents=True, exist_ok=True)
            file_dir = self.Input_Dir / file
            process_alignment(
                alignment_file=str(file_dir),
                clips_dir=str(clips_folder_dir),
                offset=offset,
                padding=padding,
                max_duration=self.Max_Duration,
                threshold=self.Threshold,
            )
