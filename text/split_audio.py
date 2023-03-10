from .ctc_utils import process_alignment
import os
from .text_utils import get_files

class SplitAudio():
    """
    This function splits the audio into clips using the timestamps in the
    segmented audio folder.

    Args:
    - offset: The number of seconds to add or subtract from the timestamps.
    - max_duration: The maximum duration of each clip.
    - threshold: The threshold to use for processing the alignment.

    Returns:
    None

    """
    def __init__(self, input_dir, output_dir, offsets=[0], paddings=[0], max_duration=40, threshold=2.5):
        self.Input_Dir = input_dir
        print(self.Input_Dir)
        self.Out_Dir = output_dir
        self.Offset = offsets
        self.Padding = paddings
        self.Max_Duration = max_duration
        self.Threshold = -threshold
        

    def run_slicing(self):
        if os.listdir(self.Out_Dir) != []:
            print("audio has already been sliced! Skipping...")
            return
        for index, file in enumerate(get_files(self.Input_Dir, '.txt')):
            try:
                offset = self.Offset[index]
            except:
                offset = self.Offset[0]
            try:
                padding = self.Padding[index]
            except:
                padding = self.Padding[0]
            clips_folder_dir = os.path.join(self.Out_Dir, file.split('_segmented')[0])
            os.mkdir(clips_folder_dir)
            file_dir = os.path.join(self.Input_Dir, file)
            process_alignment(
                alignment_file=file_dir,
                clips_dir=clips_folder_dir,
                offset = offset,
                padding = padding,
                max_duration = self.Max_Duration, 
                threshold = self.Threshold
            )