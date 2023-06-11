from .audio_utils import get_files
from .audio_utils import get_timestamps, export_from_timestamps
import os
from pyannote.audio import Pipeline

class Overlap:
    def __init__(self, input_dir=None, output_dir=None, hparams=None):
        self.Input_Dir = input_dir
        self.Output_Dir = output_dir
        self.Input_Files = get_files(self.Input_Dir, True, '.wav')
        self.Timelines = []
        self.Timestamps = []
        self.Hparams = hparams
        
        
        # Create a pipeline object using the pre-trained "pyannote/overlapped-speech-detection"
        self.Pipeline = Pipeline.from_pretrained("pyannote/overlapped-speech-detection",
                                            use_auth_token=True)
        
        self.isHparams = False
        if self.Hparams is not None:
            self.Pipeline.instantiate(self.Hparams)
            self.isHparams = True

    def analyze(self) -> list:
        """
        Analyzes overlapping speech in a set of speech audio files.

        Parameters:
        input_dir: (str) dir of input wav files 
        """
        
        
        if self.Hparams is not None and self.isHparams == False:
            self.Pipeline.instantiate(self.Hparams)
            self.isHparams = True
        
        for file in self.Input_Files:
            overlap_timeline = self.Pipeline(file)
            self.Timelines.append(overlap_timeline)


    def find_timestamps(self):
        """
        This function processes speech metrics and returns timestamps
        of overlapping segments in the audio."""
        self.Timestamps = []
        for fileindex in range(len(self.Input_Files)):
            timestamps = get_timestamps(self.Timelines[fileindex])
            self.Timestamps.append(timestamps)

    
    def update_timeline(self, new_timeline, index: int):
        """
        This function updates the timeline for a given file with the new timestamps due to finetuning
        """
        self.Timelines[index] = new_timeline

        self.Timestamps[index] = get_timestamps(new_timeline)
    

    def test_export(self):
        for index, file in enumerate(self.Input_Files):
            base_file_name = file.split('/')[-1]
            export_from_timestamps(file, os.path.join(self.Output_Dir, base_file_name), self.Timestamps[index], combine_mode='time_between')

    def run(self):
        """runs the overlap detection pipeline"""
        if os.listdir(self.Input_Dir) != []:
            self.analyze()
        if self.Timelines != []:
            self.find_timestamps()
            print("Found timestamps")
        if self.Timestamps != []:
            self.test_export()
        print(f"Analyzed files for voice detection")