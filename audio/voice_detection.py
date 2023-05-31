
from .audio_utils import get_files
from .audio_utils import get_timestamps
from .audio_utils import export_from_timestamps
import os

class VoiceDetection: 
    def __init__(self, input_dir=None, output_dir=None, sample_dir=None):
        """
        Initializes a new instance of the VoiceDetection class.

        Parameters:
            input_dir (str): The directory containing the input audio files to analyze.
            output_dir (str): The directory where the output audio files will be saved.
            sample_dir (str): The directory containing sample audio files to analyze.
        """
        if sample_dir is not None:
            self.Input_Dir = sample_dir
        else:
            self.Input_Dir = input_dir
            self.Output_Dir = output_dir
        self.Input_Files = get_files(self.Input_Dir, True, '.wav')
        self.Timelines = []
        self.Timestamps = []


    def analyze(self):
        """
        Analyzes audio files in a folder and performs voice activity detection (VAD)
        on the audio files. It uses the 'pyannote.audio' library's pre-trained 'brouhaha' model for the analysis.

        Parameters:
            input_files (list): list of files to analyze
        Returns:
            Timelines (list): List of voice activity detection output for each audio file.
        """
        from pyannote.audio import Pipeline
        vad = Pipeline.from_pretrained("pyannote/voice-activity-detection", 
                                    use_auth_token=True)

        for file in self.Input_Files:
            output = vad(file)
            self.Timelines.append(output)
                    
    

    def find_timestamps(self):
        """
        This function processes speech metrics and returns timestamps
        of speech segments in the audio.
        
        Parameters:
        speech_metrics (list): list of speech metrics for audio file(s)
        
        Returns:
        Timestamps (list): list of speech timestamps for each audio file
        """
        self.Timestamps = []
        for fileindex in range(len(self.Input_Files)):
            timestamps = get_timestamps(self.Timelines[fileindex])
            self.Timestamps.append(timestamps)


    def export(self):
        """
        Given a list of timestamps for each file, the function exports 
        the speech segments from each raw file to a new file format wav. 
        The new files are saved to a specified directory. 
        """
        for index, file in enumerate(self.Input_Files):
            base_file_name = file.split('/')[-1]
            export_from_timestamps(file, os.path.join(self.Output_Dir, base_file_name), self.Timestamps[index])

    def run(self):
        """runs the voice detection pipeline"""
        if os.listdir(self.Input_Dir) != []:
            self.analyze()
        if self.Timelines != []:
            self.find_timestamps()
        if self.Timestamps != []:
            self.export()
        print(f"Analyzed files for voice detection")
