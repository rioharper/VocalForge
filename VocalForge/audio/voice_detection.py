from .audio_utils import get_files
from .audio_utils import get_timestamps
from .audio_utils import export_from_timestamps
from pathlib import Path


class VoiceDetection:
    def __init__(self, input_dir=None, output_dir=None, sample_dir=None, hparams=None):
        """
        Initializes a new instance of the VoiceDetection class.

        Parameters:
            input_dir (str): The directory containing the input audio files to analyze.
            output_dir (str): The directory where the output audio files will be saved.
            sample_dir (str): The directory containing sample audio files to analyze.
        """
        if sample_dir is not None:
            self.Input_Dir = Path(sample_dir)
        else:
            self.Input_Dir = Path(input_dir)
            self.Output_Dir = Path(output_dir)
        self.Input_Files = get_files(str(self.Input_Dir), True, ".wav")
        self.Timelines = []
        self.Timestamps = []
        self.Hparams = hparams
        from pyannote.audio import Pipeline

        self.Pipeline = Pipeline.from_pretrained(
            "pyannote/voice-activity-detection", use_auth_token=True
        )

        # instantiate the pipeline with hyperparameters if declared
        self.isHparams = False
        if self.Hparams is not None:
            self.Pipeline.instantiate(self.Hparams)
            self.isHparams = True

    def analyze_folder(self):
        """
        Analyzes audio files in a folder and performs voice activity detection (VAD)
        on the audio files. It uses the 'pyannote.audio' library's pre-trained 'brouhaha' model for the analysis.
        """

        if self.Hparams is not None and self.isHparams == False:
            self.Pipeline.instantiate(self.Hparams)
            self.isHparams = True

        for file in self.Input_Files:
            output = self.Pipeline(file)
            self.Timelines.append(output)

    def analyze_file(self, path):
        if self.Hparams is not None and self.isHparams == False:
            self.Pipeline.instantiate(self.Hparams)
            self.isHparams = True
        """function to analyze a single file"""
        return self.Pipeline(path)

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

    def update_timeline(self, new_timeline, index: int):
        """
        This function updates the timeline for a given file with the new timestamps due to finetuning
        """
        self.Timelines[index] = new_timeline

        self.Timestamps[index] = get_timestamps(new_timeline)

    def export(self):
        """
        Given a list of timestamps for each file, the function exports
        the speech segments from each raw file to a new file format wav.
        The new files are saved to a specified directory.
        """
        for index, file in enumerate(self.Input_Files):
            base_file_name = Path(file).name
            export_from_timestamps(
                file,
                str(self.Output_Dir / base_file_name),
                self.Timestamps[index],
            )

    def run(self):
        """runs the voice detection pipeline"""
        if list(self.Input_Dir.glob("*")) != []:
            self.analyze_folder()
        if self.Timelines != []:
            self.find_timestamps()
        if self.Timestamps != []:
            self.export()
        print(f"Analyzed files for voice detection")
