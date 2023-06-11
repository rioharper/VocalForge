from .transcribe import Transcribe
from .normalize_text import NormalizeText
from .segment import Segment
from.split_audio import SplitAudio
from .create_dataset import GenerateDataset
class RefineText():
    def __init__(
        self, 
        aud_dir, 
        transcription_dir=None,
        processed_text_dir=None,
        segment_dir=None,
        sliced_audio_dir=None,
        dataset_dir=None,
        model='nvidia/stt_en_citrinet_1024_gamma_0_25',
        window_size=8000,
        offset=0,
        max_duration=40,
        max_length=25,
        threshold=2.5,
        lang='en',
    ):
        self.Aud_Dir = aud_dir
        self.Transcription_Dir = transcription_dir
        self.Processed_Text_Dir = processed_text_dir
        self.Segmented_Dir = segment_dir
        self.Sliced_Audio_Dir = sliced_audio_dir
        self.Dataset_Dir = dataset_dir
        self.Model = model
        self.Max_Length = max_length
        self.Max_Duration = max_duration
        self.Window_Size = window_size
        self.Offset = offset
        self.Threshold = threshold
        self.Lang = lang

        self.Transcription = Transcribe(
            self.Aud_Dir, 
            self.Transcription_Dir
        )
        self.NormalizeText = NormalizeText(
            input_dir = self.Transcription_Dir, 
            out_dir= self.Processed_Text_Dir,
            audio_dir = self.Aud_Dir,
            model = self.Model,
            length = self.Max_Length,
            lang=self.Lang
        )
        self.Segment = Segment(
            input_dir = self.Processed_Text_Dir, 
            output_dir = self.Segmented_Dir,
            model=self.Model,
            window_size=self.Window_Size
        )
        self.Split = SplitAudio(
            input_dir = self.Segmented_Dir, 
            output_dir = self.Sliced_Audio_Dir,
            offset = self.Offset,
            max_duration = self.Max_Duration,
            threshold = self.Threshold
        )
        self.GenerateDataset = GenerateDataset(
            segment_dir = self.Segmented_Dir, 
            sliced_aud_dir = self.Sliced_Audio_Dir,
            output_dir = self.Dataset_Dir,
            threshold = self.Threshold
        )

    def refine_text(self):
        self.Transcription.run_trancription()
        self.NormalizeText.run_processing()
        self.Segment.run_segmentation()
        self.Split.run_slicing()
        self.GenerateDataset.run_dataset_generation()