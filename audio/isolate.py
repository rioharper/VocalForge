from .audio_utils import get_files, remove_short_timestamps
import os
from pydub import AudioSegment
from scipy import spatial
from pyannote.audio import Inference
from pyannote.audio import Model

class Isolate:
    """Isolates speakers in file, then finds target speaker across all the files
        Parameters:
            isolated_dir: directory to export selected speaker
            speaker_id: path to target speaker if known
            speaker_fingerprint: fingerprint of target speaker if already calculated
            verification_threshold: (float) The higher the value, the more similar the
            two voices must be during voice verification (float, default: 0.9)
            lowest_threshold: (float) The lowest value the verification threshold can be if no speakers in the folder matches
            verification_dir: directory to seperate speakers
    """
    #TODO: add the ability to include multiple target speakers
    def __init__(
        self, 
        input_dir=None, 
        verification_dir=None, 
        export_dir=None, 
        verification_threshold=0.90, 
        lowest_threshold = 0.5,
        speaker_id=None, 
        speaker_fingerprint=None,
    ):

        self.Input_Dir = input_dir
        self.Verification_Dir = verification_dir
        self.Export_Dir=export_dir
        self.Verification_Threshold = verification_threshold
        model = Model.from_pretrained("pyannote/embedding", use_auth_token=True)
        self.Input_Files = get_files(self.Input_Dir)
        self.Speakers = []
        self.Inference = Inference(model, window="whole", device="cuda")
        self.Speaker_Id = speaker_id
        self.Speaker_Fingerprint = speaker_fingerprint
        self.Lowest_Threshold = lowest_threshold
    
    def find_speakers(self) -> list:
        """
        Finds the different speakers from the audio files in `overlap_dir` and
        returns a list of `SpeakerDiarization` instances.

        Parameters:
        -----------
        files: list of strings
            List of audio file names in `overlap_dir`
            
        Returns:
        --------
        speakers: list of SpeakerDiarization
            List of `SpeakerDiarization` instances, one for each audio file in `files`
        """

        from pyannote.audio import Pipeline
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@develop",
                                            use_auth_token=True)
        for file in self.Input_Files:
            dia = pipeline(os.path.join(self.Input_Dir, file))
            self.Speakers.append(dia)
            #print(f"Seperated speakers for {file}")

    
    def find_number_speakers(self, track) -> list:
        """
        Find the number of speakers in a given a list of pyannote tracks.

        Parameters:
        tracks (list): PyAnnote annotation object representing a speaker track.
        index (int): Index of the current audio file being processed.

        Returns:
        List[str]: A list of unique speaker names in the given track.
        """

        speakers = []
        for speech_turn, track, speaker in track.itertracks(yield_label=True):
            if speaker not in speakers:
                speakers.append(speaker)
        #print(f"File {input_dir[index]} has {len(speakers)} speaker(s)")
        return speakers

    
    def find_speakers_timestamps(self, file: tuple, speakers: list):
        """
        This function receives a file with speech segments and speakers
        labels and returns a list of speech timestamps for each speaker.

        Parameters:
        file: pyannote.core.Annotation - file containing speech segments and speakers
        speakers: list - list of speakers in the file

        Returns:
        list: list of speech timestamps for each speaker

        """
        timestamps = [ [] for i in range(len(speakers)) ]
        for speech_turn, track, speaker in file.itertracks(yield_label=True):
            speaker = speaker.split("_")[1]
            speaker = int(speaker)
            timestamps[speaker].append([speech_turn.start, speech_turn.end])

        for index, speaker in enumerate(timestamps):
            timestamps[index] = remove_short_timestamps(speaker, 1)
        return timestamps

            
    def separate_speakers(self):
        """
        Separates individual speakers from a list of speakers' tracks and saves their speech parts to a directory.
        """
        import os
        
        input_files = self.Input_Files
        
        for file_index, tracks in enumerate(self.Speakers):
            # Determine the number of speakers in the track and the timestamps of their speech parts
            speakers = self.find_number_speakers(tracks)
            speaker_timestamps = self.find_speakers_timestamps(tracks, speakers)
            
            # Load the audio file and extract the speech parts for each speaker
            audio_data = AudioSegment.from_file(os.path.join(self.Input_Dir, input_files[file_index]), format="wav")
            for speaker_index, timestamps in enumerate(speaker_timestamps):
                speaker_data = AudioSegment.empty()
                for start, stop in timestamps:
                    speaker_data += audio_data[start * 1000: stop * 1000]
                
                # Create a directory for the speaker's audio file and save it
                folder_name = os.path.splitext(input_files[file_index])[0]
                speaker_dir = os.path.join(self.Verification_Dir, folder_name)
                if not os.path.exists(speaker_dir):
                    os.mkdir(speaker_dir)
                speaker_file = os.path.join(speaker_dir, f"{speakers[speaker_index]}.wav")
                speaker_data.export(speaker_file, format="wav")