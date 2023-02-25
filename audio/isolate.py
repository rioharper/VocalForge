from .audio_utils import get_files, remove_short_timestamps
import os
from pydub import AudioSegment
from scipy import spatial


class IsolateSpeaker():
    """Isolates speakers in file, then finds target speaker across all the files
        Parameters:
            verification_threshold: (float) The higher the value, the more similar 
            two voices must be during voice verification (float, default: 0.9)
            verification_dir: directory to seperate speakers
            isolated_dir: directory to export selected speaker
        Output: Verification and Isolated folders
    """
    #TODO: add the ability to include multiple target speakers
    def __init__(
        self, 
        input_dir=None, 
        verification_dir=None, 
        isolated_speaker_dir=None, 
        verification_threshold=0.90, 
        speaker_id=None, 
        speaker_fingerprint=None,
    ):
        from pyannote.audio import Inference
        from pyannote.audio import Model
        self.Input_Dir = input_dir
        self.Verification_Dir = verification_dir
        self.Isolated_Speaker_Dir=isolated_speaker_dir
        self.Verification_Threshold = verification_threshold
        model = Model.from_pretrained("pyannote/embedding", use_auth_token=True)
        self.Input_Files = get_files(self.Input_Dir)
        self.Speakers = []
        self.Inference = Inference(model, window="whole", device="cuda")
        self.Speaker_Id = speaker_id
        self.Speaker_Fingerprint = speaker_fingerprint

    
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


    def run_seperate_speakers(self):        
        if os.listdir(self.Verification_Dir) == []:
            self.find_speakers()
            self.separate_speakers()
        else: print("Speaker(s) have already been split! Skipping...")

    def create_fingerprint(self, file_dir):
        self.Speaker_Fingerprint = self.Inference(
                file_dir
        )
    
    def verify_file(self, file_dir):
        if os.stat(file_dir).st_size > 100:
            file_fingerprint = self.Inference(file_dir)
            difference = 1 - spatial.distance.cosine(file_fingerprint, self.Speaker_Fingerprint)
            if difference > self.Verification_Threshold:
                return file_dir
            else: return None
        else: return None

    def verify_folder(self, folder_dir):
        verified_files = []
        for file in get_files(folder_dir):
            file_dir = os.path.join(folder_dir, file)
            if self.verify_file(file_dir) is not None:
                verified_files.append(file_dir)
        return verified_files

    def combine_files(self, files_dir: list):
        combined_file = AudioSegment.empty()
        for file in files_dir:
            combined_file += AudioSegment.from_file(file, format="wav")
        return combined_file

    def run_verify(self):
        if os.listdir(self.Isolated_Speaker_Dir) == []:
            if self.Speaker_Id is None:
                self.Speaker_Id =  input("Enter Target Speaker Path (.wav): ")
            self.create_fingerprint(self.Speaker_Id)
            for folder in get_files(self.Verification_Dir):
                folder_dir = os.path.join(self.Verification_Dir, folder)
                verified_files = self.verify_folder(folder_dir)
                if verified_files is []:
                    continue
                verified_speaker = self.combine_files(verified_files)
                verified_speaker.export(os.path.join(self.Isolated_Speaker_Dir, folder)+'.wav', format="wav")