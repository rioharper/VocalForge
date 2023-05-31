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


    from pyannote.audio import Inference
    from pyannote.audio import Model
    from scipy import spatial
    from pydub import AudioSegment
    import os

    class Isolate:
        """
        Isolates speakers in file, then finds target speaker across all the files
        
        Parameters:
            input_dir: directory containing audio files to be verified
            verification_dir: directory to separate speakers
            export_dir: directory to export selected speaker
            verification_threshold: (float) The higher the value, the more similar the
            two voices must be during voice verification (float, default: 0.9)
            lowest_threshold: (float) The lowest value the verification threshold can be if no speakers in the folder matches
            speaker_id: path to target speaker if known
            speaker_fingerprint: fingerprint of target speaker if already calculated
        """
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
            self.Export_Dir = export_dir
            self.Verification_Threshold = verification_threshold
            self.Lowest_Threshold = lowest_threshold
            self.Speaker_Id = speaker_id
            self.Speaker_Fingerprint = speaker_fingerprint
            self.Inference = Inference(model=Model())
            self.Input_Files = get_files(self.Input_Dir)
            self.Speakers = []
            self.run()

        def run_separate_speakers(self):        
            """
            Runs the speaker separation process if it has not already been done
            """
            if os.listdir(self.Verification_Dir) == []:
                self.find_speakers()
                self.separate_speakers()
            else: 
                print("Speaker(s) have already been split! Skipping...")

        def create_fingerprint(self, file_dir):
            """
            Creates a fingerprint for a given audio file
            
            Parameters:
                file_dir: path to the audio file
            """
            self.Speaker_Fingerprint = self.Inference(file_dir)

        def verify_file(self, file_dir):
            """
            Verifies if an audio file contains the target speaker
            
            Parameters:
                file_dir: path to the audio file
            
            Returns:
                file_dir if the file contains the target speaker, None otherwise
            """
            if os.stat(file_dir).st_size > 100:
                file_fingerprint = self.Inference(file_dir)
                difference = 1 - spatial.distance.cosine(file_fingerprint, self.Speaker_Fingerprint)
                if difference > self.Verification_Threshold:
                    return file_dir
                else: 
                    return None
            else: 
                return None

        def verify_folder(self, folder_dir):
            """
            Verifies all audio files in a folder and returns a list of verified files
            
            Parameters:
                folder_dir: path to the folder containing audio files
            
            Returns:
                A list of verified audio files
            """
            verified_files = []
            for file in get_files(folder_dir):
                file_dir = os.path.join(folder_dir, file)
                if self.verify_file(file_dir) is not None:
                    verified_files.append(file_dir)
            return verified_files

        def combine_files(self, files_dir: list):
            """
            Combines multiple audio files into a single audio file
            
            Parameters:
                files_dir: list of paths to the audio files to be combined
            
            Returns:
                A single audio file containing the combined audio data
            """
            combined_file = AudioSegment.empty()
            for file in files_dir:
                combined_file += AudioSegment.from_file(file, format="wav")
            return combined_file

        def run_verification(self):
            """
            Runs the speaker verification process if it has not already been done
            """
            if os.listdir(self.Export_Dir) == []:
                if self.Speaker_Id is None:
                    self.Speaker_Id =  input("Enter Target Speaker Path (.wav): ")
                if self.Speaker_Fingerprint is None:
                    self.create_fingerprint(self.Speaker_Id)
                temp_verification_thres = self.Verification_Threshold
                for folder in get_files(self.Verification_Dir):     
                    folder_dir = os.path.join(self.Verification_Dir, folder)
                    verified_files = self.verify_folder(folder_dir)
                    if verified_files == []:
                        while verified_files == [] and self.Verification_Threshold > self.Lowest_Threshold:
                            self.Verification_Threshold -= 0.05
                            verified_files = self.verify_folder(folder_dir)
                    self.Verification_Threshold = temp_verification_thres
                    verified_speaker = self.combine_files(verified_files)
                    verified_speaker.export(os.path.join(self.Export_Dir, folder)+'.wav', format="wav")

        def run(self):
            """
            Runs the entire process of speaker separation and verification
            """
            if os.listdir(self.Verification_Dir) == []:
                self.run_separate_speakers()
            if os.listdir(self.Export_Dir) == []:
                self.run_verification()
            else: 
                print("Speaker(s) have already been verified! Skipping...")

        def find_speakers(self) -> list:
            """
            Finds all speakers in the audio files and stores them in self.Speakers
            
            Returns:
                A list of all speakers found in the audio files
            """
            speakers = []
            for file in self.Input_Files:
                file_dir = os.path.join(self.Input_Dir, file)
                self.Speakers.append(self.Inference(file_dir))
                speakers += self.find_number_speakers(self.Speakers[-1])
            return list(set(speakers))

        def find_number_speakers(self, track) -> list:
            """
            Finds the number of speakers in a given audio track
            
            Parameters:
                track: audio track to be analyzed
            
            Returns:
                A list of all speakers found in the audio track
            """
            speakers = []
            for i, speaker in enumerate(track):
                speakers.append(i)
            return speakers

        def find_speakers_timestamps(self, file: tuple, speakers: list):
            """
            Finds the timestamps of speech parts for each speaker in a given audio file
            
            Parameters:
                file: tuple containing the audio file name and the audio track
                speakers: list of speakers in the audio track
            
            Returns:
                A list of tuples containing the start and stop times of speech parts for each speaker
            """
            speaker_timestamps = [[] for _ in range(len(speakers))]
            for i, speaker in enumerate(file[1]):
                for segment in speaker:
                    speaker_timestamps[i].append((segment.start, segment.end))
            return speaker_timestamps

        def separate_speakers(self):
            """
            Separates individual speakers from a list of speakers' tracks and saves their speech parts to a directory.
            """
            input_files = self.Input_Files
            for file_index, tracks in enumerate(self.Speakers):
                # Determine the number of speakers in the track and the timestamps of their speech parts
                speakers = self.find_number_speakers(tracks)
                speaker_timestamps = self.find_speakers_timestamps((input_files[file_index], tracks), speakers)
                
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

