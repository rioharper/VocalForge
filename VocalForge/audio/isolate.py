from .audio_utils import get_files, remove_short_timestamps
from pathlib import Path
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

    # TODO: add the ability to include multiple target speakers
    def __init__(
        self,
        input_dir=None,
        verification_dir=None,
        export_dir=None,
        verification_threshold=0.90,
        lowest_threshold=0.5,
        speaker_id=None,
        speaker_fingerprint=None,
    ):
        self.Input_Dir = Path(input_dir)
        self.Verification_Dir = Path(verification_dir)
        self.Export_Dir = Path(export_dir)
        self.Verification_Threshold = verification_threshold
        model = Model.from_pretrained("pyannote/embedding", use_auth_token=True)
        self.Input_Files = get_files(str(self.Input_Dir))
        self.Speakers = []
        self.Inference = Inference(model, window="whole", device="cuda")
        self.Speaker_Id = Path(speaker_id) if speaker_id else None
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

        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization@develop", use_auth_token=True
        )
        for file in self.Input_Files:
            dia = pipeline(str(self.Input_Dir / file))
            self.Speakers.append(dia)
            # print(f"Seperated speakers for {file}")

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
        # print(f"File {input_dir[index]} has {len(speakers)} speaker(s)")
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
        timestamps = [[] for i in range(len(speakers))]
        for speech_turn, track, speaker in file.itertracks(yield_label=True):
            speaker = speaker.split("_")[1]
            speaker = int(speaker)
            timestamps[speaker].append([speech_turn.start, speech_turn.end])

        for index, speaker in enumerate(timestamps):
            timestamps[index] = remove_short_timestamps(speaker, 1)
        return timestamps

    def export_speakers(self):
        """
        Separates individual speakers from a list of speakers' tracks and saves their speech parts to a directory.
        """
        for file_index, tracks in enumerate(self.Speakers):
            # Determine the number of speakers in the track and the timestamps of their speech parts
            speakers = self.find_number_speakers(tracks)
            speaker_timestamps = self.find_speakers_timestamps(tracks, speakers)

            # Load the audio file and extract the speech parts for each speaker
            audio_data = AudioSegment.from_file(
                str(self.Input_Dir / self.Input_Files[file_index]), format="wav"
            )
            for speaker_index, timestamps in enumerate(speaker_timestamps):
                speaker_data = AudioSegment.empty()
                for start, stop in timestamps:
                    speaker_data += audio_data[start * 1000 : stop * 1000]

                # Create a directory for the speaker's audio file and save it
                folder_name = (self.Input_Dir / self.Input_Files[file_index]).stem
                speaker_dir = self.Verification_Dir / folder_name

                if not speaker_dir.exists():
                    speaker_dir.mkdir()

                speaker_file = speaker_dir / f"{speakers[speaker_index]}.wav"
                speaker_data.export(str(speaker_file), format="wav")

    def run_separate_speakers(self):
        """
        Runs the speaker separation process if it has not already been done
        """
        if not any(self.Verification_Dir.iterdir()):
            self.find_speakers()
            self.export_speakers()
        else:
            print("Speaker(s) have already been split! Skipping...")

    # ---------------------Speaker Verification---------------------#

    def create_fingerprint(self, file_dir):
        """
        Creates a fingerprint for a given audio file

        Parameters:
            file_dir: path to the audio file
        """
        self.Speaker_Fingerprint = self.Inference(str(file_dir))

    def verify_file(self, file_dir):
        """
        Verifies if an audio file contains the target speaker

        Parameters:
            file_dir: path to the audio file

        Returns:
            file_dir if the file contains the target speaker, None otherwise
        """
        if file_dir.stat().st_size > 100:
            file_fingerprint = self.Inference(str(file_dir))
            difference = 1 - spatial.distance.cosine(
                file_fingerprint, self.Speaker_Fingerprint
            )
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
        for file in get_files(str(folder_dir)):
            file_dir = folder_dir / file
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
            combined_file += AudioSegment.from_file(str(file), format="wav")
        return combined_file

    def run_verification(self):
        """
        Runs the speaker verification process if it has not already been done
        """
        if not any(self.Export_Dir.iterdir()):
            if self.Speaker_Id is None:
                self.Speaker_Id = input("Enter Target Speaker Path (.wav): ")
            if self.Speaker_Fingerprint is None:
                self.create_fingerprint(self.Speaker_Id)
            temp_verification_thres = self.Verification_Threshold
            for folder in get_files(str(self.Verification_Dir)):
                folder_dir = self.Verification_Dir / folder
                verified_files = self.verify_folder(folder_dir)
                if verified_files == []:
                    while (
                        verified_files == []
                        and self.Verification_Threshold > self.Lowest_Threshold
                    ):
                        self.Verification_Threshold -= 0.05
                        verified_files = self.verify_folder(folder_dir)
                self.Verification_Threshold = temp_verification_thres
                if verified_files != []:
                    verified_speaker = self.combine_files(verified_files)
                    verified_speaker.export(
                        str(self.Export_Dir / f"{folder}.wav"), format="wav"
                    )
                else:
                    print(
                        f"Speaker not found in {folder_dir}. You may need to lower the verification threshold if the speaker is present."
                    )

    def run_all(self):
        """
        Runs the entire process of speaker separation and verification
        """
        if not any(self.Verification_Dir.iterdir()):
            self.run_separate_speakers()
        if not any(self.Export_Dir.iterdir()):
            self.run_verification()
        else:
            print("Speaker(s) have already been verified! Skipping...")
