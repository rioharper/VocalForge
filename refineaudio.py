import os
from pydub import AudioSegment
from VocalForge.utils.utils import get_files, concentrate_timestamps, remove_short_timestamps, download_videos, create_core_folders, create_samples
import shutil
from scipy.io import wavfile
import numpy as np
from scipy import spatial

class NonSpeechRemover: 
    def __init__(self, vad_threshold: float, noise_aggressiveness: float, 
                 input_dir=None, output_dir=None, sample_dir=None):
        if input_dir is None:
            self.Input_Dir = rawdir
            self.Output_Dir = os.path.join(workdir, "Only_Voice")
            if len(os.listdir(os.path.join(workdir, "Samples"))) > 0:
                input_dir = os.path.join(workdir, "Samples")
        elif sample_dir is not None:
            self.Input_Dir = sample_dir
        else:
            self.Input_Dir = input_dir
            self.Output_Dir = output_dir
        self.Vad_Threshold = vad_threshold
        self.Noise_Aggressiveness = noise_aggressiveness
        self.Input_Files = get_files(self.Input_Dir, '.wav')
        self.Speech_Metrics = []
        self.Timestamps = []


    def analyze_vad(self):
        """
        This function analyzes audio files in a folder and performs voice activity detection (VAD)
        on the audio files. It uses the 'pyannote.audio' library's pre-trained 'brouhaha' model for the analysis.

        Parameters:
            input_files (list): list of files to analyze
        Returns:
            speech_metrics (list): List of voice activity detection output for each audio file.
        """
        from pyannote.audio import Inference
        from pyannote.audio import Model
        model = Model.from_pretrained("pyannote/brouhaha", 
                                    use_auth_token=True)
        inference = Inference(model)

        for file in self.Input_Files:
            output = inference(self.Input_Dir+"/"+file)
            self.Speech_Metrics.append(output)
            
        #return self.Speech_Metrics
        
    

    def find_timestamps(self) -> list:
        """
        This function processes speech metrics and returns timestamps
        of speech segments in the audio.
        
        Parameters:
        speech_metrics (list): list of speech metrics for each audio file
        
        Returns:
        timestamps (list): list of speech timestamps for each audio file
        """
        import statistics
        #TODO: check for if unused file in for loop is needed
        for fileindex, file in enumerate(self.Input_Files):
            nonspeech_timestamps = []
            startpoint = False
            c50_all = []
            first_iter = True
            
            # Calculate median of c50 values for the current audio file
            for frame, (c50) in self.Speech_Metrics[fileindex]:
                c50_all.append(c50[2])
            c50_med = float(statistics.median_high(c50_all))
            
            for frame, (vad, snr, c50) in self.Speech_Metrics[fileindex]:
                vad = vad *100
                t = frame.middle
                if first_iter:
                    nonspeech_timestamps.append([0, t])
                    first_iter = False
                if vad < self.Vad_Threshold and startpoint == False:
                    nonspeech_timestamps.append([t])
                    startpoint = True
                elif c50_med * self.Noise_Aggressiveness > c50 and startpoint == False:
                    nonspeech_timestamps.append([t])
                    startpoint = True
                elif c50_med * self.Noise_Aggressiveness < c50 and startpoint == True and vad > self.Vad_Threshold:
                    nonspeech_timestamps[len(nonspeech_timestamps)-1].append(t)
                    startpoint = False
            if len(nonspeech_timestamps[len(nonspeech_timestamps)-1]) == 1:
                nonspeech_timestamps.pop(len(nonspeech_timestamps)-1)
            # Get speech timestamps by concatenating non-speech timestamps
            speech_timestamps = []
            for index, stamps in enumerate(nonspeech_timestamps):
                try:
                    #if length between VAD timestamps is less than 4 seconds, combine them
                    if nonspeech_timestamps[index+1][0] - stamps[1] > 4:
                        stamplist = [stamps[1], nonspeech_timestamps[index+1][0]]
                        speech_timestamps.append(stamplist)
                except: pass
            #speech_timestamps = concentrate_timestamps(speech_timestamps)
            self.Timestamps.append(speech_timestamps)
        
        #return self.Timestamps


    def export_voices(self):
        """
        Given a list of timestamps for each file, the function exports 
        the speech segments from each raw file to a new file format wav. 
        The new files are saved to a specified directory. 

        Parameters:
        input_dir: (str): input dir of wav files
        output_dir: (str): output of VAD wav files
        timestamps (list): A list of timestamps for each file indicating 
        the start and end of speech segments.

        Returns: 
        None
        """
        for index, file in enumerate(self.Input_Files):
            # Load raw audio file
            raw = AudioSegment.from_file(f"{self.Input_Dir}/{file}", format="wav")
            entry = AudioSegment.empty()
            # Add speech segments to the new audio file
            for stamps in self.Timestamps[index]:
                entry += raw[stamps[0]*1000:stamps[1]*1000]
            try:
                entry += raw[self.Timestamps[index][len(self.Timestamps[index])-1][1]]
            except:
                pass
            # Check if the new audio file has enough speech segments
            if len(entry) > 1000:
                # Save the new audio file to the specified directory
                fentry = entry.export(f"{self.Output_Dir}/{file}", format='wav')
            else:
                print(f"{file} doesnt have enough clean audio to export")

    def run_vad(self):
        if os.listdir(self.Output_Dir) == []:
            self.analyze_vad()
            self.find_timestamps()
            self.export_voices()
            print(f"Analyzed files for voice detection")
        else: print(f"Files already exist in {self.Output_Dir}")
        
    

class OverlapRemover:
    def __init__(self, input_dir=None, output_dir=None):
        self.Input_Dir = input_dir
        self.Output_Dir = output_dir
        if input_dir is None:
            self.Input_Dir = os.path.join(workdir, "Only_Voice")
            self.Output_Dir = os.path.join(workdir, "No_Overlap")
        self.Input_Files = get_files(self.Input_Dir, '.wav')
        self.Overlap_Timelines = []
        self.Overlap_Timestamps = []


    def analyze_overlap(self) -> list:
        """
        Analyzes overlapping speech in a set of speech audio files.

        Parameters:
        input_dir: (str) dir of input wav files 
        
        Returns:
            overlap_timeline (list): A list of overlapping speech timestamps 
                                    for each file.
        """
        from pyannote.audio import Pipeline
        
        # Create a pipeline object using the pre-trained "pyannote/overlapped-speech-detection"
        pipeline = Pipeline.from_pretrained("pyannote/overlapped-speech-detection",
                                            use_auth_token=True)
        
        # Loop through each speech file in the speech_files list
        for file in self.Input_Files:
            try:
                # Use the pipeline to analyze the file for overlapping speech
                dia = pipeline(os.path.join(self.Input_Dir, file))
            except:
                # If the pipeline fails to analyze the file, print an error message
                print(f"{file} seems to have no data in it!")
            self.Overlap_Timelines.append(dia)


    def overlap_timestamps(self) -> list:
        """
        Converts overlap timelines into timestamps of non-overlapping speech turns

        Parameters:
            overlap (list): List of overlap timelines obtained from analyze_overlap function

        Returns:
            overlap_timestamps (list): List of timestamps of non-overlapping speech turns
        """
        for i in range(len(self.Overlap_Timelines)):
            timestamps = []
            for speech_turn, track, speaker in self.Overlap_Timelines[i].itertracks(yield_label=True):
                timestamps.append([speech_turn.start, speech_turn.end])
                #print(f"{speech_files[i]} {speech_turn.start:4.1f} {speech_turn.end:4.1f} {speaker}")
            timestamps = concentrate_timestamps(timestamps, 5)
            self.Overlap_Timestamps.append(timestamps)
        #return overlap_timestamps
    

    def export_no_overlap(self):
        """
        Exports the audio files with overlapped speech removed.
        
        Parameters:
        all_timestamps (list): List of timestamps of overlapped speech
                                                for each audio file.
        
        Returns:
        None
        """
        for index, file in enumerate(self.Input_Files):
            raw_data = AudioSegment.from_file(os.path.join(self.Input_Dir, file), format="wav")
            entry = AudioSegment.empty()
            
            #if no overlap is found
            if self.Overlap_Timestamps[index]== []:
                nentry = raw_data.export(f"{self.Output_Dir}/{file}", format='wav')
                continue
            #if only one overlap section is found
            
            if len(self.Overlap_Timestamps[index]) == 1:
                #print(f"deleting from {self.Overlap_Timestamps[index][0][0]} to ")
                entry += raw_data[:self.Overlap_Timestamps[index][0][0]*1000]
                entry += raw_data[self.Overlap_Timestamps[index][0][1]*1000:]
                nentry = entry.export(f"{self.Output_Dir}/{file}", format='wav')
                continue
            #if more than one overlap section is found
            entry += raw_data[:self.Overlap_Timestamps[index][0][0]*1000]
            try: #continue until last overlap section
                for timestamp_index, timestamp in enumerate(self.Overlap_Timestamps[index]):
                    entry += raw_data[timestamp[1]*1000:self.Overlap_Timestamps[index][timestamp_index+1][0]*1000]
            except: entry += raw_data[self.Overlap_Timestamps[index][len(self.Overlap_Timestamps[index])-1][1]*1000:]
            nentry = entry.export(f"{self.Output_Dir}/{file}", format='wav')

    def run_overlap(self):
        if os.listdir(self.Output_Dir) == []:
            self.analyze_overlap()
            self.overlap_timestamps()
            self.export_no_overlap()
            print("Exported non overlapped files!")
        else: print(f"Files already exist in {self.Output_Dir}")



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
        if input_dir is None:
            self.Input_Dir = os.path.join(workdir, "No_Overlap")
            self.Verification_Dir = os.path.join(workdir, "Verification")
            self.Isolated_Speaker_Dir = os.path.join(workdir, "Isolated")
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
    


class ExportAudio():
    def __init__(self, input_dir=None, export_dir=None, noise_removed_dir=None, normalization_dir = None, sample_rate=22050):
        if input_dir is None:
            self.Input_Dir = os.path.join(workdir, "Isolated")
            self.Export_Dir = os.path.join(workdir, "Exported")
            self.Noise_Removed_Dir = os.path.join(workdir, "Noise_Removed")
            self.Normalized_Dir = os.path.join(workdir, "Normalized")
        else:
            self.Input_Dir = input_dir
            self.Export_Dir = export_dir
            self.Noise_Removed_Dir = noise_removed_dir 
        self.Input_Files = get_files(self.Input_Dir)
        self.Sample_Rate = sample_rate
    
    

    def find_all_mean_sd(self, folders_dir: str) -> tuple:
        """
        This function finds the mean and standard deviation of all wav files in the
        given folder and its subfolders.
        
        Parameters:
        folders_dir (str): The directory of the folder where all the wav files are.
        
        Returns:
        Tuple[float, float]: The mean and standard deviation of all wav files.
        """
        mean = 0
        sd = 0
        count = 0
        for folder in get_files(folders_dir):
            for file in get_files(self.Input_Dir):
                rate, data = wavfile.read(os.path.join(self.Input_Dir, file))
                mean += np.mean(data)
                sd += np.std(data)
                count += 1
        mean /= count
        sd /= count
        return mean, sd


    def normalize_folder(self, folder_dir, mean, sd):
        import pathlib
        """
        TODO: Add other normalization methods
        Normalizes audio files in `folder` directory.

        Parameters:
        folder (str): The directory containing the audio files.
        mean (float): The mean value used for normalization.
        sd (float): The standard deviation value used for normalization.

        Returns:
        None

        """
        for file in get_files(folder_dir):
            file_dir = os.path.join(folder_dir, file)
            rate, data = wavfile.read(file_dir)
            mean_subtracted = data - mean
            eps = 2**-30
            output = mean_subtracted / (sd + eps)
            normalized_file_dir = os.path.join(self.Normalized_Dir, file)
            wavfile.write(normalized_file_dir, rate, output)

    def noise_remove_folder(self, folder_dir):
        from torch import cuda
        from df.enhance import enhance, init_df, load_audio, save_audio
        model, df_state, _ = init_df(config_allow_defaults=True)
        
        for file in get_files(folder_dir):
            try:
                file_dir = os.path.join(folder_dir, file)
                audio, _ = load_audio(file_dir, sr=df_state.sr())
                enhanced = enhance(model, df_state, audio)
                save_audio(os.path.join(self.Noise_Removed_Dir, file), enhanced, df_state.sr())
                cuda.empty_cache()
            except RuntimeError:
                print(f"file is too large for GPU, skipping: {file}")
        del model, df_state
        
         

    def format_audio_folder(self, folder_dir):
        for file in get_files(folder_dir, '.wav'):
            file_dir = os.path.join(folder_dir, file)
            raw = AudioSegment.from_file(file_dir, format="wav")
            raw = raw.set_channels(1)
            raw = raw.set_frame_rate(self.Sample_Rate)
            raw.export(os.path.join(self.Export_Dir, file), format='wav')
    
    def run_export(self):
        if os.listdir(self.Export_Dir) != []:
            print("file(s) have already been formatted! Skipping...")
        else: self.format_audio_folder(self.Input_Dir)            

        if self.Noise_Removed_Dir is not None and os.listdir(self.Noise_Removed_Dir)== []:
            print("Removing Noise...") 
            self.noise_remove_folder(self.Export_Dir)

        if self.Normalized_Dir is not None and os.listdir(self.Normalized_Dir)== []:
            mean, sd = self.find_all_mean_sd(self.Input_Dir)
            print("Normalizing Audio...")
            self.normalize_folder(self.Export_Dir, mean, sd)

class RefineAudio():
    def __init__(
        self,
        input_dir=None,
        vad_dir=None,
        overlap_dir=None,
        sample_dir=None,
        verification_dir=None,
        isolated_dir=None,
        noise_removed_dir=None,
        normalization_dir=None,
        export_dir=None,
        sample_rate=None,
        vad_theshold=None,
        noise_aggressiveness=None,
        verification_threshold=None,
        speaker_id=None,
    ):
        self.Input_Dir = input_dir
        self.VAD_dir = vad_dir
        self.Overlap_Dir = overlap_dir
        self.Sample_Dir = sample_dir
        self.Verification_Dir = verification_dir
        self.Isolated_Dir = isolated_dir
        self.Export_Audio_Dir = export_dir
        self.VAD_Threshold = vad_theshold
        self.Noise_Aggressiveness = noise_aggressiveness
        self.Verification_Threshold = verification_threshold
        self.Speaker_Id = speaker_id
        self.Noise_Removed_Dir = noise_removed_dir
        self.Normalized_Dir = normalization_dir
        self.Sample_Rate = sample_rate

        self.Non_Speech_Remover = NonSpeechRemover(
            vad_threshold=self.VAD_Threshold, 
            noise_aggressiveness=self.Noise_Aggressiveness,
            input_dir=self.Input_Dir,
            output_dir=self.VAD_dir,
            sample_dir=self.Sample_Dir,
        )
        self.Overlap_Remover = OverlapRemover(
            input_dir=self.VAD_dir,
            output_dir=self.Overlap_Dir,
        )
        self.Isolate_Speaker = IsolateSpeaker(
            input_dir=self.Overlap_Dir,
            verification_dir=self.Verification_Dir,
            isolated_speaker_dir=self.Isolated_Dir,
            verification_threshold=self.Verification_Threshold,
            speaker_id=self.Speaker_Id,
        )
        self.Export_Audio = ExportAudio(
            input_dir=self.Isolated_Dir,
            export_dir=self.Export_Audio_Dir,
            noise_removed_dir=self.Noise_Removed_Dir,
            normalization_dir=self.Normalized_Dir,
            sample_rate=self.Sample_Rate,
        )
    
    def run_all(self):
        self.Non_Speech_Remover.run_vad()
        self.Overlap_Remover.run_overlap()
        self.Isolate_Speaker.run_seperate_speakers()
        self.Isolate_Speaker.run_verify()
        self.Export_Audio.run_export()


import argparse
parser = argparse.ArgumentParser(description='Modify parameters for voice data refinement')
parser.add_argument("--raw_dir", 
    help="directory for unfitered audio (str, required)", 
    required=True)
parser.add_argument("--work_dir", 
    help="directory for the various stages of refinement (str, required)", 
    required=True)
parser.add_argument("--sample_rate",
    help="exported sample rate (int, default: 22050)",
    default=22050, type=int)
parser.add_argument("--playlist_url",
    help="URL to YouTube playlist to be downloaed to raw_dir (str)",
    type=str)
parser.add_argument("--vad_threshold",
    help="The higher the value, the more selective the VAD model will be (float, default: .75)",
    default=.75, type=float)
parser.add_argument("--snr_change", 
    help="The lower the value, the more sensitive the model is to changes in SNR, such as laughter or loud noises (float, default: 0.75)", 
    default=0.75, type=float)
parser.add_argument("--samples_length", 
    help="create sample voice clips from raw_dir for testing purposes (in seconds)", 
    type=int, default=None)
parser.add_argument("--verification_threshold",
    help="The higher the value, the more similar two voices must be during voice verification (float, range: 0.0-0.99, default: 0.9)",
    default=0.90, type=float)


args = parser.parse_args()
rawdir = args.raw_dir
workdir = args.work_dir

from torch import cuda
if cuda.is_available() == False:
    print("CUDA device not found! If you have CUDA intalled, please check if its propery configured")
    print("Program will continue, but at a much slower pace.")
else: print("CUDA device configured correctly!")

if args.playlist_url is not None:
    download_videos(args.playlist_url, rawdir)

folders = ['Isolated', 'No_Overlap', 'Samples', "Only_Voice", "Verification", "Exported", 'Noise_Removed', 'Normalized']
create_core_folders(folders, workdir)
if args.samples_length is not None:
    create_samples(args.samples_length, args.raw_dir, os.path.join(args.work_dir, 'Samples'))
    rawdir = os.path.join(args.work_dir, 'Samples')
    
Refine_Audio = RefineAudio(
    sample_rate=args.sample_rate,
    vad_theshold=args.vad_threshold,
    noise_aggressiveness=args.snr_change,
    verification_threshold=args.verification_threshold,
)
Refine_Audio.run_all()
