from .audio_utils import get_files
from pydub import AudioSegment
import os

class VoiceDetection: 
    def __init__(self, input_dir=None, output_dir=None, sample_dir=None, vad_threshold=75, snr_change=0.75):
        if sample_dir is not None:
            self.Input_Dir = sample_dir
        else:
            self.Input_Dir = input_dir
            self.Output_Dir = output_dir
        self.Vad_Threshold = vad_threshold
        self.SNR_Change = snr_change
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