from .audio_utils import get_files, concentrate_timestamps
import os
from pydub import AudioSegment

class Overlap:
    def __init__(self, input_dir=None, output_dir=None):
        self.Input_Dir = input_dir
        self.Output_Dir = output_dir
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