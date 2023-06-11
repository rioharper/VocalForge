from .text_utils import get_files
import os

class Transcribe():
    """
    Transcribe audio files in a directory using a pre-trained ASR model.
    
    Returns:
        None
        
    Note:
        The function prints the names of the transcribed files and skips the ones 
        that have already been transcribed. If a file is corrupted, the function 
        prints "file corrupted! skipping...".
    """
    def __init__(self, raw_dir, out_dir, prompt=None):
        self.Raw_Dir = raw_dir
        self.Out_Dir = out_dir
        self.Prompt = prompt
        from whisper import load_model
        self.Model = load_model("large")

    def transcribe_folder(self, input_dir, do_write=True):
        """
        Transcribe all audio files in a folder.
        
        Args:
            folder (str): The path to the folder containing audio files.
            
        Returns:
            None
            
        Note:
            The function transcribes each audio file in the folder and writes the 
            transcription to a text file with the same name. If the text file 
            already exists, the function skips the transcription.
        """

        for file in get_files(input_dir):
            #try:
                text_file_dir = os.path.join(self.Out_Dir, file.split(".")[0]+".txt")
                # check if the text file already exists
                if not os.path.exists(text_file_dir):
                    # transcribe the audio file
                    aud_file_dir = os.path.join(input_dir, file)
                    result = self.Model.transcribe(aud_file_dir, initial_prompt=self.Prompt)
                    if do_write:
                        self.write_transcription(result, text_file_dir)
            # except:
            #     # print an error message if the audio file is corrupted
            #     print("file corrupted! skipping...")
    
    def write_transcription(self, result, text_file_dir):
        with open(text_file_dir, 'w', encoding="utf-8") as f:
            f.write(result['text'].strip())

    # transcribe all audio files in the `aud_dir` folder
    def run_trancription(self): 
        if os.listdir(self.Out_Dir) != []:
            print("folder(s) have already been transcribed! Skipping...")
            return
        self.transcribe_folder(self.Raw_Dir)