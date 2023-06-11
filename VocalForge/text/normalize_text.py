import os
from pydub import AudioSegment
from .process_text import format_text, split_text, normalize_text
from .text_utils import get_files

class NormalizeText():
    """
    This function prepares the audio and text files for language modeling.
    
    Args:
    1. model (str): Model name to be used for processing the text.
    2. length (int): The max length of each sentence in the output file.
    3. lang (str): Language of the input audio and text files.
    
    Returns:
    None
    """
    def __init__(
        self, 
        input_dir: str, 
        out_dir: str, 
        audio_dir: str, 
        model='nvidia/stt_en_citrinet_1024_gamma_0_25', 
        length=25, 
        lang='en',
        min_length=0
    ):
        self.Model = model
        self.Length = length
        self.Min_Length = min_length
        self.Lang = lang
        self.Input_Dir = input_dir
        self.Out_Dir = out_dir
        self.Audio_Dir = audio_dir
        from nemo.collections.asr.models import ASRModel
        self.Cfg = ASRModel.from_pretrained(model_name=self.Model, return_config=True)  

    # Check if the folders have already been processed
    
    

    def prepare_audio_file(self, aud_dir, out_dir):
        """
        This function prepares the audio file for language modeling.
        
        Args:
        1. audio_dir (str): Directory path of the raw audio file.
        2. new_dir (str): Directory path of the processed audio file.
        
        Returns:
        None
        """
        # Load audio and export as WAV format
        raw = AudioSegment.from_file(aud_dir, format="wav")
        raw = raw.export(out_dir, format='wav')
    
    def prepare_text(self, textfile_dir:str):
        """
        This function prepares the text file for language modeling.
        
        Args:
        1. textfile_dir (str): Directory path of the text file.
        
        Returns:
        sentences (str): Processed text sentences.
        """
        # Format and normalize the text
        lang = self.Lang
        cfg = self.Cfg
        transcript = format_text(textfile_dir, lang)
        sentences = split_text(
            transcript, 
            lang, 
            cfg.decoder.vocabulary, 
            self.Length, 
            additional_split_symbols=None,
            min_length=self.Min_Length
        )
        return normalize_text(sentences, lang, cfg.decoder.vocabulary)

    def write_file(self, file_num:str, text_type:str, sentences:str, outdir:str):
        """
        This function writes the processed text to a file.
        
        Args:
        1. file_num (str): File number to be used in the file name.
        2. text_type (str): Type of text to be written to the file.
        3. sentences (str): Processed text sentences to be written to the file.
        
        Returns:
        None
        """
        file_name = f"{file_num}{text_type}"
        file_dir = os.path.join(outdir, file_num, file_name)
        print(file_dir)
        with open(file_dir+'.txt', 'w', encoding='UTF-8') as f:
            for sentence in sentences:
                sentence = sentence.strip()
                f.write(sentence+"\n")

    def prepare_file(self, text_dir, aud_dir, out_dir):
        #folder_dir = os.path.join(processed_dir, file.replace('.txt',''))
        base_name = os.path.basename(text_dir)
        sentence_types = self.prepare_text(text_dir)
        folder_dir = os.path.join(out_dir, base_name.replace('.txt', ''))
        try: os.mkdir(folder_dir)
        except: pass
        for name, sentences in sentence_types.items():
            self.write_file(base_name.replace('.txt', ''), name, sentences.splitlines(), out_dir)
        self.prepare_audio_file(aud_dir, os.path.join(folder_dir, base_name.replace('.txt', '.wav')))
    
    def run_processing(self):
        print(self.Out_Dir)
        if os.listdir(self.Out_Dir) != []:
            print("folder(s) have already been processed! Skipping...")
            return
        for file in get_files(self.Input_Dir, '.txt'):
            text_dir = os.path.join(self.Input_Dir, file)
            audfile_dir = os.path.join(self.Audio_Dir, file.replace('.txt', '.wav'))
            self.prepare_file(text_dir, audfile_dir, self.Out_Dir)