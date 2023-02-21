import os
from utils.utils import create_core_folders, get_files
from pydub import AudioSegment
from utils import process_text
import statistics
import pandas as pd
import os
import shutil

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
    def __init__(self, raw_dir, out_dir=None):
        self.Raw_Dir = raw_dir
        self.Out_Dir = out_dir
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
            try:
                text_file_dir = os.path.join(self.Out_Dir, file.split(".")[0]+".txt")
                # check if the text file already exists
                if not os.path.exists(text_file_dir):
                    # transcribe the audio file
                    aud_file_dir = os.path.join(input_dir, file)
                    result = self.Model.transcribe(aud_file_dir)
                    if do_write:
                        self.write_transcription(result, text_file_dir)
            except:
                # print an error message if the audio file is corrupted
                print("file corrupted! skipping...")
    
    def write_transcription(self, result, text_file_dir):
        with open(text_file_dir, 'w', encoding="UTF-8") as f:
            f.write(result['text'].strip())

    # transcribe all audio files in the `aud_dir` folder
    def run_trancription(self): 
        if os.listdir(self.Out_Dir) != []:
            print("folder(s) have already been transcribed! Skipping...")
            return
        self.transcribe_folder(self.Raw_Dir)


class ProcessText():
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
        lang='en'
    ):
        self.Model = model
        self.Length = length
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
        transcript = process_text.format_text(textfile_dir, lang)
        sentences = process_text.split_text(
            transcript, 
            lang, 
            cfg.decoder.vocabulary, 
            self.Length, 
            additional_split_symbols=None
        )
        return process_text.normalize_text(sentences, lang, cfg.decoder.vocabulary)

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


class Segment():
    """
    This function segments audio files into multiple parts based on the timestamps obtained
    using a model. The timestamps are used to split the audio file into smaller parts and
    are saved in a text file.

    Parameters:
    model (str): Pretrained model name.
    window_size (int): Size of the window for the model.

    Returns:
    None
    """
    def __init__(self, input_dir, output_dir, model='nvidia/stt_en_citrinet_1024_gamma_0_25', window_size=8000):
        self.Input_Dir = input_dir
        self.Out_Dir = output_dir
        self.Window_Size = window_size
        import nemo.collections.asr as nemo_asr
        self.Model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model)
        self.Loss = []
        self.Median_Loss = None

    def segment_folder(self, folder_dir, out_dir):
        from utils import ctc
        """
        This function segments audio files in a folder into multiple parts based on
        the timestamps obtained using a model.

        Parameters:
        folder (str): Folder containing audio files.

        Returns:
        float: Mean loss obtained from the model.
        """
        loss_folder = []
        for aud_file in get_files(folder_dir, '.wav'):
            aud_path = os.path.join(folder_dir, aud_file)
            outfile = os.path.join(out_dir, aud_file.replace('.wav', '_segmented.txt'))
            loss = ctc.ctc(self.Model, aud_path, outfile, self.Window_Size)
            loss_folder.append(loss)
        try:
            return statistics.mean(loss_folder)
        except: return None

    def find_median_loss(self):
        self.Median_Loss = statistics.median(self.Loss)

    def run_segmentation(self):
        if os.listdir(self.Out_Dir) != []:
            print("folder(s) have already found its split timestamps! Skipping...")
            return
        for folder in get_files(self.Input_Dir):
            folder_dir = os.path.join(self.Input_Dir, folder)
            value = self.segment_folder(folder_dir, self.Out_Dir)
            if value != None: self.Loss.append(value)
        self.Median_Loss = statistics.median(self.Loss)
        self.find_median_loss()
        print(f"median loss: {self.Median_Loss}")


class SplitAudio():
    """
    This function splits the audio into clips using the timestamps in the
    segmented audio folder.

    Args:
    - offset: The number of seconds to add or subtract from the timestamps.
    - max_duration: The maximum duration of each clip.
    - threshold: The threshold to use for processing the alignment.

    Returns:
    None

    """
    def __init__(self, input_dir, output_dir, offset=0, max_duration=40, threshold=2.5):
        self.Input_Dir = input_dir
        print(self.Input_Dir)
        self.Out_Dir = output_dir
        self.Offset = offset
        self.Max_Duration = max_duration
        self.Threshold = -threshold
        

    def run_slicing(self):
        from utils.utils import process_alignment
        if os.listdir(self.Out_Dir) != []:
            print("audio has already been sliced! Skipping...")
            return
        for file in get_files(self.Input_Dir, '.txt'):
            clips_folder_dir = os.path.join(self.Out_Dir, file.split('_segmented')[0])
            os.mkdir(clips_folder_dir)
            file_dir = os.path.join(self.Input_Dir, file)
            process_alignment(
                alignment_file=file_dir,
                clips_dir=clips_folder_dir,
                offset = self.Offset, 
                max_duration = self.Max_Duration, 
                threshold = self.Threshold
            )




class generate_dataset():
    """
    Generates a dataset by processing audio files and corresponding metadata.

    TODO: 
        Add the ability for user to delete unwanted files, and have an autoupdating metadata when called
        List the dataset length (in seconds)

    Parameters:
        threshold (float): Threshold value used to filter audio data.

    Returns:
        None
    """
    def __init__(self, segment_dir, sliced_aud_dir, output_dir, threshold=2.5):
        self.Segment_Dir = segment_dir
        self.Sliced_Aud_Dir = sliced_aud_dir
        self.Out_Dir = output_dir
        self.Threshold = threshold
        self.Dataset = []


    # Create directories for storing processed data

    
    def create_metadata(self, file_path: str, thres: float):
        """
        Creates metadata for the audio data.

        Parameters:
            file_path (str): Path to audio file.
            thres (float): Threshold value used to filter audio data.

        Returns:
            pd.DataFrame: DataFrame object containing audio metadata.
        """
        thres = -thres
        name = []
        regular = []
        normalized = []
        punct = []
        with open(file_path, "r", encoding='UTF-8') as f:
            next(f)
            for index, line in enumerate(f):
                line = line.split("|")
                values = line[0].split()
                strings = line[1:]
                if float(values[2]) > thres:
                    index = format(index, '04')
                    file_name = file_path.split('/')[-1:]
                    file_name = file_name[0].split('seg')[0]
                    name.append(file_name+index)
                    regular.append(strings[0].strip())
                    normalized.append(strings[1].strip())
                    punct.append(strings[2].strip())

        metadata = {'name': name, 'regular': regular, 'normalized': normalized,
                    'punct': punct}
        df = pd.DataFrame(metadata)
        return df

    def create_dataset(self, metadata: pd.DataFrame):
        """
        Creates a dataset by copying audio files and saving metadata.

        Parameters:
            metadata (pd.DataFrame): Metadata for the audio data.
            dataset_dir (str): Directory where the dataset is to be saved.

        Returns:
            None
        """
        wav_dir = os.path.join(self.Out_Dir, 'wavs')
        try:
            os.mkdir(wav_dir)
        except:
            pass
        metadata.to_csv(os.path.join(self.Out_Dir, "metadata.csv"),
                        index=False, header=False, sep='|')
        for folder in get_files(self.Sliced_Aud_Dir):
            aud_clips_dir = os.path.join(self.Sliced_Aud_Dir, folder)
            destination = shutil.copytree(aud_clips_dir, wav_dir, dirs_exist_ok=True)
    
    def run_dataset_generation(self):
        if os.listdir(self.Out_Dir) != []:
            print("Dataset has already been created! Skipping...")
            return
        for file in get_files(self.Segment_Dir, '.txt'):
            file_dir = os.path.join(self.Segment_Dir, file)
            self.Dataset.append(self.create_metadata(file_dir, self.Threshold))
        metadata = pd.concat(self.Dataset)
        self.create_dataset(metadata)
        print("Dataset has been created!")


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
        self.ProcessText = ProcessText(
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
        self.Dataset = generate_dataset(
            segment_dir = self.Segmented_Dir, 
            sliced_aud_dir = self.Sliced_Audio_Dir,
            output_dir = self.Dataset_Dir,
            threshold = self.Threshold
        )

    def refine_text(self):
        self.Transcription.run_trancription()
        self.ProcessText.run_processing()
        self.Segment.run_segmentation()
        self.Split.run_slicing()
        self.Dataset.run_dataset_generation()


        

import argparse
parser = argparse.ArgumentParser(description='Modify parameters for dataset generation')
parser.add_argument("--aud_dir",
    help="directory for audio (str, required)",
    required=True)
parser.add_argument("--work_dir",
    help="directory for the various stages of generation (str, required)",
    required=True)
parser.add_argument("--model",
    help="name of Nvidia ASR model (str, default: nvidia/stt_en_citrinet_1024_gamma_0_25)", 
    default='nvidia/stt_en_citrinet_1024_gamma_0_25', type=str)
parser.add_argument("--max_length", 
    help="max length in words of each utterence (int, default: 25)", 
    default=25, type=int)
parser.add_argument("--max_duration",
    help="max length of a single audio clip in s (int, default: 40)",
    default=40, type=int)
parser.add_argument("--lang", 
    help="language of the speaker (str, default: en)", 
    default='en', type=str)
parser.add_argument("--window_size", 
    help="window size for ctc segmentation algorithm (int, default: 8000)", 
    default=8000, type=int)
parser.add_argument("--offset", 
    help="offset for audio clips in ms (int, default: 0)", 
    default=0, type=int)
parser.add_argument("--threshold", 
    help="min score of segmentation confidence to split (float, range: 0-10, lower=more selective, default=2.5)", 
    default=2.5, type=float)


args = parser.parse_args()
aud_dir = args.aud_dir
work_dir = args.work_dir

folders = ['transcription', 'processed', 'segments', 'sliced_audio', 'dataset']
create_core_folders(folders, work_dir)

RefineText = RefineText(
    aud_dir=aud_dir, 
    transcription_dir=os.path.join(work_dir, 'transcription'),
    processed_text_dir=os.path.join(work_dir, 'processed'),
    segment_dir=os.path.join(work_dir,'segments'),
    sliced_audio_dir=os.path.join(work_dir,'sliced_audio'),
    dataset_dir=os.path.join(work_dir, 'dataset'),
    model=args.model,
    window_size=args.window_size,
    offset=args.offset,
    max_duration=args.max_duration,
    max_length=args.max_length,
    threshold=args.threshold,
    lang=args.lang
)
RefineText.refine_text()
# transcribe()
# process(args.model, args.max_length, args.lang)
# segment(args.model, args.window_size)
# split(args.offset, args.max_duration, args.threshold)
# generate_dataset(args.threshold)

