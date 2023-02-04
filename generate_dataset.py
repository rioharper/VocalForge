import os

def get_files(dir, ext=None):
    import natsort
    files = []
    for file in os.listdir(dir):
        if ext!=None:
            if file.endswith(ext):
                files.append(file)
        else: files.append(file)
    files = natsort.natsorted(files)
    return files

def create_core_folders():
    folders = ['transcription', 'processed', 'segments', 'sliced_audio', 'dataset']
    for folder in folders:
        folderdir = os.path.join(work_dir, folder)
        if os.path.exists(folderdir) == False:
            os.makedirs(folderdir)


def transcribe():
    """
    Transcribe audio files in a directory using a pre-trained ASR model.
    
    Returns:
        None
        
    Note:
        The function prints the names of the transcribed files and skips the ones 
        that have already been transcribed. If a file is corrupted, the function 
        prints "file corrupted! skipping...".
    """
    from whisper import load_model
    
    # create a directory for transcriptions
    transcription_dir = os.path.join(work_dir, "transcription")
    
    # check if there are already transcribed files
    if os.listdir(transcription_dir) != []:
        print("folder(s) have already been transcribed! Skipping...")
        return
    
    # load ASR model
    model = load_model("large")
    
    def transcribe_folder(folder: str):
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
        for file in get_files(folder):
            try:
                # join the audio file directory
                aud_file_dir = os.path.join(aud_dir, file)
                
                # join the text file directory
                text_file_dir = os.path.join(transcription_dir, file.split(".")[0]+".txt")
                
                # check if the text file already exists
                if not os.path.exists(text_file_dir):
                    # transcribe the audio file
                    result = model.transcribe(aud_file_dir)
                    
                    # write the transcription to the text file
                    with open(text_file_dir, 'w', encoding="UTF-8") as f:
                        f.write(result['text'].strip())
                        
                    # print the transcribed file name
                    print(f"transcribed {file}")
            except:
                # print an error message if the audio file is corrupted
                print("file corrupted! skipping...")
    
    # transcribe all audio files in the `aud_dir` folder
    transcribe_folder(aud_dir)



def process(model:str, length:int, lang:str):
    """
    This function prepares the audio and text files for language modeling.
    
    Args:
    1. model (str): Model name to be used for processing the text.
    2. length (int): The max length of each sentence in the output file.
    3. lang (str): Language of the input audio and text files.
    
    Returns:
    None
    """
    # Prepare directory paths
    transcription_dir = os.path.join(work_dir, "transcription")
    processed_dir = os.path.join(work_dir, "processed")
    transcripted_folders = get_files(transcription_dir)

    # Check if the folders have already been processed
    if os.listdir(processed_dir) != []:
        print("folder(s) have already been processed! Skipping...")
        return

    # Import required modules
    import process_text
    from nemo.collections.asr.models import ASRModel
    from pydub import AudioSegment

    # Get configuration of the ASR model
    cfg = ASRModel.from_pretrained(model_name=model, return_config=True)  

    def prepare_audio(audio_dir:str, new_dir:str):
        """
        This function prepares the audio file for language modeling.
        
        Args:
        1. audio_dir (str): Directory path of the raw audio file.
        2. new_dir (str): Directory path of the processed audio file.
        
        Returns:
        None
        """
        # Load audio and export as WAV format
        raw = AudioSegment.from_file(audio_dir, format="wav")
        raw = raw.export(new_dir, format='wav')
    
    def prepare_text(textfile_dir:str):
        """
        This function prepares the text file for language modeling.
        
        Args:
        1. textfile_dir (str): Directory path of the text file.
        
        Returns:
        sentences (str): Processed text sentences.
        """
        # Format and normalize the text
        transcript = process_text.format_text(textfile_dir, lang)
        sentences = process_text.split_text(transcript, lang, cfg.decoder.vocabulary, length, additional_split_symbols=None)
        return process_text.normalize_text(sentences, lang, cfg.decoder.vocabulary)

    def write_file(file_num:str, text_type:str, sentences:str):
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
        file_dir = os.path.join(processed_dir, file_num, file_name)
        print(file_dir)
        with open(file_dir+'.txt', 'w', encoding='UTF-8') as f:
            for sentence in sentences:
                sentence = sentence.strip()
                f.write(sentence+"\n")

    def prepare_folder(folder):
        for file in get_files(transcription_dir, 'txt'):
            textfile_dir = os.path.join(transcription_dir, file)
            audfile_dir = os.path.join(aud_dir, file.replace('.txt', '.wav'))
            folder_dir = os.path.join(processed_dir, file.replace('.txt',''))
            sentence_types = prepare_text(textfile_dir)
            try: os.mkdir(folder_dir)
            except: pass
            for name, sentences in sentence_types.items():
                write_file(file.replace('.txt', ''), name, sentences.splitlines())
            prepare_audio(audfile_dir, os.path.join(folder_dir, file.replace('.txt', '.wav')))
    
    for folder in transcripted_folders:
            prepare_folder(os.path.join(transcription_dir, folder))


def segment(model: str, window_size: int):
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
    processed_dir = os.path.join(work_dir, "processed")
    segmented_dir = os.path.join(work_dir, 'segments')
    if os.listdir(segmented_dir) != []:
        print("folder(s) have already found its split timestamps! Skipping...")
        return
    import statistics
    from ctc import ctc
    import nemo.collections.asr as nemo_asr
    model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model)
    loss = []

    def segment_folder(folder):
        """
        This function segments audio files in a folder into multiple parts based on
        the timestamps obtained using a model.

        Parameters:
        folder (str): Folder containing audio files.

        Returns:
        float: Mean loss obtained from the model.
        """
        processed_folder_dir = os.path.join(processed_dir, folder)
        loss_folder = []
        for aud_file in get_files(processed_folder_dir, '.wav'):
            aud_path = os.path.join(processed_folder_dir, aud_file)
            outfile = segmented_dir + '/' + aud_file.replace('.wav', '_segmented.txt')
            loss = ctc(model, aud_path, outfile, window_size)
            loss_folder.append(loss)
        try:
            return statistics.mean(loss_folder)
        except: return None

    for folder in get_files(processed_dir):
        value = segment_folder(folder)
        if value != None: loss.append(value)
        print(f"Found timestamps for {folder}")
    print(f"median loss: {statistics.median(loss)}")

    

def split(offset: int, max_duration: int, threshold: float) -> None:
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
    sliced_audio_dir = os.path.join(work_dir, "sliced_audio")
    segmented_dir = os.path.join(work_dir, 'segments')
    if os.listdir(sliced_audio_dir) != []:
        print("audio has already been sliced! Skipping...")
        return
    from utils import process_alignment

    threshold = -threshold

    for file in get_files(segmented_dir):
        clips_folder_dir = os.path.join(sliced_audio_dir, file.split('_segmented')[0])
        os.mkdir(clips_folder_dir)
        file_dir = os.path.join(segmented_dir, file)
        process_alignment(file_dir, clips_folder_dir, offset, max_duration, threshold)
        print(f"clips exported for {file}")




def generate_dataset(threshold: float) -> None:
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
    import pandas as pd
    import os
    import shutil

    # Create directories for storing processed data
    segmented_dir = os.path.join(work_dir, 'segments')
    sliced_audio_dir = os.path.join(work_dir, "sliced_audio")
    dataset_dir = os.path.join(work_dir, "dataset")

    if os.listdir(dataset_dir) != []:
        print("Dataset has already been created! Skipping...")
        return

    def create_metadata(file_path: str, thres: float):
        """
        Creates metadata for the audio data.

        Parameters:
            file_path (str): Path of the file containing audio data.
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

    def create_dataset(metadata: pd.DataFrame, dataset_dir: str):
        """
        Creates a dataset by copying audio files and saving metadata.

        Parameters:
            metadata (pd.DataFrame): Metadata for the audio data.
            dataset_dir (str): Directory where the dataset is to be saved.

        Returns:
            None
        """
        wav_dir = os.path.join(dataset_dir, 'wavs')
        try:
            os.mkdir(dataset_dir)
        except:
            pass
        metadata.to_csv(os.path.join(dataset_dir, "metadata.csv"),
                        index=False, header=False, sep='|')
        for folder in get_files(sliced_audio_dir):
            aud_clips_dir = os.path.join(sliced_audio_dir, folder)
            destination = shutil.copytree(aud_clips_dir, wav_dir, dirs_exist_ok=True)
    
    datasets = []
    for file in get_files(segmented_dir):
        file_dir = os.path.join(segmented_dir, file)
        datasets.append(create_metadata(file_dir, threshold))
    metadata = pd.concat(datasets)
    create_dataset(metadata, dataset_dir)
    print("Dataset has been created!")



import argparse
parser = argparse.ArgumentParser(description='Modify thresholds for voice data refinement')
parser.add_argument("--aud_dir", help="Location for unfiltered audio", required=True)
parser.add_argument("--work_dir", help="Location for the various stages of refinement", required=True)
parser.add_argument("--model", help="Choose an Nvidia ASR model", default='nvidia/stt_en_citrinet_1024_gamma_0_25', type=str)
parser.add_argument("--max_length", help="Max length in words of each entry", default=25, type=int)
parser.add_argument("--lang", help="language of the speaker", default='en', type=str)
parser.add_argument("--window_size", help="Window size for ctc segmentation algorithm", default=8000, type=int)
parser.add_argument("--offset", help="delay for clips in ms", default=0, type=int)
parser.add_argument("--max_duration", help="max length of a single audio clip in s", default=40, type=int)
parser.add_argument("--threshold", help="min score of segmentation confidence to split (0-10, lower=more selective)", default=2.5, type=float)


args = parser.parse_args()
aud_dir = args.aud_dir
work_dir = args.work_dir


create_core_folders()
transcribe()
process(args.model, args.max_length, args.lang)
segment(args.model, args.window_size)
split(args.offset, args.max_duration, args.threshold)
generate_dataset(args.threshold)

