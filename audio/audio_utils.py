from pydub import AudioSegment
import os

def download_videos(playlist_url: str, out_dir):
    '''This function downloads videos from a YouTube playlist URL using the 
       "yt_dlp" library and saves them in the .wav format. 

        Inputs:
        - playlist_url: a string representing the URL of the YouTube playlist
        - out_dir: the directory for audio to be downloaded (not needed if using toolkit)
        
        Outputs:
        - None, but audio files are saved to disk in the .wav format.
        
        Dependencies:
        - "yt_dlp" library
        - "os" library'''

    import yt_dlp

    ydl_opts = {
        'format': 'wav/bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
        }],
        'outtmpl':out_dir + '/%(title)s.%(ext)s',
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        error_code = ydl.download(playlist_url)

    for count, filename in enumerate(os.listdir(out_dir)):
            dst = f"DATA{str(count)}.wav"
            src =f"{out_dir}/{filename}"
            dst =f"{out_dir}/{dst}"
            os.rename(src, dst)

def split_files(folder: str, dir: str, duration: int):
    '''This function splits audio files in the .wav format located in the
       specified folder and saves the clips in the same folder. 

        Inputs:
        - folder: a string representing the name of the folder containing the audio files.
        - dir: a string representing the directory path containing the folder.
        - duration: ms of the duration of sample clips.
        
        Outputs:
        - None, but audio clips are saved to disk in the .wav format.
        
        Dependencies:
        - "os" library
        - "pydub" library and its component "AudioSegment"'''
        
    folder_dir = os.path.join(dir, folder)
    for file in get_files(folder_dir):
        file_dir = os.path.join(dir, folder_dir, file)
        print(file_dir)
        raw = AudioSegment.from_file(file_dir, format="wav")
        raw = raw[::duration]
        for index, clip in raw:
            clip_dir = os.path.join(folder_dir, file.split(".")[0], file.split(".")[0]+str(index)+".wav")
            clip = clip.export(clip_dir, format="wav")

def get_files(dir:str, ext=None) -> list:
    '''This function returns a list of files in a specified directory with a specified file extension. 

    Inputs:
    - dir: a string representing the directory path containing the files.
    - ext (optional): a string representing the file extension to filter the files. 
      If not specified, all files will be returned.
    
    Outputs:
    - A list of sorted filenames.
    
    Dependencies:
    - "os" library
    - "natsort" library'''

    import natsort
    files = []
    for file in os.listdir(dir):
        if ext!=None:
            if file.endswith(ext):
                files.append(file)
        else: files.append(file)
    files = natsort.natsorted(files)
    return files


def create_core_folders(folders: list, workdir: str):
    for folder in folders:
        folderdir = os.path.join(workdir, folder)
        if os.path.exists(folderdir) == False:
            os.makedirs(folderdir)

def get_length(list):
    """
    This function calculates the total duration of a list of time intervals.

    Parameters:
        list (list): A list of time intervals, represented as tuples of start and end times.

    Returns:
        duration (int): The total duration of the time intervals in seconds.

    Example:
        get_length([(0, 30), (40, 50), (60, 70)])
        Returns:
        60
    """
    duration = 0
    for timestamps in list:
        duration += timestamps[1]-timestamps[0]
    return duration


def create_samples(length:int, input_dir, output_dir):    
    '''This function creates audio samples of a specified length from audio files
       in the .wav format located in a specified raw directory.

        Inputs:
        - length: an integer representing the length in seconds of the samples to be created.
        - input_dir: folder where raw wav files are located (for direct method calling)
        - samples_dir: location for output sample wav files (for direct method calling)
        
        Outputs:
        - None, but audio samples are saved to disk in the .wav format.
        
        Dependencies:
        - "os" library
        - "pydub" library and its component "AudioSegment"'''
    rawfiles = get_files(input_dir, ".wav")
    for file in rawfiles:
        raw_data = AudioSegment.from_file(input_dir+"/"+file, format="wav")
        entry = AudioSegment.empty()
        entry+=raw_data[:length *1000]
        nfilename =  os.path.join(output_dir, file)
        entry = entry.export(nfilename, format="wav")


def concentrate_timestamps(list:list, min_duration) -> list:
    '''This function takes in a list of timestamps and returns a condensed list of
       timestamps where timestamps are merged that are close to eachother.

        Inputs:
        - list: a list of timestamp tuples or a list of single timestamps.
        - min_duration: an integer representing the minimum duration between timestamps
          to be combined.
        
        Outputs:
        - A list of condensed timestamps where timestamps that are within
          {min_duration} of eachother have been merged into a single entry.
        
        Dependencies:
        - None'''

    try:
        destination = [list[0]] # start with one period already in the output
    except: return list
    for src in list[1:]: # skip the first period because it's already there
        try:
            src_start, src_end = src
        except: return destination
        current = destination[-1]
        current_start, current_end = current
        if src_start - current_end < min_duration: 
            current[1] = src_end
        else:
            destination.append(src)
    return destination


def remove_short_timestamps(list, min_duration):
    """
    Removes timestamps that are too short from a list of timestamps.

    Parameters:
    list (list): List of timestamps. Each timestamp is a list containing
                 the start and end time of a period.

    Returns:
    list: List of timestamps with short timestamps removed.
    """
    nlist = []
    for stamps in list:
            if stamps[1] - stamps[0] > min_duration:
                nlist.append([stamps[0], stamps[1]])
    return nlist