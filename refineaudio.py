import os
from pydub import AudioSegment

def download_videos(playlist_url: str):
    
    '''This function downloads videos from a YouTube playlist URL using the 
       "yt_dlp" library and saves them in the .wav format. 

        Inputs:
        - playlist_url: a string representing the URL of the YouTube playlist
        
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
        'outtmpl':rawdir + '/%(title)s.%(ext)s',
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        error_code = ydl.download(playlist_url)

    for count, filename in enumerate(os.listdir(rawdir)):
            dst = f"DATA{str(count)}.wav"
            src =f"{rawdir}/{filename}"
            dst =f"{rawdir}/{dst}"
            os.rename(src, dst)


def split_files(folder: str, dir: str):
    '''This function splits audio files in the .wav format located in the
       specified folder and saves the clips in the same folder. 

        Inputs:
        - folder: a string representing the name of the folder containing the audio files.
        - dir: a string representing the directory path containing the folder.
        
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
        raw = raw[::60000]
        for index, clip in raw:
            
            clip_dir = os.path.join(folder_dir, file.split(".")[0], file.split(".")[0]+str(index)+".wav")
            clip = clip.export(clip_dir, format="wav")


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


def get_raw_dir():
    return rawdir


def create_core_folders():
    '''This function creates a set of core folders in a specified working directory. 

        Inputs:
        - None, but the working directory path should be specified globally.
        
        Outputs:
        - None, but a set of folders is created in the working directory if they do not already exist.
        
        Dependencies:
        - "os" library'''

    folders = ['Isolated', 'No_Overlap', 'Removed_Emotion', 'Samples', "Only_Voice", "Verification", "Exported", 'Noise_Removed']
    for folder in folders:
        folderdir = os.path.join(workdir, folder)
        if os.path.exists(folderdir) == False:
            os.makedirs(folderdir)


def create_samples(length:int):    
    '''This function creates audio samples of a specified length from audio files
       in the .wav format located in a specified raw directory.

        Inputs:
        - length: an integer representing the length in seconds of the samples to be created.
        
        Outputs:
        - None, but audio samples are saved to disk in the .wav format.
        
        Dependencies:
        - "os" library
        - "pydub" library and its component "AudioSegment"'''

    samples_dir = os.path.join(workdir, "Samples")
    rawfiles = get_files(rawdir, ".wav")
    for file in rawfiles:
        raw_data = AudioSegment.from_file(rawdir+"/"+file, format="wav")
        entry = AudioSegment.empty()
        entry+=raw_data[:length *1000]
        nfilename =  os.path.join(samples_dir, file)
        entry.export(nfilename, format="wav")


def concentrate_timestamps(list:list) -> list:
    '''This function takes in a list of timestamps and returns a condensed list of
       timestamps where timestamps are merged that are close to eachother.

        Inputs:
        - list: a list of timestamp tuples or a list of single timestamps.
        
        Outputs:
        - A list of condensed timestamps where timestamps that are within
          2000 ms of eachother have been merged into a single entry.
        
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
        if src_start - current_end < 2000: 
            current[1] = src_end
        else:
            destination.append(src)
    return destination


def remove_short_timestamps(list):
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
            if stamps[1] - stamps[0] > 1:
                nlist.append([stamps[0], stamps[1]])
    return nlist



def remove_nonspeech(vad_threshold: float, noise_agressiveness: float):
    rawdir = get_raw_dir()
    if len(os.listdir(os.path.join(workdir, "Samples"))) > 0:
        rawdir = os.path.join(workdir, "Samples")
    speech_dir = os.path.join(workdir, "Only_Voice")
    raw_files = get_files(rawdir, ".wav")

    if os.listdir(speech_dir) != []:
        print("speech file(s) already found! Skipping...")
        return


    def analyze_vad():
        """
        This function analyzes audio files in a folder and performs voice activity detection (VAD)
        on the audio files. It uses the 'pyannote.audio' library's pre-trained 'brouhaha' model for the analysis.

        Parameters:
            rawdir (dir): predefined globally
        Returns:
            speech_metrics (list): List of voice activity detection output for each audio file.
        """
        from pyannote.audio import Inference
        from pyannote.audio import Model
        model = Model.from_pretrained("pyannote/brouhaha", 
                                    use_auth_token=True)
        speech_metrics = []
        inference = Inference(model)

        for file in raw_files:
            output = inference(rawdir+"/"+file)
            speech_metrics.append(output)
            print(f"Analyzed {file} for voice detection")
        return speech_metrics
    

    def find_timestamps(speech_metrics: list) -> list:
        """
        This function processes speech metrics and returns timestamps
        of speech segments in the audio.
        
        Parameters:
        speech_metrics (list): list of speech metrics for each audio file
        
        Returns:
        timestamps (list): list of speech timestamps for each audio file
        """
        import statistics
        
        timestamps = []
        for fileindex, file in enumerate(raw_files):
            nonspeech_timestamps = []
            startpoint = False
            c50_all = []
            first_iter = True
            
            # Calculate median of c50 values for the current audio file
            for frame, (c50) in speech_metrics[fileindex]:
                c50_all.append(c50[2])
            c50_med = float(statistics.median_high(c50_all))
            
            for frame, (vad, snr, c50) in speech_metrics[fileindex]:
                vad = vad *100
                t = frame.middle
                if first_iter:
                    nonspeech_timestamps.append([t])
                    first_iter = False
                if vad < vad_threshold and startpoint == False:
                    nonspeech_timestamps.append([t])
                    startpoint = True
                elif c50_med * noise_agressiveness > c50 and startpoint == False:
                    nonspeech_timestamps.append([t])
                    startpoint = True
                elif c50_med * noise_agressiveness < c50 and startpoint == True and vad > vad_threshold:
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
            speech_timestamps = concentrate_timestamps(speech_timestamps)
            timestamps.append(speech_timestamps)
        
        return timestamps


    def export_voices(timestamps: list):
        """
        Given a list of timestamps for each file, the function exports 
        the speech segments from each raw file to a new file format wav. 
        The new files are saved to a specified directory. 

        Parameters: 
        timestamps (list): A list of timestamps for each file indicating 
        the start and end of speech segments.

        Returns: 
        None
        """
        for index, file in enumerate(raw_files):
            # Load raw audio file
            raw = AudioSegment.from_file(f"{rawdir}/{file}", format="wav")
            entry = AudioSegment.empty()
            # Add speech segments to the new audio file
            for stamps in timestamps[index]:
                entry += raw[stamps[0]*1000:stamps[1]*1000]
            try:
                entry += raw[timestamps[index][len(timestamps[index])-1][1]]
            except:
                pass
            # Check if the new audio file has enough speech segments
            if len(entry) > 1000:
                # Save the new audio file to the specified directory
                fentry = entry.export(f"{speech_dir}/{file}", format='wav')
            else:
                print(f"{file} doesnt have enough clean audio to export")

    speech_metrics = analyze_vad()
    timestamps = find_timestamps(speech_metrics)
    export_voices(timestamps)
    print("exported voice-only files!")

    

def remove_overlap():
    speech_dir = os.path.join(workdir, "Only_Voice")
    overlap_dir = os.path.join(workdir, "No_Overlap")
    speech_files = get_files(speech_dir, ".wav")
    if os.listdir(overlap_dir) != []:
        print("overlap file(s) already found! Skipping...")
        return


    def analyze_overlap() -> list:
        """
        Analyzes overlapping speech in a set of speech audio files.
        
        Returns:
            overlap_timeline (list): A list of overlapping speech timestamps 
                                    for each file.
        """
        from pyannote.audio import Pipeline
        
        # Create a pipeline object using the pre-trained "pyannote/overlapped-speech-detection"
        pipeline = Pipeline.from_pretrained("pyannote/overlapped-speech-detection",
                                            use_auth_token=True)
        overlap_timeline = []
        
        # Loop through each speech file in the speech_files list
        for file in speech_files:
            try:
                # Use the pipeline to analyze the file for overlapping speech
                dia = pipeline(os.path.join(speech_dir, file))
            except:
                # If the pipeline fails to analyze the file, print an error message
                print(f"{file} seems to have no data in it!")
            overlap_timeline.append(dia)
            print(f"Analyzed {file} for overlap")
        
        # Return the overlapping speech timeline
        return overlap_timeline


    def overlap_timestamps(overlap: list) -> list:
        """
        Converts overlap timelines into timestamps of non-overlapping speech turns

        Parameters:
            overlap (list): List of overlap timelines obtained from analyze_overlap function

        Returns:
            overlap_timestamps (list): List of timestamps of non-overlapping speech turns
        """
        overlap_timestamps = []
        for i in range(len(overlap)):
            timestamps = []
            for speech_turn, track, speaker in overlap[i].itertracks(yield_label=True):
                timestamps.append([speech_turn.start, speech_turn.end])
                #print(f"{speech_files[i]} {speech_turn.start:4.1f} {speech_turn.end:4.1f} {speaker}")
            timestamps = concentrate_timestamps(timestamps)
            overlap_timestamps.append(timestamps)
        return overlap_timestamps
    

    def export_no_overlap(all_timestamps: list):
        """
        Exports the audio files with overlapped speech removed.
        
        Parameters:
        all_timestamps (list): List of timestamps of overlapped speech
                                                for each audio file.
        
        Returns:
        None
        """
        for index, file in enumerate(speech_files):
            raw_data = AudioSegment.from_file(os.path.join(speech_dir, file), format="wav")
            entry = AudioSegment.empty()
            
            try:
                entry+=raw_data[:all_timestamps[index][0][0]*1000]
            except:
                entry+=raw_data
            
            for timestampindex, timestamps in enumerate(all_timestamps[index]):
                if len(timestamps) == 1:
                    entry += raw_data[:all_timestamps[index][0][0]*1000]
                    entry += raw_data[all_timestamps[index][0][1]*1000:]
                else:
                    entry += raw_data[all_timestamps[index][timestampindex][0]*1000:all_timestamps[index][timestampindex-1][1]*1000]
            try:
                entry+=raw_data[all_timestamps[index][len(all_timestamps[index])-1][1]*1000:]
            except:
                pass
            nentry = entry.export(f"{overlap_dir}/{file}", format='wav')


    overlaps = analyze_overlap()
    timestamps = overlap_timestamps(overlaps)
    export_no_overlap(timestamps)
    print("exported non overlapped files!")



def isolate_speaker(verification_threshold):
    """Isolates speakers in file, then finds target speaker across all the files
        Output: Verification and Isolated folders
    """
    overlap_dir = os.path.join(workdir, "No_Overlap")
    speaker_dir = os.path.join(workdir, "Verification")
    isolated_dir = os.path.join(workdir, "Isolated")
    overlap_files = get_files(overlap_dir, ".wav")

    from pyannote.audio import Model
    model = Model.from_pretrained("pyannote/embedding", use_auth_token=True)
    from pyannote.audio import Pipeline
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@develop", use_auth_token=True)
    from pyannote.audio import Inference
    inference = Inference(model, window="whole", device="cuda")

    import shutil


    def find_speakers(files: list) -> list:
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
        speakers = []
        for file in files:
            dia = pipeline(os.path.join(overlap_dir, file))
            speakers.append(dia)
            print(f"Seperated speakers for {file}")
        return speakers

    
    def find_number_speakers(track, index: int) -> list:
        """
        Find the number of speakers in a given track of a audio file.

        Parameters:
        track (pyannote.core.Annotation): PyAnnote annotation object representing a speaker track.
        index (int): Index of the current audio file being processed.

        Returns:
        List[str]: A list of unique speaker names in the given track.
        """

        speakers = []
        for speech_turn, track, speaker in track.itertracks(yield_label=True):
            if speaker not in speakers:
                speakers.append(speaker)
        print(f"File {overlap_files[index]} has {len(speakers)} speaker(s)")
        return speakers

    
    def find_speakers_timestamps(file: tuple, speakers: list):
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
            timestamps[index] = remove_short_timestamps(speaker)

        return timestamps

            
    def seperate_speakers(speakers: list) -> None:
        """
        Given a list of speakers' tracks, this function separates individual 
        speakers and saves their speech parts to a directory.

        Parameters:
        speakers (list): list of tracks of multiple speakers
        
        Returns:
        None
        """
        from pydub import AudioSegment
        import os

        for fileindex, tracks in enumerate(speakers, -1):
            speakers = find_number_speakers(tracks, fileindex)
            speaker_timestamps = find_speakers_timestamps(tracks, speakers)
            for index, timestamps in enumerate(speaker_timestamps):
                raw_data = AudioSegment.from_file(os.path.join(overlap_dir,
                                                                overlap_files[fileindex]),
                                                                format="wav")
                entry = AudioSegment.empty()
                for start, stop in timestamps:
                    entry += raw_data[start * 1000: stop * 1000]
                foldername = overlap_files[fileindex].split(".")[0]
                dir = os.path.join(speaker_dir, foldername)
                if os.path.exists(dir) == False:
                    os.mkdir(dir)
                fentry = entry.export(os.path.join(speaker_dir, foldername,
                                                    speakers[index]+".wav"),
                                                    format="wav")


    def inference_folder(dir: str, target_speaker) -> list:
        """
        Infer the speaker of the audio files in a directory and verify the identity against a target speaker.

        Parameters:
        dir (str): The directory containing the audio files to be inferred.
        target_speaker (numpy.ndarray): The target speaker's embedding to verify against.

        Returns:
        List[str]: List of verified audio files in the directory.

        """
        from scipy import spatial
        folder_dir = os.path.join(speaker_dir, dir)
        verfified = []
        for file in os.listdir(folder_dir):
            try:
                speakeremb = inference(os.path.join(folder_dir, file))
                distance = 1 - spatial.distance.cosine(speakeremb, target_speaker)
                if distance > verification_threshold:
                    verfified.append(file)
            except: 
                pass
        return verfified


    def verify_folder(target_speaker):
        folders =  get_files(speaker_dir)
        verified_files = []
        for folder in folders:
            folder = os.path.join(speaker_dir, folder)
            verfified = inference_folder(folder, target_speaker)
            verified_files.append(verfified)
        return verified_files


    def transfer_to_isolated(verified_files):
        folders =  get_files(speaker_dir)
        for folderindex, folder in enumerate(verified_files):
            verified_folder = os.path.join(isolated_dir, folders[folderindex])
            os.mkdir(verified_folder)
            for file in folder:
                verfied_file = os.path.join(speaker_dir, folders[folderindex], file)
                shutil.copyfile(verfied_file, os.path.join(verified_folder, file))

            
    if os.listdir(speaker_dir) == []:
        speakers = find_speakers(overlap_files)
        seperate_speakers(speakers)
        print("Speaker(s) have already been split! Skipping...")
        
    if os.listdir(isolated_dir) == []:
        while True:
            target_speaker =  input("Choose Target Speaker From DATA0 (e.g SPEAKER_00): ")
            #try: 
            speakeremb = inference(os.path.join(speaker_dir, "DATA0", target_speaker+".wav"))
            break
            #except: print("Oops! Seems like you mistyped it, try again.")  
        verified_files = verify_folder(speakeremb)
        transfer_to_isolated(verified_files)
        print("All clips of selected speaker have been verified!")
    else: print("Speaker has already been verified! Skipping...")



"""
Not currently in use due to poor performance
TODO:
    Replace Pitch with either multiple factors (pitch, db, offset, etc) OR
    Create a model to detect the emotional state of speech
"""
def remove_emotion(emotional_threshold):
    import crepe
    from scipy.io import wavfile
    import statistics

    isolated_dir = os.path.join(workdir, "Isolated")
    woemotion_dir = os.path.join(workdir, "Removed_Emotion")
    isolated_folders = get_files(isolated_dir)
    
    def determine_emotional_state(times, frequencies):
        print('determine')
        medianfreq = statistics.median_grouped(frequencies)
        stdevfreq = statistics.stdev(frequencies) * emotional_threshold

        rawtimestamps = [[0]]
        nonemotional = True

        for time, frequency in zip(times, frequencies):
            if frequency > medianfreq + stdevfreq and nonemotional == True:
                rawtimestamps[len(rawtimestamps)-1].append(time)
                nonemotional = False
            elif  frequency < medianfreq + stdevfreq and nonemotional == False:
                rawtimestamps.append([time])
                nonemotional = True
        rawtimestamps = concentrate_timestamps(rawtimestamps)
        return rawtimestamps
    
    def remove_emotional_speech(timestamps, folder_dir, file):
        print('timestamps')
        
        file_dir = os.path.join(isolated_dir, folder_dir, file)
        raw = AudioSegment.from_file(file_dir, format="wav")
        entry = AudioSegment.empty()
        entry += raw[:timestamps[0][0]*1000]
        for stamps in range(len(timestamps)):
            entry += raw[timestamps[stamps][0]*1000:timestamps[stamps][1]*1000]
        fentry = entry.export(os.path.join(woemotion_dir, folder_dir, file), format='wav')

    def analyze_emotion_folder(folder_name):
        print('emotion folder')
        os.mkdir(os.path.join(woemotion_dir, folder_name))
        folder_dir = (os.path.join(isolated_dir, folder_name))
        folder = get_files(folder_dir)
        for file in folder:
            file_dir = os.path.join(folder_dir, file)
            sr, audio = wavfile.read(file_dir)
            times, frequencies , confidence, activation = crepe.predict(audio, sr, viterbi=True, center=False, step_size=100)
            rawtimestamps = (determine_emotional_state(times, frequencies))
            timestamps = concentrate_timestamps(rawtimestamps)
            remove_emotional_speech(timestamps, folder_name, file)
    
    if os.listdir(woemotion_dir) != []:
        print("Emotional speech has already been removed from file(s)! Skipping...")
        return
    for folder in isolated_folders:
        analyze_emotion_folder(folder)
        print(f"Removed emotional speech from {folder}")
    


def export_audio(sample_rate, do_noise_remover, do_normalization):
    woemotion_dir = os.path.join(workdir, "Isolated")
    export_dir = os.path.join(workdir, "Exported")
    noise_removed_dir = os.path.join(workdir, "Noise_Removed")
    woemotion_folders = get_files(woemotion_dir)
    
    from scipy.io import wavfile
    import numpy as np

    def find_all_mean_sd(folders_dir: str) -> tuple:
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
            folder_dir = os.path.join(folders_dir, folder)
            for file in get_files(folder_dir):
                rate, data = wavfile.read(os.path.join(folder_dir, file))
                mean += np.mean(data)
                sd += np.std(data)
                count += 1
        mean /= count
        sd /= count
        return mean, sd


    def normalize(folder, mean, sd):
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
        for file in get_files(folder, '.wav'):
            file_dir = os.path.join(export_dir, file)
            rate, data = wavfile.read(file_dir)
            mean_subtracted = data - mean
            eps = 2**-30
            output = mean_subtracted / (sd + eps)
            normalized_file_dir = os.path.join(normalization_dir, file)
            wavfile.write(normalized_file_dir, rate, output)


    def noise_remove(folder_dir):
        for file in get_files(folder_dir):
            file_dir = os.path.join(folder_dir, file)
            os.system(f"deepFilter -m DeepFilterNet2 {file_dir} --output-dir {noise_removed_dir}")
            os.rename(os.path.join(noise_removed_dir, file.split('.')[0]+"_DeepFilterNet2.wav"), 
                os.path.join(noise_removed_dir, file.replace('_DeepFilterNet2', '')))
        

    def format_audio(folder):
        folder_dir = os.path.join(woemotion_dir, folder)
        combined = AudioSegment.empty()
        if len(get_files(folder_dir, '.wav')) == 0:
            return
        for file in get_files(folder_dir, '.wav'):
            file_dir = os.path.join(woemotion_dir, folder_dir, file)
            raw = AudioSegment.from_file(file_dir, format="wav")
            raw = raw.set_channels(1)
            raw = raw.set_frame_rate(sample_rate)
            combined += raw
        combined = combined.export(os.path.join(export_dir, folder+'.wav'), format='wav')
    
    if os.listdir(export_dir) != []:
        print("file(s) have already been formatted! Skipping...")
        return

    if do_normalization:
        mean, sd = find_all_mean_sd(woemotion_dir)
        try: os.mkdir(os.path.join(workdir, 'Normalized'))
        except: pass
        normalization_dir = os.path.join(workdir, 'Normalized')
    for folder in woemotion_folders:
        format_audio(folder)
        print(f"Formatted Audio for {folder}")
    if do_noise_remover:
        print("Removing Noise...") 
        noise_remove(export_dir)
    if do_normalization:
        print("Normalizing Audio...")
        normalize(export_dir, mean, sd)



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
parser.add_argument("--speaker_threshold",
    help="The lower the value, the more sensitive speaker seperation is (float, default: 0.2)",
    default=0.2, type=float)
parser.add_argument("--verification_threshold",
    help="The higher the value, the more similar two voices must be during voice verification (float, range: 0.0-0.99, default: 0.9)",
    default=0.90, type=float)
parser.add_argument("--playlist_url",
    help="URL to YouTube playlist to be downloaed to raw_dir (str)",
    type=str)
parser.add_argument("--vad_threshold",
    help="The higher the value, the more selective the VAD model will be (int, default: 75)",
    default=75, type=int)
parser.add_argument("--snr_change", 
    help="The lower the value, the more sensitive the model is to changes in SNR, such as laughter or loud noises (float, default: 0.75)", 
    default=0.75, type=float)
parser.add_argument("--samples_length", 
    help="create sample voice clips from raw_dir for testing purposes (in seconds)", 
    type=int, default=None)
parser.add_argument("--do_noise_reduction", 
    help="use deepfilternet 2 to reduce noise in the exported files (bool, default: False)", 
    type=bool, default=False)
parser.add_argument("--do_normalize",
    help="use mean/sd normalization, can be useful for some DL models (bool, default: False)", 
    type=bool, default=False)
#parser.add_argument("--emotional_threshold", help="Threshold to be considered emotional value (0.0-2.0)", default=1.2, type=float)




args = parser.parse_args()
rawdir = args.raw_dir
workdir = args.work_dir


SPEAKER_DIARIZATION = {
            # onset/offset activation thresholds
            "onset": args.speaker_threshold, "offset": args.speaker_threshold, #lower makes the model more sensitve to speaker change and/or overlap
            # remove speech regions shorter than that many seconds.
            "min_duration_on": 1.0,
            # fill non-speech regions shorter than that many seconds.
            "min_duration_off": 0.0
}


from torch import cuda
if cuda.is_available() == False:
    print("CUDA device not found! If you have CUDA intalled, please check if its propery configured")
    print("Program will continue, but at a much slower pace.")
else: print("CUDA device configured correctly!")

if args.playlist_url is not None:
    download_videos(args.playlist_url)
create_core_folders()
if args.samples_length!=None:
    create_samples(args.samples_length)
print(args.do_normalize)

remove_nonspeech(args.vad_threshold, args.snr_change)
remove_overlap()
isolate_speaker(args.verification_threshold)
#remove_emotion(args.emotional_threshold)
export_audio(args.sample_rate, args.do_noise_reduction, args.do_normalize)