import os
def download_videos(playlist_url):
    playlist_url = "https://www.youtube.com/playlist?list=" + playlist_url.split("=")[1]
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

def get_length(list):
    duration = 0
    for timestamps in list:
        duration += timestamps[1]-timestamps[0]
    return duration

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

def get_raw_dir():
    return rawdir

def create_core_folders():
    folders = ['Isolated', 'No_Overlap', 'Removed_Emotion', 'Samples', "Only_Voice", "Verification", "Exported"]
    for folder in folders:
        folderdir = os.path.join(workdir, folder)
        if os.path.exists(folderdir) == False:
            os.makedirs(folderdir)

def create_samples(length):
    from pydub import AudioSegment
    samples_dir = os.path.join(workdir, "Samples")
    rawfiles = get_files(rawdir, ".wav")
    for file in rawfiles:
        raw_data = AudioSegment.from_file(rawdir+"/"+file, format="wav")
        entry = AudioSegment.empty()
        entry+=raw_data[:length *1000]
        nfilename =  os.path.join(samples_dir, file)
        entry.export(nfilename, format="wav")

def concentrate_timestamps(list):
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
    nlist = []
    for stamps in list:
            if stamps[1] - stamps[0] > 1:
                nlist.append([stamps[0], stamps[1]])
    return nlist


def remove_nonspeech(vad_threshold, noise_agressiveness):
    rawdir = get_raw_dir()
    if len(os.listdir(os.path.join(workdir, "Samples"))) > 0:
        rawdir = os.path.join(workdir, "Samples")
    speech_dir = os.path.join(workdir, "Only_Voice")
    raw_files = get_files(rawdir, ".wav")

    if os.listdir(speech_dir) != []:
        print("speech file(s) already found! Skipping...")
        return

    def analyze_vad():
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
    

    def find_timestamps(speech_metrics):
        import statistics
        timestamps = []
        for fileindex, file in enumerate(raw_files):
            nonspeech_timestamps = []
            startpoint = False
            c50_all=[]
            first_iter = True

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


            speech_timestamps = []
            for index, stamps in enumerate(nonspeech_timestamps):
                try:
                    if nonspeech_timestamps[index+1][0] - stamps[1] > 4:
                        stamplist = [stamps[1], nonspeech_timestamps[index+1][0]]
                        speech_timestamps.append(stamplist)
                except: pass
            speech_timestamps = concentrate_timestamps(speech_timestamps)
            timestamps.append(speech_timestamps)
        return timestamps

    def export_voices(timestamps):
        from pydub import AudioSegment
        for index, file in enumerate(raw_files):
            raw = AudioSegment.from_file(f"{rawdir}/{file}", format="wav")
            entry = AudioSegment.empty()
            for stamps in timestamps[index]:
                entry += raw[stamps[0]*1000:stamps[1]*1000]
            try: entry+=raw[timestamps[index][len(timestamps[index])-1][1]]
            except: pass
            fentry = entry.export(f"{speech_dir}/{file}", format='wav')

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

    def analyze_overlap():
        
        from pyannote.audio import Pipeline
        pipeline = Pipeline.from_pretrained("pyannote/overlapped-speech-detection",
                                            use_auth_token=True)

        overlap_timeline = []
        for file in speech_files:
            dia = pipeline(os.path.join(speech_dir, file))
            overlap_timeline.append(dia)
            print(f"Analyzed {file} for overlap")
        return overlap_timeline

    def overlap_timestamps(overlap):
        overlap_timestamps = []
        for i in range(len(overlap)):
            timestamps = []
            for speech_turn, track, speaker in overlap[i].itertracks(yield_label=True):
                timestamps.append([speech_turn.start, speech_turn.end])
                #print(f"{speech_files[i]} {speech_turn.start:4.1f} {speech_turn.end:4.1f} {speaker}")
            timestamps = concentrate_timestamps(timestamps)
            overlap_timestamps.append(timestamps)
        return overlap_timestamps
    
    def export_no_overlap(all_timestamps):
        from pydub import AudioSegment
        for index, file in enumerate(speech_files):
            raw_data = AudioSegment.from_file(os.path.join(speech_dir, file), format="wav")
            entry = AudioSegment.empty()

            try:
                entry+=raw_data[:all_timestamps[index][0][0]*1000]
            except: entry+=raw_data

            for timestampindex, timestamps in enumerate(all_timestamps[index]):
                if len(timestamps) == 1:
                    print([all_timestamps[index][0][0], all_timestamps[index][0][1]])
                    entry += raw_data[:all_timestamps[index][0][0]*1000]
                    entry += raw_data[all_timestamps[index][0][1]*1000:]
                else:
                    entry += raw_data[all_timestamps[index][timestampindex][0]*1000:all_timestamps[index][timestampindex-1][1]*1000]
            try:
                entry+=raw_data[all_timestamps[index][len(all_timestamps[index])-1][1]*1000:]
            except: pass
            nentry = entry.export(f"{overlap_dir}/{file}", format='wav')

    overlaps = analyze_overlap()
    timestamps = overlap_timestamps(overlaps)
    export_no_overlap(timestamps)
    print("exported non overlapped files!")


def isolate_speaker(verification_threshold):
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

    def find_speakers(files):
        from pyannote.audio import Pipeline
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@develop", use_auth_token=True)
        speakers = []
        for file in files:
            dia = pipeline(os.path.join(overlap_dir, file))
            speakers.append(dia)
            print(f"Seperated speakers for {file}")
        return speakers
    
    def find_number_speakers(track, index):
        speakers = []
        for speech_turn, track, speaker in track.itertracks(yield_label=True):
            if speaker not in speakers:
                speakers.append(speaker)
        print(f"File {overlap_files[index]} has {len(speakers)} speaker(s)")
        return speakers
    
    def find_speakers_timestamps(file, speakers):
        timestamps = [ [] for i in range(len(speakers)) ]
        for speech_turn, track, speaker in file.itertracks(yield_label=True):
            speaker = speaker.split("_")[1]
            speaker = int(speaker)
            #print(f"Speaker {speaker} has spoken between {speech_turn.start} and {speech_turn.end}")
            timestamps[speaker].append([speech_turn.start, speech_turn.end])
        for index, speaker in enumerate(timestamps):
            timestamps[index] = remove_short_timestamps(speaker)
        return timestamps
            
    def seperate_speakers(speakers):
        from pydub import AudioSegment
        for fileindex, tracks in enumerate(speakers, -1):
            speakers = find_number_speakers(tracks, fileindex)
            speaker_timestamps = find_speakers_timestamps(tracks, speakers)
            for index, timestamps in enumerate(speaker_timestamps):
                raw_data = AudioSegment.from_file(os.path.join(overlap_dir, overlap_files[fileindex]), format="wav")
                entry = AudioSegment.empty()
                for start, stop in timestamps:
                    entry += raw_data[start*1000:stop*1000]
                foldername = overlap_files[fileindex].split(".")[0]
                dir = os.path.join(speaker_dir, foldername)
                if os.path.exists(dir) == False:
                    os.mkdir(dir)
                #print(f"{speakers[index]} has been seperated from {overlap_files[fileindex]}")
                fentry = entry.export(os.path.join(speaker_dir, foldername, speakers[index]+".wav"), format="wav")


    def inference_folder(dir, target_speaker):
        from scipy import spatial
        folder_dir = os.path.join(speaker_dir, dir)
        verfified = []
        for file in os.listdir(folder_dir):
            try:
                speakeremb = inference(os.path.join(folder_dir, file))
                distance = 1 - spatial.distance.cosine(speakeremb, target_speaker)
                if distance > verification_threshold:
                    verfified.append(file)
            except: pass
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


def remove_emotion(emotional_threshold):
    import crepe
    from scipy.io import wavfile
    import statistics

    isolated_dir = os.path.join(workdir, "Isolated")
    woemotion_dir = os.path.join(workdir, "Removed_Emotion")
    isolated_folders = get_files(isolated_dir)
    
    def determine_emotional_state(times, frequencies):
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
        from pydub import AudioSegment
        file_dir = os.path.join(isolated_dir, folder_dir, file)
        raw = AudioSegment.from_file(file_dir, format="wav")
        entry = AudioSegment.empty()
        entry += raw[:timestamps[0][0]*1000]
        for stamps in range(len(timestamps)):
            entry += raw[timestamps[stamps][0]*1000:timestamps[stamps][1]*1000]
        fentry = entry.export(os.path.join(woemotion_dir, folder_dir, file), format='wav')

    def analyze_emotion_folder(folder_name):
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
    

def export_audio(sample_rate):
    woemotion_dir = os.path.join(workdir, "Removed_Emotion")
    export_dir = os.path.join(workdir, "Exported")
    woemotion_folders = get_files(woemotion_dir, ".wav")

    from pydub import AudioSegment

    def format_audio(file_name, folder_dir):
        file_dir = os.path.join(woemotion_dir, folder_dir, file_name)
        raw = AudioSegment.from_file(file_dir, format="wav")
        raw.set_channels(0)
        raw.set_frame_rate(sample_rate)

    # def format_folder(folder_name):
    
    # # if os.listdir(export_dir) != []:
    # #     print("All Done!")
    # #     return
    # # for folder in woemotion_folders:
    # #     analyze_emotion_folder(folder)
    # #     print(f"Removed emotional speech from {folder}")




import argparse
parser = argparse.ArgumentParser(description='Modify thresholds for voice data refinement')
parser.add_argument("--rawdir", help="Location for unfiltered audio", required=True)
parser.add_argument("--workdir", help="Location for the various stages of refinement", required=True)
# parser.add_argument("--onset", help="Onset speaker activation threshold 0.0-1.5", default=0.4)
parser.add_argument("--emotional_threshold", help="Change the shift in pitch needed to be considered emotional value 0.0-2.0", default=1.2, type=float)
parser.add_argument("--speaker_threshold", help="Threshold of how dissimilar two speakers must be to be seperated, value 0.0-1.5", default=0.75, type=float)
parser.add_argument("--verification_threshold", help="How similar two speakers must be to be considered the same person, value 0.0-1.0", default=0.90, type=float)
parser.add_argument("--playlist_url", help="YouTube video playlist to download")
parser.add_argument("--vad_threshold", help="Higher the number the more selective the VAD model is (0-100)", default=75, type=int)
parser.add_argument("--noise_agressiveness", help="Higher the number the less sensitive to background noise (0.0-1.5)", default=0.75, type=float)
parser.add_argument("--samples_length", help="Length of each sample (in seconds)", type=int, default=None)


args = parser.parse_args()
rawdir = args.rawdir
workdir = args.workdir


SPEAKER_DIARIZATION = {
            # onset/offset activation thresholds
            "onset": args.speaker_threshold, "offset": args.speaker_threshold, #lower makes the model more sensitve to speaker change and/or overlap
            # remove speech regions shorter than that many seconds.
            "min_duration_on": 1.0,
            # fill non-speech regions shorter than that many seconds.
            "min_duration_off": 0.0
}

if args.playlist_url is not None:
    download_videos(args.playlist_url)
create_core_folders()
if args.samples_length!=None:
    create_samples(args.samples_length)

remove_nonspeech(args.vad_threshold, args.noise_agressiveness)
remove_overlap()
isolate_speaker(args.verification_threshold)
remove_emotion(args.emotional_threshold)