# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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


def create_core_folders(folders: list, workdir: str):
    for folder in folders:
        folderdir = os.path.join(workdir, folder)
        if os.path.exists(folderdir) == False:
            os.makedirs(folderdir)


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

import logging
import logging.handlers
import math
import os
from pathlib import PosixPath
from typing import List, Tuple, Union
import statistics
import ctc_segmentation as cs
import numpy as np
from tqdm import tqdm
import scipy.io.wavfile as wav

from nemo.collections.common.tokenizers.sentencepiece_tokenizer import SentencePieceTokenizer


def get_segments(
    log_probs: np.ndarray,
    path_wav: Union[PosixPath, str],
    transcript_file: Union[PosixPath, str],
    output_file: str,
    vocabulary: List[str],
    tokenizer: SentencePieceTokenizer,
    bpe_model: bool,
    index_duration: float,
    window_size: int = 8000
) -> None:
    """
    Segments the audio into segments and saves segments timings to a file
    Args:
        log_probs: Log probabilities for the original audio from an ASR model, shape T * |vocabulary|.
                   values for blank should be at position 0
        path_wav: path to the audio .wav file
        transcript_file: path to
        output_file: path to the file to save timings for segments
        vocabulary: vocabulary used to train the ASR model, note blank is at position len(vocabulary) - 1
        tokenizer: ASR model tokenizer (for BPE models, None for char-based models)
        bpe_model: Indicates whether the model uses BPE
        window_size: the length of each utterance (in terms of frames of the CTC outputs) fits into that window.
        index_duration: corresponding time duration of one CTC output index (in seconds)
    """

    with open(transcript_file, "r") as f:
        text = f.readlines()
        text = [t.strip() for t in text if t.strip()]

    # add corresponding original text without pre-processing
    transcript_file_no_preprocessing = transcript_file.replace(".txt", "_with_punct.txt")
    if not os.path.exists(transcript_file_no_preprocessing):
        raise ValueError(f"{transcript_file_no_preprocessing} not found.")

    with open(transcript_file_no_preprocessing, "r") as f:
        text_no_preprocessing = f.readlines()
        text_no_preprocessing = [t.strip() for t in text_no_preprocessing if t.strip()]

    # add corresponding normalized original text
    transcript_file_normalized = transcript_file.replace(".txt", "_with_punct_normalized.txt")
    if not os.path.exists(transcript_file_normalized):
        raise ValueError(f"{transcript_file_normalized} not found.")

    with open(transcript_file_normalized, "r") as f:
        text_normalized = f.readlines()
        text_normalized = [t.strip() for t in text_normalized if t.strip()]

    # if len(text_no_preprocessing) != len(text):
    #     raise ValueError(f"{transcript_file} and {transcript_file_no_preprocessing} do not match")

    # if len(text_normalized) != len(text):
    #     raise ValueError(f"{transcript_file} and {transcript_file_normalized} do not match")

    config = cs.CtcSegmentationParameters()
    config.char_list = vocabulary
    config.min_window_size = window_size
    config.index_duration = index_duration

    if bpe_model:
        ground_truth_mat, utt_begin_indices = _prepare_tokenized_text_for_bpe_model(text, tokenizer, vocabulary, 0)
    else:
        config.excluded_characters = ".,-?!:»«;'›‹()"
        config.blank = vocabulary.index(" ")
        ground_truth_mat, utt_begin_indices = cs.prepare_text(config, text)

    _print(ground_truth_mat, config.char_list)

    # set this after text prepare_text()
    config.blank = 0
    logging.debug(f"Syncing {transcript_file}")
    logging.debug(
        f"Audio length {os.path.basename(path_wav)}: {log_probs.shape[0]}. "
        f"Text length {os.path.basename(transcript_file)}: {len(ground_truth_mat)}"
    )

    timings, char_probs, char_list = cs.ctc_segmentation(config, log_probs, ground_truth_mat)
    _print(ground_truth_mat, vocabulary)
    segments = determine_utterance_segments(config, utt_begin_indices, char_probs, timings, text, char_list)

    write_output(output_file, path_wav, segments, text, text_no_preprocessing, text_normalized)
    values = []
    for segment in segments:
        values.append(segment[2])
    print(f"average segmentation loss: {statistics.mean(values)}")
    for i, (word, segment) in enumerate(zip(text, segments)):
        if i < 5:
            logging.debug(f"{segment[0]:.2f} {segment[1]:.2f} {segment[2]:3.4f} {word}")
    logging.info(f"segmentation of {transcript_file} complete.")
    return statistics.mean(values)



def process_alignment(alignment_file: str, clips_dir: str, offset, max_duration, threshold):
    """ Cut original audio file into audio segments based on alignment_file
    Args:
        alignment_file: path to the file with segmented text and corresponding time stamps.
            The first line of the file contains the path to the original audio file
        clips_dir: path to a directory to save audio clips
        args: main script args
    """
    if not os.path.exists(alignment_file):
        raise ValueError(f"{alignment_file} not found")

    base_name = os.path.basename(alignment_file).replace("_segmented.txt", "")

    # read the segments, note the first line contains the path to the original audio
    segments = []
    with open(alignment_file, "r") as f:
        for line in f:
            line = line.split("|")
            # read audio file name from the first line
            if len(line) == 1:
                audio_file = line[0].strip()
                continue
            line = line[0].split()
            segments.append((float(line[0]) + offset / 1000, float(line[1]) + offset / 1000, float(line[2])))

    # cut the audio into segments and save the final manifests at output_dir
    sampling_rate, signal = wav.read(audio_file)
    original_dur = len(signal) / sampling_rate

    low_score_dur = 0
    high_score_dur = 0
    for i, (st, end, score) in enumerate(segments):
        segment = signal[round(st * sampling_rate) : round(end * sampling_rate)]
        duration = len(segment) / sampling_rate
        if duration > max_duration:
            continue
        if duration > 0:
            if score >= threshold:
                high_score_dur += duration
                audio_filepath = os.path.join(clips_dir, f"{base_name}_{i:04}.wav")
                wav.write(audio_filepath, sampling_rate, segment)
            else:
                low_score_dur += duration
 
    # keep track of duration of the deleted segments
    del_duration = 0
    begin = 0

    for i, (st, end, _) in enumerate(segments):
        if st - begin > 0.01:
            segment = signal[int(begin * sampling_rate) : int(st * sampling_rate)]
            duration = len(segment) / sampling_rate
            del_duration += duration
        begin = end

    segment = signal[int(begin * sampling_rate) :]
    duration = len(segment) / sampling_rate
    del_duration += duration

    print(f"Original duration  : {round(original_dur / 60)}min")
    print(f"High score segments: {round(high_score_dur / 60)}min ({round(high_score_dur/original_dur*100)}%)")
    print(f"Low score segments : {round(low_score_dur / 60)}min ({round(low_score_dur/original_dur*100)}%)")

def _prepare_tokenized_text_for_bpe_model(text: List[str], tokenizer, vocabulary: List[str], blank_idx: int = 0):
    """ Creates a transition matrix for BPE-based models"""
    space_idx = vocabulary.index("▁")
    ground_truth_mat = [[-1, -1]]
    utt_begin_indices = []
    for uttr in text:
        ground_truth_mat += [[blank_idx, space_idx]]
        utt_begin_indices.append(len(ground_truth_mat))
        token_ids = tokenizer.text_to_ids(uttr)
        # blank token is moved from the last to the first (0) position in the vocabulary
        token_ids = [idx + 1 for idx in token_ids]
        ground_truth_mat += [[t, -1] for t in token_ids]

    utt_begin_indices.append(len(ground_truth_mat))
    ground_truth_mat += [[blank_idx, space_idx]]
    ground_truth_mat = np.array(ground_truth_mat, np.int64)
    return ground_truth_mat, utt_begin_indices


def _print(ground_truth_mat, vocabulary, limit=20):
    """Prints transition matrix"""
    chars = []
    for row in ground_truth_mat:
        chars.append([])
        for ch_id in row:
            if ch_id != -1:
                chars[-1].append(vocabulary[int(ch_id)])

    for x in chars[:limit]:
        logging.debug(x)


def _get_blank_spans(char_list, blank="ε"):
    """
    Returns a list of tuples:
        (start index, end index (exclusive), count)
    ignores blank symbols at the beginning and end of the char_list
    since they're not suitable for split in between
    """
    blanks = []
    start = None
    end = None
    for i, ch in enumerate(char_list):
        if ch == blank:
            if start is None:
                start, end = i, i
            else:
                end = i
        else:
            if start is not None:
                # ignore blank tokens at the beginning
                if start > 0:
                    end += 1
                    blanks.append((start, end, end - start))
                start = None
                end = None
    return blanks


def _compute_time(index, align_type, timings):
    """Compute start and end time of utterance.
    Adapted from https://github.com/lumaku/ctc-segmentation
    Args:
        index:  frame index value
        align_type:  one of ["begin", "end"]
    Return:
        start/end time of utterance in seconds
    """
    middle = (timings[index] + timings[index - 1]) / 2
    if align_type == "begin":
        return max(timings[index + 1] - 0.5, middle)
    elif align_type == "end":
        return min(timings[index - 1] + 0.5, middle)


def determine_utterance_segments(config, utt_begin_indices, char_probs, timings, text, char_list):
    """Utterance-wise alignments from char-wise alignments.
    Adapted from https://github.com/lumaku/ctc-segmentation
    Args:
        config: an instance of CtcSegmentationParameters
        utt_begin_indices: list of time indices of utterance start
        char_probs:  character positioned probabilities obtained from backtracking
        timings: mapping of time indices to seconds
        text: list of utterances
    Return:
        segments, a list of: utterance start and end [s], and its confidence score
    """
    segments = []
    min_prob = np.float64(-10000000000.0)
    for i in tqdm(range(len(text))):
        start = _compute_time(utt_begin_indices[i], "begin", timings)
        end = _compute_time(utt_begin_indices[i + 1], "end", timings)

        start_t = start / config.index_duration_in_seconds
        start_t_floor = math.floor(start_t)

        # look for the left most blank symbol and split in the middle to fix start utterance segmentation
        if char_list[start_t_floor] == config.char_list[config.blank]:
            start_blank = None
            j = start_t_floor - 1
            while char_list[j] == config.char_list[config.blank] and j > start_t_floor - 20:
                start_blank = j
                j -= 1
            if start_blank:
                start_t = int(round(start_blank + (start_t_floor - start_blank) / 2))
            else:
                start_t = start_t_floor
            start = start_t * config.index_duration_in_seconds

        else:
            start_t = int(round(start_t))

        end_t = int(round(end / config.index_duration_in_seconds))

        # Compute confidence score by using the min mean probability after splitting into segments of L frames
        n = config.score_min_mean_over_L
        if end_t <= start_t:
            min_avg = min_prob
        elif end_t - start_t <= n:
            min_avg = char_probs[start_t:end_t].mean()
        else:
            min_avg = np.float64(0.0)
            for t in range(start_t, end_t - n):
                min_avg = min(min_avg, char_probs[t : t + n].mean())
        segments.append((start, end, min_avg))
    return segments


def write_output(
    out_path: str,
    path_wav: str,
    segments: List[Tuple[float]],
    text: str,
    text_no_preprocessing: str,
    text_normalized: str,
):
    """
    Write the segmentation output to a file
    out_path: Path to output file
    path_wav: Path to the original audio file
    segments: Segments include start, end and alignment score
    text: Text used for alignment
    text_no_preprocessing: Reference txt without any pre-processing
    text_normalized: Reference text normalized
    """
    # Uses char-wise alignments to get utterance-wise alignments and writes them into the given file
    with open(str(out_path), "w") as outfile:
        outfile.write(str(path_wav) + "\n")

        for i, segment in enumerate(segments):
            if isinstance(segment, list):
                for j, x in enumerate(segment):
                    start, end, score = x
                    score = -0.2
                    outfile.write(
                        f"{start} {end} {score} | {text[i][j]} | {text_no_preprocessing[i][j]} | {text_normalized[i][j]}\n"
                    )
            else:
                start, end, score = segment
                outfile.write(
                    f"{start} {end} {score} | {text[i]} | {text_no_preprocessing[i]} | {text_normalized[i]}\n"
                )