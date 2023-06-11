import os
from pathlib import Path
import numpy as np
import scipy.io.wavfile as wav
import torch
from joblib import Parallel, delayed
import nemo.collections.asr as nemo_asr
from tqdm import tqdm
from nemo.collections.common.tokenizers.sentencepiece_tokenizer import SentencePieceTokenizer
import logging
import logging.handlers
import math
from pathlib import PosixPath
from typing import List, Tuple, Union
import statistics
import ctc_segmentation as cs

def ctc(model: str,
        output_dir: str,
        data_dir: str,
        vocabulary: str,
        window_len: int = 8000) -> None:
    """
    Transcribes audio files using Encoder-Decoder with Connectionist Temporal Classification (CTC) model.

    Args:
        model (Union[str, nemo_asr.models.EncDecCTCModelBPE]): Path to the pretrained checkpoint or name of a 
                                                               pretrained model from 
                                                               `nemo_asr.models.EncDecCTCModel.get_available_model_names()`
        output_dir (str): Directory where the transcribed segments will be saved.
        data_dir (str): Directory containing audio files.
        vocabulary (str): Vocabulary file containing a list of words used during training.
        window_len (int, optional): Window length in milliseconds. Defaults to 8000.

    Raises:
        ValueError: If neither a checkpoint file or a pretrained model name is provided.

    """
    bpe_model = isinstance(model, nemo_asr.models.EncDecCTCModelBPE)

    # Get tokenizer used during training, None for char based models
    if bpe_model:
        tokenizer = asr_model.tokenizer
    else:
        tokenizer = None

    # Load the ASR model
    if os.path.exists(model):
        asr_model = nemo_asr.models.EncDecCTCModel.restore_from(model)
    elif model in nemo_asr.models.EncDecCTCModel.get_available_model_names():
        asr_model = nemo_asr.models.EncDecCTCModel.from_pretrained(model, strict=False)
    else:
        try:
            asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model)
        except:
            raise ValueError(
                f"Provide path to the pretrained checkpoint or choose from {nemo_asr.models.EncDecCTCModel.get_available_model_names()}"
            )

    # Get audio file paths
    data = Path(data_dir)
    audio_paths = data.glob("*.wav")

    all_log_probs = []
    all_transcript_file = []
    all_segment_file = []
    all_wav_paths = []

    # Create segments directory
    segments_dir = os.path.join(output_dir, "segments")
    os.makedirs(segments_dir, exist_ok=True)

    index_duration = None

    # Transcribe each audio file
    for path_audio in audio_paths:
        transcript_file = os.path.join(data_dir, path_audio.name.replace(".wav", ".txt"))
        segment_file = os.path.join(
            segments_dir, f"{window_len}_" + path_audio.name.replace(".wav", "_segments.txt")
        )
        if not os.path.exists(transcript_file):
            print(f"{transcript_file} not found. Skipping {path_audio.name}")
            continue
        sample_rate, signal = wav.read(path_audio)
        if len(signal) == 0:
            print(f"Skipping {path_audio.name}")
            continue

        assert (
            sample_rate == sample_rate
        ), f"Sampling rate of the audio file {path_audio} doesn't match --sample_rate={sample_rate}"

        original_duration = len(signal) / sample_rate
        log_probs = asr_model.transcribe(paths2audio_files=[str(path_audio)], batch_size=1, logprobs=True)[0]
        blank_col = log_probs[:, -1].reshape((log_probs.shape[0], 1))
        log_probs = np.concatenate((blank_col, log_probs[:, :-1]), axis=1)


        all_log_probs.append(log_probs)
        all_segment_file.append(str(segment_file))
        all_transcript_file.append(str(transcript_file))
        all_wav_paths.append(path_audio)

        if index_duration is None:
            index_duration = len(signal) / log_probs.shape[0] / sample_rate

    asr_model_type = type(asr_model)
    del asr_model
    torch.cuda.empty_cache()

    if len(all_log_probs) > 0:
        for i in tqdm(range(len(all_log_probs))):
            delayed(get_segments)(
                all_log_probs[i],
                all_wav_paths[i],
                all_transcript_file[i],
                all_segment_file[i],
                vocabulary,
                tokenizer,
                bpe_model,
                index_duration,
                window_len
            )





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



def process_alignment(alignment_file: str, clips_dir: str, max_duration, threshold, offset=0, padding=0):
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
            segment = [float(line[0]) + float(offset) + padding, float(line[1]) + float(offset) - padding]
            segments.append((float(line[0]) + float(offset) + padding, float(line[1]) + float(offset) - padding, float(line[2])))

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