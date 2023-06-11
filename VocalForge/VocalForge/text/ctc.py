import os
from torch import cuda
import numpy as np
import nemo.collections.asr as nemo_asr
import scipy.io.wavfile as wav
from .ctc_utils import get_segments

def split_on_newline(dir):
    text = ''
    text_list = []
    with open(dir, 'r', encoding='utf-8') as f:
        text = f.read()
    text = text.splitlines()
    for index, line in enumerate(text):
        if len(line) == 0:
            text.pop(index)
    return text


def ctc(model, aud_path: str, out_file: str, window_size: int):
    """
    Performs automatic speech recognition and saves the recognized text in a file

    Parameters:
    model (EncDecCTCModelBPE): The trained CTC model
    aud_path (str): Path to the input audio file
    out_file (str): Path to the output file where the recognized text will be saved
    window_size (int): Size of the sliding window in seconds used to break up the audio into segments

    Returns:
    None

    """
    # Check if the model is a BPE model
    bpe_model = isinstance(model, nemo_asr.models.EncDecCTCModelBPE)

    # Get the tokenizer used during training
    # tokenizer is None for char-based models
    if bpe_model:
        tokenizer = model.tokenizer
    else:
        tokenizer = None

    # Read the audio file
    sample_rate, signal = wav.read(aud_path)

    # Path to the raw transcript file
    raw_path = aud_path.replace('.wav', '.txt')

    # Get the vocabulary used by the model
    vocabulary = ["ε"] + list(model.cfg.decoder.vocabulary)

    # Get the log probabilities of each token for each time step
    log_probs = model.transcribe(
        paths2audio_files=[str(aud_path)], batch_size=1, logprobs=True
    )[0]
    blank_col = log_probs[:, -1].reshape((log_probs.shape[0], 1))
    log_probs = np.concatenate((blank_col, log_probs[:, :-1]), axis=1)

    # Calculate the index duration
    index_duration = len(signal) / log_probs.shape[0] / sample_rate

    # Free up memory
    del model
    cuda.empty_cache()

    # Get the recognized text segments
    
    return get_segments(
        log_probs,
        aud_path,
        raw_path,
        out_file,
        vocabulary,
        tokenizer,
        bpe_model,
        index_duration,
        window_size,
    )
