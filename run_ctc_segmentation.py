import os

from pathlib import Path

import numpy as np
import scipy.io.wavfile as wav
import torch
from joblib import Parallel, delayed
import nemo.collections.asr as nemo_asr
from tqdm import tqdm

import os

import numpy as np
from tqdm import tqdm
from utils import get_segments
from nemo.collections.common.tokenizers.sentencepiece_tokenizer import SentencePieceTokenizer



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
