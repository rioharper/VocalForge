from .audio_utils import get_files
import os
from typing import Union
import numpy as np
from scipy.io import wavfile
from pydub import AudioSegment
from torch import cuda
from df.enhance import enhance, init_df, load_audio, save_audio


def normalize_audio(file_or_folder: Union[str, os.PathLike], mean=None, sd=None) -> None:
    """
    Normalizes audio files in `file_or_folder` directory or file.

    Parameters:
    file_or_folder (str or os.PathLike): The directory or file containing the audio files.
    mean (float): The mean value used for normalization. If None, the mean of all audio files will be used.
    sd (float): The standard deviation value used for normalization. If None, the standard deviation of all audio files will be used.

    Returns:
    None
    """
    if os.path.isdir(file_or_folder):
        # Get all WAV files in the folder
        wav_files = get_files(file_or_folder, '.wav', full_dir=True)

        # Calculate mean and standard deviation if not provided
        if mean is None or sd is None:
            data = []
            for file in wav_files:
                rate, d = wavfile.read(file)
                data.append(d)
            data = np.concatenate(data)
            mean = np.mean(data)
            sd = np.std(data)

        # Normalize each file
        for file in wav_files:
            rate, data = wavfile.read(file)
            data = (data - mean) / sd
            wavfile.write(file, rate, data)
    else:
        # Normalize the file
        rate, data = wavfile.read(file_or_folder)
        data = (data - mean) / sd
        wavfile.write(file_or_folder, rate, data)


def noise_remove(file_or_folder: Union[str, os.PathLike], sample_rate=22050) -> None:
    """
    Removes noise from audio files in `file_or_folder` directory or file.

    Parameters:
    file_or_folder (str or os.PathLike): The directory or file containing the audio files.
    sample_rate (int): The sample rate of the audio files.

    Returns:
    None
    """
    model, df_state, _ = init_df(config_allow_defaults=True)

    if os.path.isdir(file_or_folder):
        for file in get_files(file_or_folder, '.wav'):
            try:
                file_dir = os.path.join(file_or_folder, file)
                audio, _ = load_audio(file_dir, sr=sample_rate)
                enhanced = enhance(model, df_state, audio)
                save_audio(file_dir, enhanced, sr=sample_rate)
                cuda.empty_cache()
            except RuntimeError:
                print(f"file is too large for GPU, skipping: {file}")
    else:
        try:
            audio, _ = load_audio(file_or_folder, sr=sample_rate)
            enhanced = enhance(model, df_state, audio)
            save_audio(file_or_folder, enhanced, sr=sample_rate)
            cuda.empty_cache()
        except RuntimeError:
            print(f"file is too large for GPU, skipping: {file_or_folder}")

    del model, df_state


def format_audio(file_or_folder: Union[str, os.PathLike], export_dir: Union[str, os.PathLike], sample_rate=22050) -> None:
    """
    Formats audio files in `file_or_folder` directory or file and saves them to `export_dir`.

    Parameters:
    file_or_folder (str or os.PathLike): The directory or file containing the audio files.
    export_dir (str or os.PathLike): The directory to save the formatted audio file(s).
    sample_rate (int): The sample rate of the audio files.

    Returns:
    None
    """
    if os.path.isdir(file_or_folder):
        for file in get_files(file_or_folder, '.wav'):
            file_dir = os.path.join(file_or_folder, file)
            raw = AudioSegment.from_file(file_dir, format="wav")
            raw = raw.set_channels(1)
            raw = raw.set_frame_rate(sample_rate)
            raw.export(os.path.join(export_dir, file), format='wav')
    else:
        raw = AudioSegment.from_file(file_or_folder, format="wav")
        raw = raw.set_channels(1)
        raw = raw.set_frame_rate(sample_rate)
        raw.export(os.path.join(export_dir, os.path.basename(file_or_folder)), format='wav')


def process_audio(input_dir: Union[str, os.PathLike], 
                  export_dir: Union[str, os.PathLike], 
                  noise_removed_dir: Union[str, os.PathLike] = None, 
                  normalization_dir: Union[str, os.PathLike] = None, 
                  sample_rate=22050, 
                  mean=None, sd=None) -> None:
    """
    Processes audio files in `input_dir` directory and saves them to `export_dir`.

    Parameters:
    input_dir (str or os.PathLike): The directory containing the input audio files.
    export_dir (str or os.PathLike): The directory to save the formatted audio files.
    noise_removed_dir (str or os.PathLike): The directory to save the noise-removed audio files. If None, noise removal is skipped.
    normalization_dir (str or os.PathLike): The directory to save the normalized audio files. If None, normalization is skipped.
    sample_rate (int): The sample rate of the audio files.
    mean (float): The mean value used for normalization. If None, the mean of all audio files will be used.
    sd (float): The standard deviation value used for normalization. If None, the standard deviation of all audio files will be used.

    Returns:
    None
    """
    if os.listdir(export_dir) != []:
        print("file(s) have already been formatted! Skipping...")
    else:
        format_audio(input_dir, export_dir, sample_rate)

    if noise_removed_dir is not None and os.listdir(noise_removed_dir) == []:
        print("Removing Noise...")
        noise_remove(export_dir, sample_rate)
        export_dir = noise_removed_dir

    if normalization_dir is not None and os.listdir(normalization_dir) == []:
        print("Normalizing Audio...")
        normalize_audio(export_dir, mean, sd)
        export_dir = normalization_dir

