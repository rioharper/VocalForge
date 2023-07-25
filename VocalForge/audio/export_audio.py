from typing import Optional
import os
from pydub import AudioSegment
from pydub.effects import normalize
from torch import cuda
from df.enhance import enhance, init_df, load_audio, save_audio
from .audio_utils import get_files


class ExportAudio:
    """
    A class for exporting audio files with various processing options.

    Args:
        input_dir (str, optional): The directory containing the input audio files. Defaults to None.
        export_dir (str, optional): The directory to export the formatted audio files. Defaults to None.
        noise_removed_dir (str, optional): The directory to export the noise-removed audio files. Defaults to None.
        normalization_dir (str, optional): The directory to export the normalized audio files. Defaults to None.
        sample_rate (int, optional): The sample rate of the audio files. Defaults to 22050.
    """

    def __init__(
        self,
        input_dir: Optional[str] = None,
        export_dir: Optional[str] = None,
        noise_removed_dir: Optional[str] = None,
        normalization_dir: Optional[str] = None,
        sample_rate: int = 22050,
    ):
        self.Input_Dir = input_dir
        self.Export_Dir = export_dir
        self.Noise_Removed_Dir = noise_removed_dir
        self.Normalization_Dir = normalization_dir
        self.Input_Files = get_files(self.Input_Dir)
        self.Sample_Rate = sample_rate

    def normalize(self, file_dir: str, output_dir: str) -> None:
        """
        Normalize an audio file and export it to the specified directory.

        Args:
            file_dir (str): The directory of the audio file to normalize.
            output_dir (str): The directory to export the normalized audio file.
        """
        audio = AudioSegment.from_file(file_dir, format="wav")
        normalized = normalize(audio)
        normalized.export(
            os.path.join(output_dir, os.path.basename(file_dir)), format="wav"
        )

    def normalize_folder(self) -> None:
        """
        Normalize all audio files in the export directory and export them to the normalization directory.
        """
        for file in get_files(self.Export_Dir):
            file_dir = os.path.join(self.Export_Dir, file)
            audio = AudioSegment.from_file(file_dir, format="wav")
            normalized = normalize(audio)
            normalized.export(
                os.path.join(self.Normalization_Dir, os.path.basename(file)),
                format="wav",
            )

    def noise_remove(self) -> None:
        """
        Removes noise from audio files in `file_or_folder` directory or file.

        Parameters:
        file_or_folder (str or os.PathLike): The directory or file containing the audio files.
        sample_rate (int): The sample rate of the audio files.

        Returns:
        None
        """
        model, df_state, _ = init_df(config_allow_defaults=True)

        for file in self.Input_Files:
            print(f"Removing Noise from {file}...")
            try:
                file_dir = os.path.join(self.Export_Dir, file)
                audio, _ = load_audio(file_dir, sr=self.Sample_Rate)
                enhanced = enhance(model, df_state, audio)
                save_audio(file_dir, enhanced, sr=self.Sample_Rate)
                cuda.empty_cache()
            except RuntimeError:
                print(f"file is too large for GPU, skipping: {file}")

    def format_audio_folder(self) -> None:
        """
        Format all audio files in the input directory and export them to the export directory.
        """
        for file in get_files(self.Input_Dir, full_dir=True, ext=".wav"):
            raw = AudioSegment.from_file(file, format="wav")
            raw = raw.set_channels(1)
            raw = raw.set_frame_rate(self.Sample_Rate)
            raw.export(
                os.path.join(self.Export_Dir, os.path.basename(file)), format="wav"
            )

    def run(self) -> None:
        """
        Run the audio export process with the specified options.
        """
        if os.listdir(self.Export_Dir) != []:
            print("file(s) have already been formatted! Skipping...")
        else:
            self.format_audio_folder()

        if (
            self.Noise_Removed_Dir is not None
            and os.listdir(self.Noise_Removed_Dir) == []
        ):
            print("Removing Noise...")
            self.noise_remove()

        if (
            self.Normalization_Dir is not None
            and os.listdir(self.Normalization_Dir) == []
        ):
            print("Normalizing Audio...")
            self.normalize_folder()
