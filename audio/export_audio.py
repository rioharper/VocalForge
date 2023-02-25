from .audio_utils import get_files
import numpy as np
from scipy.io import wavfile
import os
from pydub import AudioSegment

class ExportAudio():
    def __init__(self, input_dir=None, export_dir=None, noise_removed_dir=None, normalization_dir = None, sample_rate=22050):
        self.Input_Dir = input_dir
        self.Export_Dir = export_dir
        self.Noise_Removed_Dir = noise_removed_dir 
        self.Input_Files = get_files(self.Input_Dir)
        self.Sample_Rate = sample_rate
    
    

    def find_all_mean_sd(self, folders_dir: str) -> tuple:
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
            for file in get_files(self.Input_Dir):
                rate, data = wavfile.read(os.path.join(self.Input_Dir, file))
                mean += np.mean(data)
                sd += np.std(data)
                count += 1
        mean /= count
        sd /= count
        return mean, sd


    def normalize_folder(self, folder_dir, mean, sd):
        import pathlib
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
        for file in get_files(folder_dir):
            file_dir = os.path.join(folder_dir, file)
            rate, data = wavfile.read(file_dir)
            mean_subtracted = data - mean
            eps = 2**-30
            output = mean_subtracted / (sd + eps)
            normalized_file_dir = os.path.join(self.Normalized_Dir, file)
            wavfile.write(normalized_file_dir, rate, output)

    def noise_remove_folder(self, input_folder_dir=None):
        if input_folder_dir is None:
            input_folder_dir = self.Input_Dir
        elif input_folder_dir is None and self.Input_Dir is None:
            raise ValueError("Please provide either the folder_dir or the Noise_Removed_Dir")
        
        from torch import cuda
        from df.enhance import enhance, init_df, load_audio, save_audio
        model, df_state, _ = init_df(config_allow_defaults=True)
        
        for file in get_files(input_folder_dir, '.wav'):
            try:
                file_dir = os.path.join(input_folder_dir, file)
                audio, _ = load_audio(file_dir, sr=df_state.sr())
                enhanced = enhance(model, df_state, audio)
                save_audio(os.path.join(self.Noise_Removed_Dir, file), enhanced, df_state.sr())
                cuda.empty_cache()
            except RuntimeError:
                print(f"file is too large for GPU, skipping: {file}")
        del model, df_state
        
         

    def format_audio_folder(self, folder_dir):
        for file in get_files(folder_dir, '.wav'):
            file_dir = os.path.join(folder_dir, file)
            raw = AudioSegment.from_file(file_dir, format="wav")
            raw = raw.set_channels(1)
            raw = raw.set_frame_rate(self.Sample_Rate)
            raw.export(os.path.join(self.Export_Dir, file), format='wav')
    
    def run_export(self):
        if os.listdir(self.Export_Dir) != []:
            print("file(s) have already been formatted! Skipping...")
        else: self.format_audio_folder(self.Input_Dir)            

        if self.Noise_Removed_Dir is not None and os.listdir(self.Noise_Removed_Dir)== []:
            print("Removing Noise...") 
            self.noise_remove_folder(self.Export_Dir)

        if self.Normalized_Dir is not None and os.listdir(self.Normalized_Dir)== []:
            mean, sd = self.find_all_mean_sd(self.Input_Dir)
            print("Normalizing Audio...")
            self.normalize_folder(self.Export_Dir, mean, sd)