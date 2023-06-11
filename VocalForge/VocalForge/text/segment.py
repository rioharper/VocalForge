from .text_utils import get_files
import os
import statistics
from .ctc import ctc

class Segment():
    """
    This function segments audio files into multiple parts based on the timestamps obtained
    using a model. The timestamps are used to split the audio file into smaller parts and
    are saved in a text file.

    Parameters:
    model (str): Pretrained model name.
    window_size (int): Size of the window for the model.

    Returns:
    None
    """
    def __init__(self, input_dir, output_dir, model='nvidia/stt_en_citrinet_1024_gamma_0_25', window_size=8000):
        self.Input_Dir = input_dir
        self.Out_Dir = output_dir
        self.Window_Size = window_size
        import nemo.collections.asr as nemo_asr
        self.Model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model)
        self.Loss = []
        self.Median_Loss = None

    def segment_folder(self, folder_dir, out_dir):
        """
        This function segments audio files in a folder into multiple parts based on
        the timestamps obtained using a model.

        Parameters:
        folder (str): Folder containing audio files.

        Returns:
        float: Mean loss obtained from the model.
        """
        loss_folder = []
        for aud_file in get_files(folder_dir, '.wav'):
            aud_path = os.path.join(folder_dir, aud_file)
            outfile = os.path.join(out_dir, aud_file.replace('.wav', '_segmented.txt'))
            loss = ctc(self.Model, aud_path, outfile, self.Window_Size)
            loss_folder.append(loss)
        try:
            return statistics.mean(loss_folder)
        except: return None

    def find_median_loss(self):
        self.Median_Loss = statistics.median(self.Loss)

    def run_segmentation(self):
        if os.listdir(self.Out_Dir) != []:
            print("folder(s) have already found its split timestamps! Skipping...")
            return
        for folder in get_files(self.Input_Dir):
            folder_dir = os.path.join(self.Input_Dir, folder)
            value = self.segment_folder(folder_dir, self.Out_Dir)
            if value != None: self.Loss.append(value)
        self.Median_Loss = statistics.median(self.Loss)
        self.find_median_loss()
        print(f"median loss: {self.Median_Loss}")