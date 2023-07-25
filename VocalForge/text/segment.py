from .text_utils import get_files
from pathlib import Path
import statistics
from .ctc import ctc


class Segment:
    """
    This function segments audio files into multiple parts based on the timestamps obtained
    using a model. The timestamps are used to split the audio file into smaller parts and
    are saved in a text file.

    Parameters:
    input_dir (str): Directory path of the audio files.
    output_dir (str): Directory path of the segmented audio files.
    model (str): Nvidia Model to use for segmentation.
    window_size (int): The amount (in ms) of audio to process at a time. May have to increase if files are too large.

    Generates:
    Median_Loss (float): Median loss obtained from the model timing for all audio files.
    Loss (list): List of losses obtained from the model timing for each audio file.
    """

    def __init__(
        self,
        input_dir,
        output_dir,
        model="nvidia/stt_en_citrinet_1024_gamma_0_25",
        window_size=8000,
    ):
        self.Input_Dir = input_dir
        self.Output_Dir = output_dir
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
        folder_dir (str): Directory path of the audio files.
        out_dir (str): Directory path of the segmented audio files.

        Returns:
        float: Mean loss obtained from the model.
        """
        loss_folder = []
        folder_dir = Path(folder_dir)
        out_dir = Path(out_dir)
        for aud_file in get_files(str(folder_dir), ext=".wav"):
            aud_path = folder_dir / aud_file
            outfile = out_dir / aud_file.replace(".wav", "_segmented.txt")
            loss = ctc(self.Model, str(aud_path), str(outfile), self.Window_Size)
            loss_folder.append(loss)
        try:
            return statistics.mean(loss_folder)
        except:
            return None

    def find_median_loss(self):
        self.Median_Loss = statistics.median(self.Loss)

    def run(self):
        input_dir = Path(self.Input_Dir)
        output_dir = Path(self.Output_Dir)
        for folder in get_files(str(input_dir)):
            folder_dir = input_dir / folder
            value = self.segment_folder(str(folder_dir), str(output_dir))
            if value is not None:
                self.Loss.append(value)
        self.Median_Loss = statistics.median(self.Loss)
        self.find_median_loss()
