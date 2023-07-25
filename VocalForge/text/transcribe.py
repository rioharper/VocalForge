from .text_utils import get_files
import os
from whisper import load_model


class Transcribe:
    """
    A class for transcribing audio files and writing the transcriptions to text files.

    Attributes:
        Input_Dir (str): The directory containing the audio files to transcribe.
        Output_Dir (str): The directory to write the transcriptions to.
        Prompt (str, optional): The initial prompt to use for the transcription. Defaults to None.
        Model (object): The model to use for the transcription.
        Texts (list): A list of the transcribed texts.
        do_write (bool): Whether to write the transcriptions to text files.

    Methods:
        analyze(do_write=True):
            Transcribes audio files in the input directory and writes the transcriptions to text files in the output directory.

            Args:
                do_write (bool, optional): Whether to write the transcriptions to text files. Defaults to True.

        write_transcription(result, text_file_dir):
            Writes the transcription to a text file.

            Args:
                result (dict): The transcription result.
                text_file_dir (str): The directory to write the transcription to.

        run():
            Transcribes all audio files in the `Input_Dir` folder.
    """

    def __init__(
        self, input_dir, output_dir, model="large", prompt=None, do_write=True
    ):
        """
        Initializes the Transcribe class.

        Args:
            input_dir (str): The directory containing the audio files to transcribe.
            output_dir (str): The directory to write the transcriptions to.
            model (str, optional): The model to use for the transcription. Defaults to "large".
            prompt (str, optional): The initial prompt to use for the transcription. Defaults to None.
            do_write (bool, optional): Whether to write the transcriptions to text files. Defaults to True.
        """
        self.Input_Dir = input_dir
        self.Output_Dir = output_dir
        self.Prompt = prompt
        self.Model = load_model(model)
        self.Texts = []
        self.Do_Write = do_write

    def analyze(self):
        """
        Transcribes audio files in the input directory and writes the transcriptions to text files in the output directory.

        Args:
            do_write (bool, optional): Whether to write the transcriptions to text files. Defaults to True.
        """
        for file in get_files(self.Input_Dir):
            text_file_dir = os.path.join(self.Output_Dir, file.split(".")[0] + ".txt")
            # check if the text file already exists
            if not os.path.exists(text_file_dir):
                # transcribe the audio file
                aud_file_dir = os.path.join(self.Input_Dir, file)
                result = self.Model.transcribe(aud_file_dir, initial_prompt=self.Prompt)
                self.Texts.append(result["text"].strip())
                if self.Do_Write:
                    self.write_transcription(result, text_file_dir)

    def write_transcription(self, result, text_file_dir):
        """
        Writes the transcription to a text file.

        Args:
            result (dict): The transcription result.
            text_file_dir (str): The directory to write the transcription to.
        """
        with open(text_file_dir, "w", encoding="utf-8") as f:
            f.write(result["text"].strip())

    def run(self):
        """
        Transcribes all audio files in the `Input_Dir` folder.
        """
        self.analyze()
