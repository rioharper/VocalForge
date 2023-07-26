from pathlib import Path
from pydub import AudioSegment
from .process_text import format_text, split_text, normalize_text
from .text_utils import get_files


class NormalizeText:
    """
    This function prepares the audio and text files for language modeling.

    Args:
    1. input_dir (str): Directory path of the raw text files.
    2. output_dir (str): Directory path of the processed text files.
    3. audio_dir (str): Directory path of the raw audio files.
    4. model (str): Nvidia Model to use for language modeling.
    5. max_length (int): max length of the text sentences.
    7. min_length (int): min length of the text sentences.
    6. lang (str): Language of the text, i.e 'en' or 'ru'

    """

    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        audio_dir: str,
        model="nvidia/stt_en_citrinet_1024_gamma_0_25",
        max_length=25,
        min_length=0,
        lang="en",
    ):
        self.Model = model
        self.Max_Length = max_length
        self.Min_Length = min_length
        self.Lang = lang
        self.Input_Dir = Path(input_dir)
        self.Output_Dir = Path(output_dir)
        self.Audio_Dir = Path(audio_dir)
        from nemo.collections.asr.models import ASRModel

        self.Cfg = ASRModel.from_pretrained(model_name=self.Model, return_config=True)

    def prepare_audio_file(self, aud_dir, out_dir):
        """
        This function prepares the audio file for language modeling. Currently it only moves the file to the output directory.

        Args:
        1. audio_dir (str): Directory path of the raw audio file.
        2. new_dir (str): Directory path of the processed audio file.

        Returns:
        None
        """
        # Load audio and export as WAV format
        raw = AudioSegment.from_file(aud_dir, format="wav")
        raw = raw.export(out_dir, format="wav")

    def prepare_text(self, textfile_dir: str):
        """
        This function prepares the text file for language modeling.

        Args:
        1. textfile_dir (str): Directory path of the text file.

        Returns:
        sentences (str): Processed text sentences.
        """
        # Format and normalize the text
        lang = self.Lang
        cfg = self.Cfg
        transcript = format_text(textfile_dir, lang)
        sentences = split_text(
            transcript,
            lang,
            cfg.decoder.vocabulary,
            self.Max_Length,
            additional_split_symbols=None,
            min_length=self.Min_Length,
        )
        return normalize_text(sentences, lang, cfg.decoder.vocabulary)

    def write_file(self, file_num: str, text_type: str, sentences: str, outdir: str):
        """
        This function writes the processed text to a file.

        Args:
        1. file_num (str): File number to be used in the file name.
        2. text_type (str): Type of text to be written to the file.
        3. sentences (str): Processed text sentences to be written to the file.
        4. outdir (str): Directory path of the processed text files.
        """
        file_name = f"{file_num}{text_type}"
        file_dir = outdir / file_num / file_name
        print(file_dir)
        with open(file_dir.with_suffix(".txt"), "w", encoding="UTF-8") as f:
            for sentence in sentences:
                sentence = sentence.strip()
                f.write(sentence + "\n")

    def prepare_file(self, text_dir, aud_dir, out_dir):
        """
        This function prepares the text and audio files for language modeling.

        Args:
        1. text_dir (str): Directory path of the raw text file.
        2. aud_dir (str): Directory path of the raw audio file.
        3. out_dir (str): Directory path of the processed text and audio files.
        """
        base_name = text_dir.name
        sentence_types = self.prepare_text(text_dir)
        folder_dir = out_dir / base_name.replace(".txt", "")
        folder_dir.mkdir(exist_ok=True)
        for name, sentences in sentence_types.items():
            self.write_file(
                base_name.replace(".txt", ""), name, sentences.splitlines(), out_dir
            )
        self.prepare_audio_file(aud_dir, folder_dir / base_name.replace(".txt", ".wav"))

    def run(self):
        """
        This function runs the text normalization pipeline.
        """
        for file in get_files(self.Input_Dir, ".txt"):
            text_dir = self.Input_Dir / file
            audfile_dir = self.Audio_Dir / file.replace(".txt", ".wav")
            self.prepare_file(text_dir, audfile_dir, self.Output_Dir)
