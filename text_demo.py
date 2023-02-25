import os
from text.text_utils import create_core_folders
from text import RefineText
import argparse

parser = argparse.ArgumentParser(description='Modify parameters for dataset generation')
parser.add_argument("--aud_dir",
    help="directory for audio (str, required)",
    required=True)
parser.add_argument("--work_dir",
    help="directory for the various stages of generation (str, required)",
    required=True)
parser.add_argument("--model",
    help="name of Nvidia ASR model (str, default: nvidia/stt_en_citrinet_1024_gamma_0_25)", 
    default='nvidia/stt_en_citrinet_1024_gamma_0_25', type=str)
parser.add_argument("--max_length", 
    help="max length in words of each utterence (int, default: 25)", 
    default=25, type=int)
parser.add_argument("--max_duration",
    help="max length of a single audio clip in s (int, default: 40)",
    default=40, type=int)
parser.add_argument("--lang", 
    help="language of the speaker (str, default: en)", 
    default='en', type=str)
parser.add_argument("--window_size", 
    help="window size for ctc segmentation algorithm (int, default: 8000)", 
    default=8000, type=int)
parser.add_argument("--offset", 
    help="offset for audio clips in ms (int, default: 0)", 
    default=0, type=int)
parser.add_argument("--threshold", 
    help="min score of segmentation confidence to split (float, range: 0-10, lower=more selective, default=2.5)", 
    default=2.5, type=float)


args = parser.parse_args()
aud_dir = args.aud_dir
work_dir = args.work_dir

folders = ['transcription', 'processed', 'segments', 'sliced_audio', 'dataset']
create_core_folders(folders, work_dir)

RefineText = RefineText(
    aud_dir=aud_dir, 
    transcription_dir=os.path.join(work_dir, 'transcription'),
    processed_text_dir=os.path.join(work_dir, 'processed'),
    segment_dir=os.path.join(work_dir,'segments'),
    sliced_audio_dir=os.path.join(work_dir,'sliced_audio'),
    dataset_dir=os.path.join(work_dir, 'dataset'),
    model=args.model,
    window_size=args.window_size,
    offset=args.offset,
    max_duration=args.max_duration,
    max_length=args.max_length,
    threshold=args.threshold,
    lang=args.lang
)
RefineText.refine_text()