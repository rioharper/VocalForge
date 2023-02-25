import argparse
import os
from audio.audio_utils import create_core_folders, download_videos, create_samples
from audio import RefineAudio

parser = argparse.ArgumentParser(description='Modify parameters for voice data refinement')
parser.add_argument("--raw_dir", 
    help="directory for unfitered audio (str, required)", 
    required=True)
parser.add_argument("--work_dir", 
    help="directory for the various stages of refinement (str, required)", 
    required=True)
parser.add_argument("--sample_rate",
    help="exported sample rate (int, default: 22050)",
    default=22050, type=int)
parser.add_argument("--playlist_url",
    help="URL to YouTube playlist to be downloaed to raw_dir (str)",
    type=str)
parser.add_argument("--vad_threshold",
    help="The higher the value, the more selective the VAD model will be (float, default: .75)",
    default=.75, type=float)
parser.add_argument("--snr_change", 
    help="The lower the value, the more sensitive the model is to changes in SNR, such as laughter or loud noises (float, default: 0.75)", 
    default=0.75, type=float)
parser.add_argument("--samples_length", 
    help="create sample voice clips from raw_dir for testing purposes (in seconds)", 
    type=int, default=None)
parser.add_argument("--verification_threshold",
    help="The higher the value, the more similar two voices must be during voice verification (float, range: 0.0-0.99, default: 0.9)",
    default=0.90, type=float)


args = parser.parse_args()
rawdir = args.raw_dir
workdir = args.work_dir

from torch import cuda
if cuda.is_available() == False:
    print("CUDA device not found! If you have CUDA intalled, please check if its propery configured")
    print("Program will continue, but at a much slower pace.")
else: print("CUDA device configured correctly!")

if args.playlist_url is not None:
    download_videos(args.playlist_url, rawdir)

folders = ['Isolated', 'No_Overlap', 'Samples', "Only_Voice", "Verification", "Exported", 'Noise_Removed', 'Normalized']
create_core_folders(folders, workdir)
if args.samples_length is not None:
    create_samples(args.samples_length, args.raw_dir, os.path.join(args.work_dir, 'Samples'))
    rawdir = os.path.join(args.work_dir, 'Samples')
    
Refine_Audio = RefineAudio(
    input_dir=args.raw_dir,
    vad_dir=os.path.join(args.work_dir, 'Only_Voice'),
    overlap_dir=os.path.join(args.work_dir, 'No_Overlap'),
    verification_dir=os.path.join(args.work_dir, 'Verification'),
    isolated_dir=os.path.join(args.work_dir, 'Isolated'),
    export_dir=os.path.join(args.work_dir, 'Exported'),
    normalized_dir=os.path.join(args.work_dir, 'Normalized'),
    noise_removed_dir=os.path.join(args.work_dir, 'Noise_Removed'),
    sample_rate=args.sample_rate,
    vad_theshold=args.vad_threshold,
    noise_aggressiveness=args.snr_change,
    verification_threshold=args.verification_threshold,
)
Refine_Audio.run_all()
