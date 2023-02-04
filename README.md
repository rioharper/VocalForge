# An End-to-End Toolkit for Voice Datasets

`vocal-forge` is an open-source toolkit written in Python 🐍  that is meant to cut down the time to create datasets for, TTS models, hotword detection models, and more so you can spend more time training, and less time sifting through audio data.

Using [Nvidia's NEMO](https://github.com/NVIDIA/NeMo), [PyAnnote](https://github.com/pyannote/pyannote-audio), [CTC segmentation](https://github.com/lumaku/ctc-segmentation) , [OpenAI's Whisper](https://github.com/openai/whisper), this repo will take you from raw audio to a fully formatted dataset, refining both the audio and text automatically.

*NOTE: While this does reduce time on spent on dataset curation, verifying the output at each step is important as it isn't perfect*

![a flow chart of how this repo works](https://github.com/rioharper/VocalForge/blob/main/media/join_processes.svg?raw=true)


## Features:

#### `refineaudio.py`
- ⬇️ **Download audio**  from a YouTube playlist (perfect for podcasts/interviews) OR input your own raw audio files (wav format)
- 🎵 **Remove Non Speech Data**
- 🗣🗣 **Remove Overlapping Speech** 
- 👥 **Split Audio File Into Speakers** 
- 👤 **Isolate the same speaker across multiple files (voice verification)** 
- 🧽 **Use DeepFilterNet to reduce background noise**
- 🧮 **Normalize Audio**
- ➡️ **Export with user defined parameters**

#### `generate_dataset.py`
- 📜 **Batch transcribe text using OpenAI's Whisper**
- 🧮 **Run [text normalization](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/nlp/text_normalization/wfst/wfst_text_normalization.html)**
- 🫶 **Use CTC segmentation to line up text to audio**
- 🖖 **Split audio based on quality of CTC segmentation confidence**
- ✅ **Generate a metadata.csv and dataset in the format of LJSpeech** 


## Setup/Requirements

🐧 **NEMO only supports Linux**

Python 3.8 has been tested, newer versions should work

CUDA is required to run all models

a [Hugging Face account](https://huggingface.co/) is required (it's free and super helpful!)

```bash
#install system libraries
apt-get update && apt-get install -y libsndfile1 ffmpeg

conda create -n vocal-forge python=3.8 pytorch=1.11.0 torchvision=0.12.0 torchaudio=0.11.0 cudatoolkit=11.3.1 -c pytorch

conda activate vocal-forge
git clone https://github.com/rioharper/VocalForge
cd VocalForge
pip install -r requirements.txt

#enter huggingface token, token can be found at https://huggingface.co/settings/tokens
huggingface-cli login
```


Pyannote models need to be "signed up for" in Hugging Face for research purposes. Don't worry, all it asks for is your purpose, website and organization. The following models will have to be manually visited and given the appropriate info:
![an example of signing up for a model](https://github.com/rioharper/VocalForge/blob/main/media/huggingface.png?raw=true)
- [Brouhaha (VAD model)](https://huggingface.co/pyannote/brouhaha)
- [Overlapped Speech Detection](https://huggingface.co/pyannote/overlapped-speech-detection)
- [Speaker Diarization](https://huggingface.co/pyannote/speaker-diarization)


## Parameters
Error rate will vary widely depending on how you set the following parameters, so make sure to play around with them! Each dataset is it's own snowflake.

##### `refineaudio.py --help`
```
--raw_dir directory for unfitered audio (str, required)
--work_dir directory for the various stages of refinement (str, required)
--sample rate Exported sample rate (int, default: 22050)
--speaker_threshold The lower the value, the more sensitive speaker seperation is (float, default: 0.2)
--verification_threshold The higher the value, the more similar two voices must be during voice verification (float, default: 0.9)
--playlist_url URL to YouTube playlist to be downloaed to raw_dir (str)
--vad_threshold The higher the value, the more selective the VAD model will be (int, default: 75)
--snr_change The lower the value, the more sensitive the model is to changes in SNR, such as laughter or loud noises (float, default: 0.75)
--sample_length create sample voice clips from raw_dir for testing purposes (in seconds)
--do_noise_reduction use deepfilternet 2 to reduce noise in the exported files (bool, default: False)
--do_normalize use mean/sd normalization, can be useful for some DL models (bool, default: False)
```

##### `generate_dataset.py --help`
```
--raw_dir directory for audio (str, required)
--work_dir directory for the various stages of generation (str, required)
--model name of Nvidia ASR model (str, default: nvidia/stt_en_citrinet_1024_gamma_0_25)
--max_length max length in words of each utterence (int, default: 25)
--max_duration max length of a single audio clip in s (int, default: 40)
--lang language of the speaker (str, default: en)
--window_size window size for ctc segmentation algorithm (int, default: 8000)
--offset offset for audio clips in ms (int, default: 0)
--threshold min score of segmentation confidence to split (float, range: 0-10, lower=more selective, default=2.5)
```


## TODO
- [ ] Refactor functions for API and toolkit support
- [ ] "Sync" datasets with the metadata file if audio clips are deleted after being generated
- [ ] Add a step in the audio refinement processs to remove emotional speech
- [ ] Add other normalization methods for audio
- [ ] Add other dataset formats for generation
- [ ] Utilize TTS models to automatically generate datasets, with audio augmentation to create diversity
