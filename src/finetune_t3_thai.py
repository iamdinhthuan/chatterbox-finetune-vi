import argparse
import logging
import os
import json
import random
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any
import time

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import librosa
import numpy as np

# Set up safe globals for PyTorch serialization
# Add all numpy dtypes for PyTorch 2.6+ compatibility
import numpy.dtypes
safe_globals = [
    np.ndarray, 
    np.dtype,
    np.dtypes.UInt32DType,
    np.dtypes.Int64DType,
    np.dtypes.Float32DType,
    np.dtypes.Float64DType,
    # Add the actual dtype classes as well
    np.uint32,
    np.int64,
    np.float32,
    np.float64,
]
try:
    torch.serialization.add_safe_globals(safe_globals)
except AttributeError:
    # Fallback for older PyTorch versions
    pass

from transformers import (
    HfArgumentParser,
    EarlyStoppingCallback,
    set_seed,
    TrainerCallback,
    Trainer,
    PretrainedConfig
)
from transformers import TrainingArguments as HfTrainingArguments
from transformers.trainer_utils import get_last_checkpoint
from transformers.trainer import ParallelMode
from datasets import load_dataset, DatasetDict, VerificationMode, logging as ds_logging, DownloadConfig
import datasets

from chatterbox.tts import ChatterboxTTS, punc_norm, REPO_ID
from chatterbox.models.t3.t3 import T3, T3Cond
from chatterbox.models.t3.modules.t3_config import T3Config
from chatterbox.models.s3tokenizer import S3_SR
from chatterbox.utils.preprocessed_dataset import PreprocessedDataset

try:
    from chatterbox.utils.training_args import CustomTrainingArguments
except ImportError:
    # Fallback to local definition if not available
    CustomTrainingArguments = None

logger = logging.getLogger(__name__)

# --- Custom Training Arguments ---
if CustomTrainingArguments is None:
    @dataclass
    class CustomTrainingArguments(HfTrainingArguments):
        early_stopping_patience: Optional[int] = field(
            default=None, metadata={"help": "Enable early stopping with specified patience. Default: None (disabled)."}
        )
        use_torch_profiler: bool = field(
            default=False, metadata={"help": "Enable PyTorch profiler for performance analysis"}
        )

# --- Argument Classes (ModelArguments, DataArguments) ---
@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    local_model_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to local directory containing ve.safetensors, t3_cfg.safetensors, etc. Overrides model_name_or_path for loading."}
    )
    model_config: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a json file specifying local paths to models to load."}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    freeze_voice_encoder: bool = field(default=True, metadata={"help": "Freeze the Voice Encoder."})
    freeze_s3gen: bool = field(default=True, metadata={"help": "Freeze the S3Gen model (speech token to waveform)."})
    tokenizer_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to custom tokenizer.json file. If not provided, will use default tokenizer from model."}
    )

@dataclass
class DataArguments:
    dataset_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the directory containing audio files and text files. Used if dataset_name is not provided."}
    )
    metadata_file: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a metadata file. Used if dataset_name is not provided."}
    )
    train_metadata_file: Optional[str] = field(
        default=None,
        metadata={"help": "Path to train metadata CSV file. If provided with val_metadata_file, will use separate files instead of splitting."}
    )
    val_metadata_file: Optional[str] = field(
        default=None,
        metadata={"help": "Path to validation metadata CSV file. If provided with train_metadata_file, will use separate files instead of splitting."}
    )
    audio_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Directory containing audio files (overrides metadata file parent directory)."}
    )
    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the Hugging Face datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={"help": "The configuration name of the dataset to use (via the Hugging Face datasets library)."}
    )
    train_split_name: str = field(default="train", metadata={"help": "The name of the training data set split."})
    eval_split_name: Optional[str] = field(default="validation", metadata={"help": "The name of the evaluation data set split."})
    text_column_name: str = field(default="text", metadata={"help": "The name of the text column in the HF dataset."})
    audio_column_name: str = field(default="audio", metadata={"help": "The name of the audio column in the HF dataset."})
    max_text_len: int = field(default=256, metadata={"help": "Maximum length of text tokens (including BOS/EOS)."})
    max_speech_len: int = field(default=800, metadata={"help": "Maximum length of speech tokens (including BOS/EOS)."})
    audio_prompt_duration_s: float = field(
        default=3.0, metadata={"help": "Duration of audio (from start) to use for T3 conditioning prompt tokens (in seconds)."}
    )
    eval_split_size: float = field(
        default=0.0005, metadata={"help": "Fraction of data to use for evaluation if splitting manually. Not used if dataset_name provides eval split."}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    ignore_verifications: bool = field(
        default=False, metadata={"help":"Set to true to ignore dataset verifications."}
    )
    use_streaming: bool = field(
        default=False, metadata={"help": "Use streaming mode for datasets to reduce memory usage."}
    )
    use_preprocessed: bool = field(
        default=False, metadata={"help": "Use preprocessed .pt files for 2-4x faster training."}
    )
    preprocessed_dir: Optional[str] = field(
        default="./preprocessed_data", metadata={"help": "Directory containing preprocessed .pt files."}
    )
    

# --- Dataset Classes ---
class SpeechFineTuningIterableDataset(torch.utils.data.IterableDataset):
    """Iterable dataset for streaming support"""
    def __init__(self,
                 data_args: DataArguments,
                 t3_config: T3Config,
                 hf_dataset: Union[datasets.Dataset, List[Dict[str, str]]],
                 is_hf_format: bool,
                 model_dir: str,
                 m_paths: Optional[dict] = None,
                 device: str = "cpu",
                 transcripts: dict = None):
        # Store raw args
        self.data_args = data_args
        self.chatterbox_model = None
        self.chatterbox_t3_config = t3_config
        self.dataset_source = hf_dataset
        self.is_hf_format = is_hf_format
        self.transcripts = transcripts or {}

        # Placeholders for components, will be initialized lazily
        self.text_tokenizer = None
        self.speech_tokenizer = None
        self.voice_encoder = None

        # Path to model checkpoint directory for lazy loading
        self._model_dir = model_dir
        self.m_paths = m_paths
        self._device = device

        # Sampling and conditioning setup
        self.s3_sr = S3_SR
        self.enc_cond_audio_len_samples = int(data_args.audio_prompt_duration_s * self.s3_sr)

        self._init_model()

    def __iter__(self):
        """Iterate through the dataset"""
        worker_info = torch.utils.data.get_worker_info()
        
        if self.is_hf_format:
            # For streaming datasets
            dataset_iter = iter(self.dataset_source)
            for item in dataset_iter:
                processed_item = self._process_streaming_item(item)
                if processed_item is not None:
                    yield processed_item
        else:
            # For local file lists
            for item in self.dataset_source:
                processed_item = self._process_local_item(item)
                if processed_item is not None:
                    yield processed_item

    def _process_streaming_item(self, item):
        """Process a single item from streaming dataset"""
        # Extract text - handle multiple formats
        text = None
        
        # Try standard column name
        if self.data_args.text_column_name in item:
            text = item[self.data_args.text_column_name]
        # Thai GigaSpeech2 format
        elif "transcript" in item:
            text = item["transcript"]
        # Thai GigaSpeech2 with transcript lookup
        elif "__key__" in item and self.transcripts:
            # Extract segment ID from key
            key = item.get('__key__', '')
            if key:
                parts = key.split('/')
                if parts:
                    segment_id = parts[-1]
                    text = self.transcripts.get(segment_id, "")
        
        if not text:
            logger.debug(f"No text found for item. Available keys: {list(item.keys()) if hasattr(item, 'keys') else 'N/A'}")
            return None

        # Extract audio - handle multiple formats
        audio_data = None
        
        # Try standard column name
        if self.data_args.audio_column_name in item:
            audio_data = item[self.data_args.audio_column_name]
        # Thai GigaSpeech2 format
        elif "audio" in item:
            audio_data = item["audio"]
        elif "wav" in item:
            audio_data = item["wav"]
        
        if audio_data is None:
            logger.error(f"No audio found. Available keys: {list(item.keys()) if hasattr(item, 'keys') else 'N/A'}")
            return None

        # Process audio
        wav_16k = self._load_and_process_audio(audio_data)
        if wav_16k is None:
            return None

        # Process into model format
        return self._process_audio_text_pair(wav_16k, text)

    def _process_local_item(self, item):
        """Process a single item from local files"""
        audio_path = item["audio"]
        text = item["text"]
        
        try:
            wav_16k, _ = librosa.load(audio_path, sr=self.s3_sr, mono=True)
            return self._process_audio_text_pair(wav_16k, text)
        except Exception as e:
            logger.error(f"Error loading audio {audio_path}: {e}")
            return None

    def _load_and_process_audio(self, audio_data):
        """Load and process audio from various formats"""
        try:
            # Handle different audio formats
            if isinstance(audio_data, str):
                wav_array, original_sr = librosa.load(audio_data, sr=None, mono=True)
            elif isinstance(audio_data, dict) and "array" in audio_data and "sampling_rate" in audio_data:
                wav_array = audio_data["array"]
                original_sr = audio_data["sampling_rate"]
            elif isinstance(audio_data, (bytes, bytearray)):
                # Handle streaming bytes
                import io
                wav_array, original_sr = librosa.load(io.BytesIO(audio_data), sr=None, mono=True)
            else:
                logger.error(f"Unexpected audio data format: {type(audio_data)}")
                return None

            if not isinstance(wav_array, np.ndarray):
                wav_array = np.array(wav_array)

            # Resample if needed
            if original_sr != self.s3_sr:
                wav_16k = librosa.resample(wav_array, orig_sr=original_sr, target_sr=self.s3_sr)
            else:
                wav_16k = wav_array.copy()
            
            # Convert to mono if needed
            if wav_16k.ndim > 1:
                wav_16k = librosa.to_mono(wav_16k)
            
            # Convert to float32
            if wav_16k.dtype != np.float32:
                wav_16k = wav_16k.astype(np.float32)

            return wav_16k
            
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            return None

    def _process_audio_text_pair(self, wav_16k, text):
        """Process audio and text into model input format"""
        # This is the same processing as the original __getitem__ method
        if wav_16k is None or text is None or len(wav_16k) == 0:
            return None

        try:
            # Ensure model is loaded
            self._init_model()
            speaker_emb_np = self.voice_encoder.embeds_from_wavs([wav_16k], sample_rate=self.s3_sr)
            speaker_emb = torch.from_numpy(speaker_emb_np[0])
        except Exception as e:
            logger.error(f"Error getting speaker embedding: {e}")
            return None

        normalized_text = punc_norm(text)
        raw_text_tokens = self.text_tokenizer.text_to_tokens(normalized_text).squeeze(0)
        text_tokens = F.pad(raw_text_tokens, (1, 0), value=self.chatterbox_t3_config.start_text_token)
        text_tokens = F.pad(text_tokens, (0, 1), value=self.chatterbox_t3_config.stop_text_token)
        if len(text_tokens) > self.data_args.max_text_len:
            text_tokens = text_tokens[:self.data_args.max_text_len-1]
            text_tokens = torch.cat([text_tokens, torch.tensor([self.chatterbox_t3_config.stop_text_token], device=text_tokens.device)])
        text_token_len = torch.tensor(len(text_tokens), dtype=torch.long)

        try:
            raw_speech_tokens_batch, speech_token_lengths_batch = self.speech_tokenizer.forward([wav_16k])
            if raw_speech_tokens_batch is None or speech_token_lengths_batch is None:
                logger.error(f"S3Tokenizer returned None")
                return None
            raw_speech_tokens = raw_speech_tokens_batch.squeeze(0)[:speech_token_lengths_batch.squeeze(0).item()]
        except Exception as e:
            logger.error(f"Error getting speech tokens: {e}")
            return None
            
        speech_tokens = F.pad(raw_speech_tokens, (1, 0), value=self.chatterbox_t3_config.start_speech_token)
        speech_tokens = F.pad(speech_tokens, (0, 1), value=self.chatterbox_t3_config.stop_speech_token)
        if len(speech_tokens) > self.data_args.max_speech_len:
            speech_tokens = speech_tokens[:self.data_args.max_speech_len-1]
            speech_tokens = torch.cat([speech_tokens, torch.tensor([self.chatterbox_t3_config.stop_speech_token], device=speech_tokens.device)])
        speech_token_len = torch.tensor(len(speech_tokens), dtype=torch.long)

        cond_audio_segment = wav_16k[:self.enc_cond_audio_len_samples]
        if len(cond_audio_segment) == 0:
            cond_prompt_speech_tokens = torch.zeros(self.chatterbox_t3_config.speech_cond_prompt_len, dtype=torch.long)
        else:
            try:
                cond_prompt_tokens_batch, _ = self.speech_tokenizer.forward([cond_audio_segment], max_len=self.chatterbox_t3_config.speech_cond_prompt_len)
                if cond_prompt_tokens_batch is None:
                    cond_prompt_speech_tokens = torch.zeros(self.chatterbox_t3_config.speech_cond_prompt_len, dtype=torch.long)
                else:
                    cond_prompt_speech_tokens = cond_prompt_tokens_batch.squeeze(0)
            except Exception as e:
                cond_prompt_speech_tokens = torch.zeros(self.chatterbox_t3_config.speech_cond_prompt_len, dtype=torch.long)

        if cond_prompt_speech_tokens.size(0) != self.chatterbox_t3_config.speech_cond_prompt_len:
            current_len = cond_prompt_speech_tokens.size(0)
            target_len = self.chatterbox_t3_config.speech_cond_prompt_len
            if current_len > target_len:
                cond_prompt_speech_tokens = cond_prompt_speech_tokens[:target_len]
            else:
                cond_prompt_speech_tokens = F.pad(cond_prompt_speech_tokens, (0, target_len - current_len), value=0)
        
        emotion_adv_scalar = 0.5
        emotion_adv_scalar_tensor = torch.tensor(emotion_adv_scalar, dtype=torch.float)

        return {
            "text_tokens": text_tokens.long(),
            "text_token_lens": text_token_len.long(),
            "speech_tokens": speech_tokens.long(),
            "speech_token_lens": speech_token_len.long(),
            "t3_cond_speaker_emb": speaker_emb.float(),
            "t3_cond_prompt_speech_tokens": cond_prompt_speech_tokens.long(),
            "t3_cond_emotion_adv": emotion_adv_scalar_tensor,
        }

    def _init_model(self):
        """
        Lazy-load the ChatterboxTTS model and its components.
        """
        if self.chatterbox_model is None:
            from chatterbox.tts import ChatterboxTTS
            # Load model from checkpoint directory, on CPU by default
            if self.m_paths:
                self.chatterbox_model = ChatterboxTTS.from_specified(
                    voice_encoder_path=Path(self._model_dir) / self.m_paths["voice_encoder_path"],
                    t3_path=Path(self._model_dir) / self.m_paths["t3_path"],
                    s3gen_path=Path(self._model_dir) / self.m_paths["s3gen_path"],
                    tokenizer_path= self.m_paths["tokenizer_path"],
                    conds_path=Path(self._model_dir) / self.m_paths["conds_path"], 
                    device="cpu"
                    )
            else:
                self.chatterbox_model = ChatterboxTTS.from_local(self._model_dir, device=self._device)
            self.text_tokenizer = self.chatterbox_model.tokenizer
            self.speech_tokenizer = self.chatterbox_model.s3gen.tokenizer
            self.voice_encoder = self.chatterbox_model.ve

    def __getstate__(self):
        # Drop unpickleable objects; they will be reloaded in each worker
        state = self.__dict__.copy()
        state['chatterbox_model'] = None
        state['text_tokenizer'] = None
        state['speech_tokenizer'] = None
        state['voice_encoder'] = None
        return state

    def __setstate__(self, state):
        # Restore state and reload model
        self.__dict__.update(state)
        self._init_model()


# Keep the original class for backward compatibility
class SpeechFineTuningDataset(Dataset):
    def __init__(self,
                 data_args: DataArguments,

                 t3_config: T3Config,
                 hf_dataset: Union[datasets.Dataset, List[Dict[str, str]]],
                 is_hf_format: bool,
                 model_dir: str,
                 m_paths: Optional[dict] = None,
                 device: str = "cpu"):
        # Store raw args
        self.data_args = data_args
        self.chatterbox_model = None
        self.chatterbox_t3_config = t3_config
        self.dataset_source = hf_dataset
        self.is_hf_format = is_hf_format

        # Placeholders for components, will be initialized lazily
        self.text_tokenizer = None
        self.speech_tokenizer = None
        self.voice_encoder = None

        # Path to model checkpoint directory for lazy loading
        self._model_dir = model_dir
        self.m_paths = m_paths
        self._device = device

        # Sampling and conditioning setup
        self.s3_sr = S3_SR
        self.enc_cond_audio_len_samples = int(data_args.audio_prompt_duration_s * self.s3_sr)


        self._init_model()

    def __len__(self):
        return len(self.dataset_source)

    def _load_audio_text_from_item(self, idx):
        if self.is_hf_format:
            item = self.dataset_source[idx]
            text = item[self.data_args.text_column_name]
            audio_data = item[self.data_args.audio_column_name]
            
            if isinstance(audio_data, str):
                 wav_array, original_sr = librosa.load(audio_data, sr=None, mono=True)
            elif isinstance(audio_data, dict) and "array" in audio_data and "sampling_rate" in audio_data:
                wav_array = audio_data["array"]
                original_sr = audio_data["sampling_rate"]
            else:
                logger.error(f"Unexpected audio data format for item {idx}: {type(audio_data)}. Skipping.")
                return None, None

            if not isinstance(wav_array, np.ndarray):
                logger.error(f"Audio array is not numpy for item {idx}: {type(wav_array)}. Skipping.")
                return None, None

            if original_sr != self.s3_sr:
                wav_16k = librosa.resample(wav_array, orig_sr=original_sr, target_sr=self.s3_sr)
            else:
                wav_16k = wav_array.copy()
            
            if wav_16k.ndim > 1: wav_16k = librosa.to_mono(wav_16k)
            if wav_16k.dtype != np.float32:
                wav_16k = wav_16k.astype(np.float32)

            item_info_for_log = f"Item {idx} (text: '{text[:30]}...', audio_len: {len(wav_16k)}, audio_dtype: {wav_16k.dtype})"

            return wav_16k, text
        else:
            item = self.dataset_source[idx]
            audio_path = item["audio"]
            text = item["text"]
            try:
                wav_16k, _ = librosa.load(audio_path, sr=self.s3_sr, mono=True)
                return wav_16k, text
            except Exception as e:
                logger.error(f"Error loading audio {audio_path}: {e}")
                return None, None

    def __getitem__(self, idx) -> Optional[Dict[str, Union[torch.Tensor, float]]]:
        wav_16k, text = self._load_audio_text_from_item(idx)
        if wav_16k is None or text is None or len(wav_16k) == 0:
            return None

        try:
            speaker_emb_np = self.voice_encoder.embeds_from_wavs([wav_16k], sample_rate=self.s3_sr)
            speaker_emb = torch.from_numpy(speaker_emb_np[0])
        except Exception as e:
            logger.error(f"Error getting speaker embedding for item {idx}: {e}. Skipping.")
            return None

        normalized_text = punc_norm(text)
        raw_text_tokens = self.text_tokenizer.text_to_tokens(normalized_text).squeeze(0)
        text_tokens = F.pad(raw_text_tokens, (1, 0), value=self.chatterbox_t3_config.start_text_token)
        text_tokens = F.pad(text_tokens, (0, 1), value=self.chatterbox_t3_config.stop_text_token)
        if len(text_tokens) > self.data_args.max_text_len:
            text_tokens = text_tokens[:self.data_args.max_text_len-1]
            text_tokens = torch.cat([text_tokens, torch.tensor([self.chatterbox_t3_config.stop_text_token], device=text_tokens.device)])
        text_token_len = torch.tensor(len(text_tokens), dtype=torch.long)

        try:
            raw_speech_tokens_batch, speech_token_lengths_batch = self.speech_tokenizer.forward([wav_16k])
            if raw_speech_tokens_batch is None or speech_token_lengths_batch is None:
                logger.error(f"S3Tokenizer returned None for item {idx}. Skipping.")
                return None
            raw_speech_tokens = raw_speech_tokens_batch.squeeze(0)[:speech_token_lengths_batch.squeeze(0).item()]
        except Exception as e:
            logger.error(f"Error getting speech tokens for item {idx}: {e}. Skipping.")
            return None
            
        speech_tokens = F.pad(raw_speech_tokens, (1, 0), value=self.chatterbox_t3_config.start_speech_token)
        speech_tokens = F.pad(speech_tokens, (0, 1), value=self.chatterbox_t3_config.stop_speech_token)
        if len(speech_tokens) > self.data_args.max_speech_len:
            speech_tokens = speech_tokens[:self.data_args.max_speech_len-1]
            speech_tokens = torch.cat([speech_tokens, torch.tensor([self.chatterbox_t3_config.stop_speech_token], device=speech_tokens.device)])
        speech_token_len = torch.tensor(len(speech_tokens), dtype=torch.long)

        cond_audio_segment = wav_16k[:self.enc_cond_audio_len_samples]
        if len(cond_audio_segment) == 0 :
            cond_prompt_speech_tokens = torch.zeros(self.chatterbox_t3_config.speech_cond_prompt_len, dtype=torch.long)
        else:
            try:
                cond_prompt_tokens_batch, _ = self.speech_tokenizer.forward([cond_audio_segment], max_len=self.chatterbox_t3_config.speech_cond_prompt_len)
                if cond_prompt_tokens_batch is None:
                    #  logger.error(f"S3Tokenizer returned None for cond_prompt for item {idx}. Using zeros.")
                     cond_prompt_speech_tokens = torch.zeros(self.chatterbox_t3_config.speech_cond_prompt_len, dtype=torch.long)
                else:
                    cond_prompt_speech_tokens = cond_prompt_tokens_batch.squeeze(0)
            except Exception as e:
                # logger.error(f"Error getting cond prompt tokens for item {idx}: {e}. Using zeros.")
                cond_prompt_speech_tokens = torch.zeros(self.chatterbox_t3_config.speech_cond_prompt_len, dtype=torch.long)

        if cond_prompt_speech_tokens.size(0) != self.chatterbox_t3_config.speech_cond_prompt_len:
            current_len = cond_prompt_speech_tokens.size(0)
            target_len = self.chatterbox_t3_config.speech_cond_prompt_len
            if current_len > target_len: cond_prompt_speech_tokens = cond_prompt_speech_tokens[:target_len]
            else: cond_prompt_speech_tokens = F.pad(cond_prompt_speech_tokens, (0, target_len - current_len), value=0)
        
        emotion_adv_scalar=0.5
        emotion_adv_scalar_tensor = torch.tensor(emotion_adv_scalar, dtype=torch.float)

        return_dict = {
            "text_tokens": text_tokens.long(),
            "text_token_lens": text_token_len.long(),
            "speech_tokens": speech_tokens.long(),
            "speech_token_lens": speech_token_len.long(),
            "t3_cond_speaker_emb": speaker_emb.float(),
            "t3_cond_prompt_speech_tokens": cond_prompt_speech_tokens.long(),
            "t3_cond_emotion_adv": emotion_adv_scalar_tensor,
        }

        return return_dict

    def _init_model(self):
        """
        Lazy-load the ChatterboxTTS model and its components.
        """
        if self.chatterbox_model is None:
            from chatterbox.tts import ChatterboxTTS
            # Load model from checkpoint directory, on CPU by default
            if self.m_paths:
                self.chatterbox_model = ChatterboxTTS.from_specified(
                    voice_encoder_path=Path(self._model_dir) / self.m_paths["voice_encoder_path"],
                    t3_path=Path(self._model_dir) / self.m_paths["t3_path"],
                    s3gen_path=Path(self._model_dir) / self.m_paths["s3gen_path"],
                    tokenizer_path= self.m_paths["tokenizer_path"],
                    conds_path=Path(self._model_dir) / self.m_paths["conds_path"], 
                    device="cpu"
                    )
            else:
                self.chatterbox_model = ChatterboxTTS.from_local(self._model_dir, device=self._device)
            self.text_tokenizer = self.chatterbox_model.tokenizer
            self.speech_tokenizer = self.chatterbox_model.s3gen.tokenizer
            self.voice_encoder = self.chatterbox_model.ve

    def __getstate__(self):
        # Drop unpickleable objects; they will be reloaded in each worker
        state = self.__dict__.copy()
        state['chatterbox_model'] = None
        state['text_tokenizer'] = None
        state['speech_tokenizer'] = None
        state['voice_encoder'] = None
        return state

    def __setstate__(self, state):
        # Restore state and reload model
        self.__dict__.update(state)
        self._init_model()

# --- Data Collator ---
@dataclass
class SpeechDataCollator:
    t3_config: T3Config  # Chatterbox T3Config
    text_pad_token_id: int
    speech_pad_token_id: int

    def __call__(self, features: List[Optional[Dict[str, Any]]]) -> Dict[str, Any]:
        valid_features = [f for f in features if f is not None]

        if not valid_features:
            logger.warning("SpeechDataCollator received no valid features. Returning empty batch.")
            return {}
        features = valid_features

        batch_size = len(features)
        text_tokens_list = [f["text_tokens"] for f in features]
        speech_tokens_list = [f["speech_tokens"] for f in features]
        max_text_len = max(len(t) for t in text_tokens_list)
        max_speech_len = max(len(t) for t in speech_tokens_list)

        # Pad text tokens
        padded_text_tokens = torch.stack([
            F.pad(t, (0, max_text_len - len(t)), value=self.text_pad_token_id)
            for t in text_tokens_list
        ])  # shape: (B, max_text_len)

        # Pad speech tokens
        padded_speech_tokens = torch.stack([
            F.pad(s, (0, max_speech_len - len(s)), value=self.speech_pad_token_id)
            for s in speech_tokens_list
        ])  # shape: (B, max_speech_len)

        # Collect lengths
        text_token_lens = torch.stack([f["text_token_lens"] for f in features])      # (B,)
        speech_token_lens = torch.stack([f["speech_token_lens"] for f in features])  # (B,)

        # Collect conditionals
        t3_cond_speaker_emb = torch.stack([f["t3_cond_speaker_emb"] for f in features])             # (B, D_speaker)
        t3_cond_prompt_speech_tokens = torch.stack([f["t3_cond_prompt_speech_tokens"] for f in features])  # (B, prompt_len)
        emotion_adv_scalars = torch.stack([f["t3_cond_emotion_adv"] for f in features])  # (B, 1, 1)
        t3_cond_emotion_adv = emotion_adv_scalars.view(batch_size, 1, 1)

        IGNORE_ID = -100
        prompt_len = self.t3_config.speech_cond_prompt_len

        # --- Build labels_text ---
        # Shift off BOS from padded_text_tokens: new length = max_text_len - 1
        shifted_text = padded_text_tokens[:, 1:].contiguous()  # shape: (B, max_text_len - 1)
        T_text = shifted_text.size(1)

        # Mask positions t >= (text_len - 1)
        text_lens_minus_one = (text_token_lens - 1).clamp(min=0)  # (B,)
        arange_text = torch.arange(T_text, device=shifted_text.device)  # (T_text,)
        mask_pad_text = arange_text[None] >= text_lens_minus_one[:, None]  # (B, T_text)

        labels_text = shifted_text.clone()           # (B, T_text)
        labels_text[mask_pad_text] = IGNORE_ID       # set pad/beyond to -100

        # --- Build labels_speech ---
        # Shift off BOS from padded_speech_tokens: new length = max_speech_len - 1
        shifted_speech = padded_speech_tokens[:, 1:].contiguous()  # shape: (B, max_speech_len - 1)
        T_speech = shifted_speech.size(1)

        # Mask positions t >= (speech_len - 1)
        speech_lens_minus_one = (speech_token_lens - 1).clamp(min=0)  # (B,)
        arange_speech = torch.arange(T_speech, device=shifted_speech.device)  # (T_speech,)
        mask_pad_speech = arange_speech[None] >= speech_lens_minus_one[:, None]  # (B, T_speech)

        # Mask positions t < prompt_len
        # Prevent masking away the entire target sequence by capping the prompt mask per-sample.
        per_sample_prompt_cap = torch.minimum(
            torch.full_like(speech_lens_minus_one, prompt_len),
            torch.clamp(speech_lens_minus_one - 1, min=0)
        )
        mask_prompt = arange_speech[None, :] < per_sample_prompt_cap[:, None]

        # Combine masks
        mask_speech_total = mask_pad_speech | mask_prompt  # (B, T_speech)

        labels_speech = shifted_speech.clone()          # (B, T_speech)
        labels_speech[mask_speech_total] = IGNORE_ID    # set prompt & pad to -100

        return {
            "text_tokens": padded_text_tokens, 
            "text_token_lens": text_token_lens,
            "speech_tokens": padded_speech_tokens, 
            "speech_token_lens": speech_token_lens,
            "t3_cond_speaker_emb": t3_cond_speaker_emb,
            "t3_cond_prompt_speech_tokens": t3_cond_prompt_speech_tokens,
            "t3_cond_emotion_adv": t3_cond_emotion_adv,
            "labels_text": labels_text,       # (B, max_text_len - 1) masked with -100
            "labels_speech": labels_speech,   # (B, max_speech_len - 1) masked with -100
            "labels": labels_speech,          # Add this for Trainer compatibility - use speech labels as main
        }
# --- Model Wrapper ---
class T3ForFineTuning(torch.nn.Module):
    def __init__(self, t3_model: T3, chatterbox_t3_config: T3Config):
        super().__init__()
        self.t3 = t3_model
        self.chatterbox_t3_config = chatterbox_t3_config

        class HFCompatibleConfig(PretrainedConfig):
            model_type = "chatterbox_t3_finetune"
            def __init__(self, **kwargs):
                super().__init__(**kwargs)

        hf_config_instance = HFCompatibleConfig()
        hf_config_instance.llama_config_name = chatterbox_t3_config.llama_config_name
        hf_config_instance.text_tokens_dict_size = chatterbox_t3_config.text_tokens_dict_size
        hf_config_instance.speech_tokens_dict_size = chatterbox_t3_config.speech_tokens_dict_size
        hf_config_instance.max_text_tokens = chatterbox_t3_config.max_text_tokens
        hf_config_instance.max_speech_tokens = chatterbox_t3_config.max_speech_tokens
        hf_config_instance.speech_cond_prompt_len = chatterbox_t3_config.speech_cond_prompt_len
        hf_config_instance.start_text_token = chatterbox_t3_config.start_text_token
        hf_config_instance.stop_text_token = chatterbox_t3_config.stop_text_token
        hf_config_instance.start_speech_token = chatterbox_t3_config.start_speech_token
        hf_config_instance.stop_speech_token = chatterbox_t3_config.stop_speech_token
        self.config = hf_config_instance

    def forward(self,
                text_tokens,
                text_token_lens,
                speech_tokens,
                speech_token_lens,
                t3_cond_speaker_emb,
                t3_cond_prompt_speech_tokens,
                t3_cond_emotion_adv,
                labels_text = None,
                labels_speech=None,
                labels=None):

        # Use "labels" as fallback for labels_speech (for Trainer compatibility)
        if labels_speech is None and labels is not None:
            labels_speech = labels

        current_t3_cond = T3Cond(
                                speaker_emb=t3_cond_speaker_emb,
                                cond_prompt_speech_tokens=t3_cond_prompt_speech_tokens,
                                cond_prompt_speech_emb=None,
                                emotion_adv=t3_cond_emotion_adv
                                ).to(device=self.t3.device)

        loss_text, loss_speech, speech_logits = self.t3.loss(
                                t3_cond=current_t3_cond,
                                text_tokens=text_tokens,
                                text_token_lens=text_token_lens,
                                speech_tokens=speech_tokens,
                                speech_token_lens=speech_token_lens,
                                labels_text =labels_text,
                                labels_speech=labels_speech
                                )
        
        total_loss = loss_text + loss_speech

        return total_loss, speech_logits



trainer_instance: Optional[Trainer] = None


class SafeCheckpointTrainer(Trainer):
    """Custom trainer that handles PyTorch 2.6 checkpoint loading issues"""
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Override to ensure loss is properly extracted from tuple output.
        """
        outputs = model(**inputs)
        
        # Model returns tuple (loss, logits)
        if isinstance(outputs, (tuple, list)):
            loss = outputs[0]
        else:
            loss = outputs
        
        return (loss, outputs) if return_outputs else loss
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """
        Override prediction_step for evaluation to properly return loss.
        """
        has_labels = all(inputs.get(k) is not None for k in self.label_names)
        
        # Move inputs to device
        inputs = self._prepare_inputs(inputs)
        
        # Forward pass
        with torch.no_grad():
            outputs = model(**inputs)
            
            # Extract loss and logits from tuple (loss, logits)
            if isinstance(outputs, (tuple, list)):
                loss = outputs[0] if len(outputs) > 0 else None
                logits = outputs[1] if len(outputs) > 1 else None
            else:
                loss = outputs
                logits = None
        
        # Detach and move to CPU, check for NaN
        if loss is not None and isinstance(loss, torch.Tensor):
            loss = loss.detach()
            # Skip NaN losses - don't let them contaminate the average
            if torch.isnan(loss) or torch.isinf(loss):
                if not hasattr(self, '_nan_count'):
                    self._nan_count = 0
                    self._nan_samples = []
                self._nan_count += 1
                
                # Log detailed info for first 10 NaN samples
                if self._nan_count <= 10:
                    # Extract sample info - handle both tensor and scalar
                    text_len = inputs.get('text_token_lens', torch.tensor(0))
                    if isinstance(text_len, torch.Tensor):
                        text_len = text_len.item() if text_len.numel() == 1 else text_len[0].item()
                    
                    speech_len = inputs.get('speech_token_lens', torch.tensor(0))
                    if isinstance(speech_len, torch.Tensor):
                        speech_len = speech_len.item() if speech_len.numel() == 1 else speech_len[0].item()
                    
                    # audio_path and text might not be in inputs (not passed by collator)
                    # Try to extract from batch if available
                    audio_path = 'unknown (not in batch)'
                    text = 'N/A (not in batch)'
                    
                    logger.warning(
                        f"⚠️ NaN/Inf loss #{self._nan_count}:\n"
                        f"  Audio: {audio_path}\n"
                        f"  Text: {text}\n"
                        f"  Text tokens: {text_len}\n"
                        f"  Speech tokens: {speech_len}\n"
                        f"  Loss value: {loss.item() if isinstance(loss, torch.Tensor) else loss}"
                    )
                    
                    self._nan_samples.append({
                        'audio_path': audio_path,
                        'text_len': text_len,
                        'speech_len': speech_len,
                        'loss': loss.item()
                    })
                
                loss = None  # Return None so Trainer skips this batch
        
        if prediction_loss_only:
            return (loss, None, None)
        
        # Detach logits
        if logits is not None and isinstance(logits, torch.Tensor):
            logits = logits.detach()
        
        # Return (loss, logits, labels)
        return (loss, logits, None)
    
    def evaluate(self, *args, **kwargs):
        """Override evaluate to log NaN summary after evaluation"""
        # Reset NaN counter before evaluation
        if hasattr(self, '_nan_count'):
            old_nan_count = self._nan_count
            old_nan_samples = getattr(self, '_nan_samples', [])
            self._nan_count = 0
            self._nan_samples = []
        
        # Run evaluation
        result = super().evaluate(*args, **kwargs)
        
        # Log summary after evaluation
        if hasattr(self, '_nan_count') and self._nan_count > 0:
            logger.warning(
                f"\n{'='*60}\n"
                f"📊 EVALUATION SUMMARY:\n"
                f"  Total NaN/Inf batches: {self._nan_count}\n"
                f"  Evaluation dataset size: {len(self.eval_dataset)}\n"
                f"  NaN ratio: {self._nan_count / len(self.eval_dataset) * 100:.2f}%\n"
            )
            
            if self._nan_samples:
                logger.warning("🔍 Sample NaN cases (first 10):")
                for i, sample in enumerate(self._nan_samples[:10], 1):
                    logger.warning(
                        f"  {i}. {sample['audio_path']}\n"
                        f"     Text len: {sample['text_len']}, Speech len: {sample['speech_len']}"
                    )
                
                # Analyze patterns
                if len(self._nan_samples) >= 3:
                    avg_text_len = sum(s['text_len'] for s in self._nan_samples) / len(self._nan_samples)
                    avg_speech_len = sum(s['speech_len'] for s in self._nan_samples) / len(self._nan_samples)
                    logger.warning(
                        f"\n📈 PATTERN ANALYSIS:\n"
                        f"  Avg text len in NaN samples: {avg_text_len:.1f}\n"
                        f"  Avg speech len in NaN samples: {avg_speech_len:.1f}\n"
                    )
            
            logger.warning(f"{'='*60}\n")
        
        return result
    
    def _load_rng_state(self, checkpoint):
        """Override to handle weights_only loading issues"""
        if checkpoint is None:
            return
        
        rng_file = os.path.join(checkpoint, "rng_state.pth")
        if not os.path.exists(rng_file):
            logger.info(f"Checkpoint {checkpoint} does not contain an RNG state file.")
            return
        
        try:
            # First try with weights_only=True (default in PyTorch 2.6)
            checkpoint_rng_state = torch.load(rng_file)
        except Exception as e:
            logger.warning(f"Failed to load RNG state with weights_only=True: {e}")
            try:
                # Fallback to weights_only=False
                checkpoint_rng_state = torch.load(rng_file, weights_only=False)
                logger.info("Successfully loaded RNG state with weights_only=False")
            except Exception as e2:
                logger.warning(f"Failed to load RNG state entirely: {e2}")
                logger.warning("Continuing without RNG state restoration")
                return
        
        # Rest of the original implementation
        if self.args.world_size > 1:
            process_index = self.args.process_index
            rng_file = os.path.join(checkpoint, f"rng_state_{process_index}.pth")
            if not os.path.exists(rng_file):
                logger.info(f"Checkpoint {checkpoint} does not contain an RNG state for process {process_index}.")
                return
            checkpoint_rng_state = torch.load(rng_file)
        
        random.setstate(checkpoint_rng_state["python"])
        np.random.set_state(checkpoint_rng_state["numpy"])
        torch.random.set_rng_state(checkpoint_rng_state["cpu"])
        if torch.cuda.is_available():
            if self.args.parallel_mode == ParallelMode.DISTRIBUTED:
                torch.cuda.random.set_rng_state_all(checkpoint_rng_state["cuda"])
            else:
                torch.cuda.random.set_rng_state(checkpoint_rng_state["cuda"])


def run_training(model_args, data_args, training_args):
    """Main training function"""
    global trainer_instance
    
    # Enable PyTorch profiler if requested
    use_torch_profiler = getattr(training_args, 'use_torch_profiler', False)
    profiler_output_dir = os.path.join(training_args.output_dir, "profiler_output")
    
    if use_torch_profiler:
        os.makedirs(profiler_output_dir, exist_ok=True)
        logger.info(f"PyTorch profiler enabled, output dir: {profiler_output_dir}")
        # Initialize PyTorch profiler
        from torch.profiler import profile, schedule, ProfilerActivity, tensorboard_trace_handler
        prof = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(wait=1, warmup=1, active=3, repeat=2),
            on_trace_ready=tensorboard_trace_handler(profiler_output_dir),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )
        prof.start()
    
    # Auto-detect and resume from the last checkpoint if not explicitly provided
    if training_args.resume_from_checkpoint is None and not training_args.overwrite_output_dir:
        # Add numpy types to safe globals for PyTorch 2.6+
        try:
            torch.serialization.add_safe_globals([np.core.multiarray._reconstruct])
        except AttributeError:
            # For newer numpy versions
            torch.serialization.add_safe_globals([np._core.multiarray._reconstruct])
        
        # Add numpy.ndarray to safe globals
        torch.serialization.add_safe_globals([np.ndarray])
        
        last_ckpt = get_last_checkpoint(training_args.output_dir)
        if last_ckpt:
            # Check if we can load the RNG state file
            rng_file = os.path.join(last_ckpt, "rng_state.pth")
            if os.path.exists(rng_file):
                try:
                    # Try loading with weights_only=False for compatibility
                    test_load = torch.load(rng_file, weights_only=False)
                    training_args.resume_from_checkpoint = last_ckpt
                    logger.info(f"Found existing checkpoint, resuming from: {last_ckpt}")
                except Exception as e:
                    logger.warning(f"Failed to load RNG state from checkpoint: {e}")
                    logger.warning("Starting fresh training. Use --overwrite_output_dir to ignore checkpoints.")
            else:
                training_args.resume_from_checkpoint = last_ckpt
                logger.info(f"Found existing checkpoint, resuming from: {last_ckpt}")

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("Model parameters %s", model_args)
    logger.info("Data parameters %s", data_args)
    set_seed(training_args.seed)

    logger.info("Loading ChatterboxTTS model...")

    original_model_dir_for_copy: Optional[Path] = None
    m_paths = None
    
    if model_args.model_config:
        logger.info(f"Loading model from model config file: {model_args.model_config}")
        with open(model_args.model_config, "r") as file:
            m_paths = json.load(file)
        # Download the base model if using model_config
        repo_name = "ResembleAI/chatterbox"
        from huggingface_hub import snapshot_download
        download_dir = Path(training_args.output_dir) / "chatterbox_weights"
        download_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(repo_name, local_dir_use_symlinks=False, local_dir=download_dir)
        chatterbox_model = ChatterboxTTS.from_specified(
            voice_encoder_path=download_dir / m_paths["voice_encoder_path"],
            t3_path=download_dir / m_paths["t3_path"],
            s3gen_path=download_dir / m_paths["s3gen_path"],
            tokenizer_path=m_paths["tokenizer_path"],
            conds_path=download_dir / m_paths["conds_path"], 
            device="cpu"
        )
        original_model_dir_for_copy = download_dir
    elif model_args.local_model_dir:
        logger.info(f"Loading model from local directory: {model_args.local_model_dir}")
        local_dir_path = Path(model_args.local_model_dir)
        chatterbox_model = ChatterboxTTS.from_local(ckpt_dir=str(local_dir_path), device="cpu")
        original_model_dir_for_copy = local_dir_path
    else:
        repo_to_download = model_args.model_name_or_path or REPO_ID
        logger.info(f"Loading model from Hugging Face Hub: {repo_to_download}")
        download_dir = Path(training_args.output_dir) / "pretrained_model_download"
        download_dir.mkdir(parents=True, exist_ok=True)
        # Download model files (excluding tokenizer if custom one is provided)
        if model_args.tokenizer_path:
            files_to_download = ["ve.safetensors", "t3_cfg.safetensors", "s3gen.safetensors"]
            logger.info(f"Using custom tokenizer from: {model_args.tokenizer_path}")
        else:
            files_to_download = ["ve.safetensors", "t3_cfg.safetensors", "s3gen.safetensors", "tokenizer.json"]

        from huggingface_hub import hf_hub_download as hf_download

        for f in files_to_download:
            try: hf_download(repo_id=repo_to_download, filename=f, local_dir=download_dir, local_dir_use_symlinks=False, cache_dir=model_args.cache_dir)
            except Exception as e: logger.warning(f"Could not download {f} from {repo_to_download}: {e}.")

        try: hf_download(repo_id=repo_to_download, filename="conds.pt", local_dir=download_dir, local_dir_use_symlinks=False, cache_dir=model_args.cache_dir)
        except: logger.info("conds.pt not found on Hub or failed to download for this model.")

        # Copy custom tokenizer if provided
        if model_args.tokenizer_path:
            import shutil
            custom_tokenizer_path = Path(model_args.tokenizer_path)
            if custom_tokenizer_path.exists():
                target_tokenizer_path = download_dir / "tokenizer.json"
                shutil.copy(custom_tokenizer_path, target_tokenizer_path)
                logger.info(f"Copied custom tokenizer to: {target_tokenizer_path}")
            else:
                logger.error(f"Custom tokenizer not found at: {model_args.tokenizer_path}")
                raise FileNotFoundError(f"Custom tokenizer not found: {model_args.tokenizer_path}")

        chatterbox_model = ChatterboxTTS.from_local(ckpt_dir=download_dir, device="cpu")
        original_model_dir_for_copy = download_dir

    t3_model = chatterbox_model.t3
    chatterbox_t3_config_instance = t3_model.hp

    if model_args.freeze_voice_encoder:
        for param in chatterbox_model.ve.parameters(): param.requires_grad = False
        logger.info("Voice Encoder frozen.")
    if model_args.freeze_s3gen:
        for param in chatterbox_model.s3gen.parameters(): param.requires_grad = False
        logger.info("S3Gen model frozen.")
    for param in t3_model.parameters(): param.requires_grad = True
    logger.info("T3 model set to trainable.")

    logger.info("Loading and processing dataset...")
    raw_datasets = DatasetDict()
    verification_mode = VerificationMode.NO_CHECKS if data_args.ignore_verifications else VerificationMode.BASIC_CHECKS

    train_hf_dataset: Union[datasets.Dataset, List[Dict[str,str]]]
    eval_hf_dataset: Optional[Union[datasets.Dataset, List[Dict[str,str]]]] = None 

    if data_args.dataset_name:
        logger.info(f"Loading dataset '{data_args.dataset_name}' from Hugging Face Hub.")
        
        # Handle Thai GigaSpeech2 dataset specially
        ds_logging.set_verbosity_info()           # show INFO from datasets
        ds_logging.enable_progress_bar()
        download_config = DownloadConfig()
        
        logger.info("Loading dataset...")
        start_time = time.time()
        
        if data_args.dataset_name == "speechcolab/gigaspeech2" and data_args.use_streaming:
            raw_datasets_loaded = load_dataset(
                data_args.dataset_name,
                data_files={'train': 'data/th/train/*.tar.gz'},
                split='train',
                streaming=True,
                download_config=download_config,
                verification_mode=verification_mode
            )
            # For streaming, we don't have splits, so we'll handle evaluation differently
            train_hf_dataset = raw_datasets_loaded
            eval_hf_dataset = None  # Could implement train/eval split for streaming if needed
            is_hf_format_train, is_hf_format_eval = True, True
        else:
            raw_datasets_loaded = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                cache_dir=model_args.cache_dir,
                download_config=download_config,
                verification_mode=verification_mode,
                streaming=data_args.use_streaming
            )
            
            if data_args.use_streaming:
                # For streaming datasets, we can't check keys
                train_hf_dataset = raw_datasets_loaded
                eval_hf_dataset = None
                is_hf_format_train, is_hf_format_eval = True, True
            else:
                if data_args.train_split_name not in raw_datasets_loaded:
                    raise ValueError(f"Train split '{data_args.train_split_name}' not found. Available: {list(raw_datasets_loaded.keys())}")
                train_hf_dataset = raw_datasets_loaded[data_args.train_split_name]
        
        logger.info("Dataset loaded.")
        end_time = time.time()
        logger.info(f"Time taken to load dataset: {end_time - start_time} seconds")
        
        if training_args.do_eval:
            if data_args.eval_split_name and data_args.eval_split_name in raw_datasets_loaded:
                eval_hf_dataset = raw_datasets_loaded[data_args.eval_split_name]
            elif "validation" in raw_datasets_loaded: eval_hf_dataset = raw_datasets_loaded["validation"]
            elif "test" in raw_datasets_loaded: eval_hf_dataset = raw_datasets_loaded["test"]
            elif data_args.eval_split_size > 0 and hasattr(train_hf_dataset, "__len__") and len(train_hf_dataset) > 1 : # Ensure dataset is splittable
                logger.info(f"Splitting train dataset for evaluation with ratio {data_args.eval_split_size}")
                split_dataset = train_hf_dataset.train_test_split(test_size=data_args.eval_split_size, seed=training_args.seed)
                train_hf_dataset, eval_hf_dataset = split_dataset["train"], split_dataset["test"]
                logger.info(f"Evaluation set size: {len(eval_hf_dataset)}")
            else: logger.warning("Evaluation requested but no eval split found/configured or train dataset too small to split. Skipping eval dataset.")
        is_hf_format_train, is_hf_format_eval = True, True
    else:
        # Helper function to load metadata from CSV file
        def load_metadata_csv(csv_path: Path, audio_dir: Optional[Path] = None):
            files = []
            dataset_root = audio_dir if audio_dir else csv_path.parent
            skipped_count = 0
            missing_audio_count = 0

            with open(csv_path, 'r', encoding='utf-8') as f:
                for line_idx, line in enumerate(f):
                    # Skip header line
                    if line_idx == 0 and line.strip().lower().startswith('audio'):
                        continue

                    # Skip empty lines
                    if not line.strip():
                        continue

                    parts = line.strip().split('|')
                    if len(parts) != 2:
                        parts = line.strip().split('\t')

                    if len(parts) == 2:
                        audio_file, text = parts

                        # Skip if audio_file or text is empty
                        if not audio_file.strip() or not text.strip():
                            skipped_count += 1
                            if skipped_count <= 10:  # Only log first 10
                                logger.warning(f"Empty audio or text at line {line_idx+1}. Skipping.")
                            continue

                        audio_path = Path(audio_file) if Path(audio_file).is_absolute() else dataset_root / audio_file
                        if audio_path.exists():
                            files.append({"audio": str(audio_path), "text": text})
                        else:
                            missing_audio_count += 1
                            if missing_audio_count <= 10:  # Only log first 10 missing files
                                logger.warning(f"Audio file not found: {audio_path} (line {line_idx+1}). Skipping.")
                    else:
                        skipped_count += 1
                        if skipped_count <= 10:  # Only log first 10 malformed lines
                            logger.warning(f"Skipping malformed line in metadata (line {line_idx+1}): {line.strip()[:100]}")

            # Summary at the end
            if skipped_count > 10:
                logger.warning(f"Total {skipped_count} malformed/empty lines skipped (only first 10 logged)")
            if missing_audio_count > 10:
                logger.warning(f"Total {missing_audio_count} audio files not found (only first 10 logged)")

            return files

        # Check if separate train/val files are provided
        if data_args.train_metadata_file and data_args.val_metadata_file:
            # Load separate train and validation files
            train_csv_path = Path(data_args.train_metadata_file)
            val_csv_path = Path(data_args.val_metadata_file)
            audio_dir_path = Path(data_args.audio_dir) if data_args.audio_dir else None

            logger.info(f"Loading train data from: {train_csv_path}")
            train_files = load_metadata_csv(train_csv_path, audio_dir_path)
            logger.info(f"Loaded {len(train_files)} training samples")

            logger.info(f"Loading validation data from: {val_csv_path}")
            val_files = load_metadata_csv(val_csv_path, audio_dir_path)
            logger.info(f"Loaded {len(val_files)} validation samples")

            if not train_files:
                raise ValueError(f"No training data found in {train_csv_path}")

            train_hf_dataset = train_files # type: ignore
            eval_hf_dataset = val_files if val_files else None # type: ignore

        elif data_args.metadata_file:
            # Original behavior: load single file and split
            all_files = []
            metadata_path = Path(data_args.metadata_file)
            # Use audio_dir if provided, otherwise use metadata file parent directory
            dataset_root = Path(data_args.audio_dir) if data_args.audio_dir else metadata_path.parent

            logger.info(f"Loading data from single metadata file: {metadata_path}")
            all_files = load_metadata_csv(metadata_path, dataset_root)
            logger.info(f"Loaded {len(all_files)} samples from metadata file")

            if not all_files:
                raise ValueError(f"No data found in {metadata_path}")

            np.random.shuffle(all_files)
            train_hf_dataset = all_files # type: ignore

            # Split for evaluation if needed
            if data_args.eval_split_size > 0 and training_args.do_eval and len(all_files) > 1:
                split_idx = int(len(all_files) * (1 - data_args.eval_split_size))
                if split_idx == 0:
                    split_idx = 1  # Ensure at least one for train
                if split_idx == len(all_files):
                    split_idx = len(all_files) - 1  # Ensure at least one for eval
                train_hf_dataset, eval_hf_dataset = all_files[:split_idx], all_files[split_idx:] # type: ignore
                logger.info(f"Split into {len(train_hf_dataset)} train and {len(eval_hf_dataset)} eval samples")

        elif data_args.dataset_dir:
            # Load from directory with .wav and .txt files
            all_files = []
            dataset_path = Path(data_args.dataset_dir)
            for audio_file_path in dataset_path.rglob("*.wav"):
                text_file_path = audio_file_path.with_suffix(".txt")
                if text_file_path.exists():
                    with open(text_file_path, 'r', encoding='utf-8') as f:
                        text = f.read().strip()
                    all_files.append({"audio": str(audio_file_path), "text": text})

            if not all_files:
                raise ValueError(f"No data files found in {dataset_path}")

            logger.info(f"Loaded {len(all_files)} samples from dataset directory")
            np.random.shuffle(all_files)
            train_hf_dataset = all_files # type: ignore

            # Split for evaluation if needed
            if data_args.eval_split_size > 0 and training_args.do_eval and len(all_files) > 1:
                split_idx = int(len(all_files) * (1 - data_args.eval_split_size))
                if split_idx == 0:
                    split_idx = 1
                if split_idx == len(all_files):
                    split_idx = len(all_files) - 1
                train_hf_dataset, eval_hf_dataset = all_files[:split_idx], all_files[split_idx:] # type: ignore
        else:
            raise ValueError("No data source provided. Specify train_metadata_file+val_metadata_file, metadata_file, or dataset_dir.")

        is_hf_format_train, is_hf_format_eval = False, False

    # Check if we're dealing with a streaming dataset
    is_streaming = hasattr(train_hf_dataset, '__iter__') and not hasattr(train_hf_dataset, '__len__')
    
    # Initialize transcripts dict (used for Thai dataset)
    transcripts = {}
    
    if is_streaming or data_args.use_streaming:
        # Load transcripts for Thai dataset if needed
        if data_args.dataset_name == "speechcolab/gigaspeech2":
            logger.info("Loading Thai transcripts...")
            from huggingface_hub import hf_hub_download
            import csv
            try:
                tsv_path = hf_hub_download(
                    repo_id="speechcolab/gigaspeech2",
                    filename="data/th/train_refined.tsv",
                    repo_type="dataset"
                )
                with open(tsv_path, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f, delimiter='\t')
                    for row in reader:
                        if len(row) >= 2:
                            segment_id = row[0].strip()
                            transcript = row[1].strip()
                            transcripts[segment_id] = transcript
                logger.info(f"Loaded {len(transcripts)} Thai transcripts")
            except Exception as e:
                logger.warning(f"Failed to load transcripts: {e}")
        
        # Use IterableDataset for streaming
        if model_args.model_config:
            train_dataset = SpeechFineTuningIterableDataset(
                data_args,
                chatterbox_t3_config_instance,
                train_hf_dataset,
                is_hf_format_train,
                model_dir=str(original_model_dir_for_copy),
                m_paths=m_paths,
                device="cpu",
                transcripts=transcripts
            )
        else:
            train_dataset = SpeechFineTuningIterableDataset(
                data_args,
                chatterbox_t3_config_instance,
                train_hf_dataset,
                is_hf_format_train,
                model_dir=str(original_model_dir_for_copy),
                m_paths=None,
                device="cpu",
                transcripts=transcripts
            )
    else:
        # Use regular Dataset for non-streaming
        if data_args.use_preprocessed:
            # Use preprocessed dataset for faster training
            logger.info(f"🚀 Using preprocessed dataset from {data_args.preprocessed_dir}")
            train_dataset = PreprocessedDataset(
                preprocessed_dir=data_args.preprocessed_dir,
                max_text_len=data_args.max_text_len,
                max_speech_len=data_args.max_speech_len,
                split='train',
                eval_split_size=data_args.eval_split_size
            )
        elif model_args.model_config:
            train_dataset = SpeechFineTuningDataset(
                data_args,
                chatterbox_t3_config_instance,
                train_hf_dataset,
                is_hf_format_train,
                model_dir=str(original_model_dir_for_copy),
                m_paths=m_paths,
                device="cpu"
            )
        else:
            train_dataset = SpeechFineTuningDataset(
                data_args,
                chatterbox_t3_config_instance,
                train_hf_dataset,
                is_hf_format_train,
                model_dir=str(original_model_dir_for_copy),
                m_paths=None,
                device="cpu"
            )

    eval_dataset = None
    if eval_hf_dataset and training_args.do_eval:
        if is_streaming or data_args.use_streaming:
            if model_args.model_config:
                eval_dataset = SpeechFineTuningIterableDataset(
                    data_args,
                    chatterbox_t3_config_instance,
                    eval_hf_dataset,
                    is_hf_format_eval,
                    model_dir=str(original_model_dir_for_copy),
                    m_paths=m_paths,
                    device="cpu",
                    transcripts=transcripts
                )
            else:
                eval_dataset = SpeechFineTuningIterableDataset(
                    data_args,
                    chatterbox_t3_config_instance,
                    eval_hf_dataset,
                    is_hf_format_eval,
                    model_dir=str(original_model_dir_for_copy),
                    m_paths=None,
                    device="cpu",
                    transcripts=transcripts
                )
        else:
            if data_args.use_preprocessed:
                # Use validation split of preprocessed dataset
                logger.info(f"📊 Using preprocessed dataset for validation (split={data_args.eval_split_size})")
                eval_dataset = PreprocessedDataset(
                    preprocessed_dir=data_args.preprocessed_dir,
                    max_text_len=data_args.max_text_len,
                    max_speech_len=data_args.max_speech_len,
                    split='val',
                    eval_split_size=data_args.eval_split_size
                )
            elif model_args.model_config:
                eval_dataset = SpeechFineTuningDataset(
                    data_args,
                    chatterbox_t3_config_instance,
                    eval_hf_dataset,
                    is_hf_format_eval,
                    model_dir=str(original_model_dir_for_copy),
                    m_paths=m_paths,
                    device="cpu"
                )
            else:
                eval_dataset = SpeechFineTuningDataset(
                    data_args,
                    chatterbox_t3_config_instance,
                    eval_hf_dataset,
                    is_hf_format_eval,
                    model_dir=str(original_model_dir_for_copy),
                    m_paths=None,
                    device="cpu"
                )

    data_collator = SpeechDataCollator(chatterbox_t3_config_instance, 
                                       chatterbox_t3_config_instance.stop_text_token,
                                       chatterbox_t3_config_instance.stop_speech_token)

    hf_trainable_model = T3ForFineTuning(t3_model, chatterbox_t3_config_instance)

    # If evaluation was requested but no eval_dataset was built (e.g. streaming), disable eval
    if training_args.do_eval and eval_dataset is None:
        logger.warning("Evaluation requested but no eval_dataset found; disabling evaluation.")
        training_args.do_eval = False
        training_args.eval_strategy = "no"
        if hasattr(training_args, "eval_on_start"):
            training_args.eval_on_start = False
    
    callbacks = []
    if training_args.early_stopping_patience is not None and training_args.early_stopping_patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=training_args.early_stopping_patience))
    
    if use_torch_profiler:
        # Add profiler stepping callback
        class ProfilerCallback(TrainerCallback):
            def on_step_end(self, args, state, control, **kwargs):
                prof.step()
        callbacks.append(ProfilerCallback())

    trainer_instance = SafeCheckpointTrainer(
        model=hf_trainable_model,
        args=training_args,
        train_dataset=train_dataset, 
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=callbacks if callbacks else None
    )

    if training_args.label_names is None: trainer_instance.label_names = ["labels"]


    if training_args.do_train:
        logger.info("*** Training T3 model ***")
        # Patch previous trainer_state.json to update batch size before resuming
        ckpt = training_args.resume_from_checkpoint
        if ckpt:
            ts_path = os.path.join(ckpt, "trainer_state.json")
            if os.path.exists(ts_path):
                # Load existing state, update batch size, and rewrite file cleanly
                with open(ts_path, "r") as rf:
                    state = json.load(rf)
                state["train_batch_size"] = training_args.per_device_train_batch_size
                with open(ts_path, "w") as wf:
                    json.dump(state, wf, indent=2)
                logger.info(f"Updated train_batch_size in {ts_path} to {training_args.per_device_train_batch_size}")
        
        # Setup PyTorch profiler if enabled
        if use_torch_profiler:
            from torch.profiler import record_function
            with record_function("training"):
                train_result = trainer_instance.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
            prof.stop()
        else:
            train_result = trainer_instance.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer_instance.save_model()
        
        logger.info("Saving finetuned T3 model weights for ChatterboxTTS...")
        t3_to_save = trainer_instance.model.t3 if hasattr(trainer_instance.model, 't3') else trainer_instance.model.module.t3
        finetuned_t3_state_dict = t3_to_save.state_dict()
        
        output_t3_safetensor_path = Path(training_args.output_dir) / "t3_cfg.safetensors"
        from safetensors.torch import save_file
        save_file(finetuned_t3_state_dict, output_t3_safetensor_path)
        logger.info(f"Finetuned T3 model weights saved to {output_t3_safetensor_path}")

        if original_model_dir_for_copy:
            import shutil
            for f_name in ["ve.safetensors", "s3gen.safetensors", "tokenizer.json"]:
                src_path = original_model_dir_for_copy / f_name
                if src_path.exists(): shutil.copy2(src_path, Path(training_args.output_dir) / f_name)
            if (original_model_dir_for_copy / "conds.pt").exists():
                shutil.copy2(original_model_dir_for_copy / "conds.pt", Path(training_args.output_dir) / "conds.pt")
            logger.info(f"Full model components structured in {training_args.output_dir}")

        metrics = train_result.metrics
        trainer_instance.log_metrics("train", metrics)
        trainer_instance.save_metrics("train", metrics)
        trainer_instance.save_state()

    if training_args.do_eval and eval_dataset:
        logger.info("*** Evaluating T3 model ***")
        metrics = trainer_instance.evaluate()
        trainer_instance.log_metrics("eval", metrics)
        trainer_instance.save_metrics("eval", metrics)

    logger.info("Finetuning script finished.")
    
    
def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, CustomTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # Use preprocessing_num_workers as dataloader_num_workers if set
    if data_args.preprocessing_num_workers is not None:
        training_args.dataloader_num_workers = data_args.preprocessing_num_workers
    
    run_training(model_args, data_args, training_args)


if __name__ == "__main__":
    main()
