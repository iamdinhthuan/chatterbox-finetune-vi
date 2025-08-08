import gc
import json
import logging
import os
import time
from dataclasses import dataclass, field
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Union

import datasets
import librosa
import soundfile as sf
import torchaudio
import torchaudio.functional as AF
import numpy as np
import pandas as pd
import psutil
# pykakasi and langdetect are not needed for Vietnamese-only training
import torch
import torch.nn as nn
import torch.nn.functional as F
import webdataset as wds
import yaml
from datasets import load_dataset, DatasetDict, VerificationMode, Audio, logging as ds_logging, DownloadConfig
from huggingface_hub import snapshot_download
from torch.utils.data import Dataset, IterableDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
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
from transformers.trainer_utils import speed_metrics

from chatterbox.models.s3tokenizer import S3_SR
from chatterbox.models.t3.modules.t3_config import T3Config
from chatterbox.models.t3.t3 import T3Cond
from chatterbox.tts import ChatterboxTTS, punc_norm, REPO_ID
from chatterbox.utils.training_args import CustomTrainingArguments


class ChatterboxT3Wrapper(nn.Module):
    """Wrapper for T3 model to handle training with HuggingFace Trainer"""

    def __init__(self, t3_model):
        super().__init__()
        self.t3 = t3_model

    def forward(self,
                text_tokens,
                text_token_lens,
                speech_tokens,
                speech_token_lens,
                t3_cond_speaker_emb,
                t3_cond_prompt_speech_tokens,
                t3_cond_emotion_adv,
                labels_text=None,
                labels_speech=None):
        # Create T3Cond object from individual components
        current_t3_cond = T3Cond(
            speaker_emb=t3_cond_speaker_emb,
            cond_prompt_speech_tokens=t3_cond_prompt_speech_tokens,
            cond_prompt_speech_emb=None,
            emotion_adv=t3_cond_emotion_adv
        ).to(device=self.t3.device)

        # Call T3 loss method
        loss_text, loss_speech, speech_logits = self.t3.loss(
            t3_cond=current_t3_cond,
            text_tokens=text_tokens,
            text_token_lens=text_token_lens,
            speech_tokens=speech_tokens,
            speech_token_lens=speech_token_lens,
            labels_text=labels_text,
            labels_speech=labels_speech
        )

        total_loss = loss_text + loss_speech
        return total_loss, speech_logits


logger = logging.getLogger(__name__)


# --- Global Collate Function ---
def speech_collate_fn(batch):
    """Global collate function for speech training data"""
    # Filter out None samples
    batch = [item for item in batch if item is not None]
    if not batch:
        return None

    # Pad sequences to same length
    max_text_len = max(len(item["text_tokens"]) for item in batch)
    max_speech_len = max(len(item["speech_tokens"]) for item in batch)

    # Prepare batch tensors
    batch_size = len(batch)
    text_tokens = torch.zeros(batch_size, max_text_len, dtype=torch.long)
    speech_tokens = torch.zeros(batch_size, max_speech_len, dtype=torch.long)
    text_lens = torch.zeros(batch_size, dtype=torch.long)
    speech_lens = torch.zeros(batch_size, dtype=torch.long)
    speaker_embs = torch.stack([item["t3_cond_speaker_emb"] for item in batch])
    prompt_tokens = torch.stack([item["t3_cond_prompt_speech_tokens"] for item in batch])
    emotion_advs = torch.stack([item["t3_cond_emotion_adv"] for item in batch])

    # Create labels for training (shifted versions of tokens)
    labels_text = torch.full((batch_size, max_text_len - 1), -100, dtype=torch.long)
    labels_speech = torch.full((batch_size, max_speech_len - 1), -100, dtype=torch.long)

    for i, item in enumerate(batch):
        text_len = len(item["text_tokens"])
        speech_len = len(item["speech_tokens"])
        text_tokens[i, :text_len] = item["text_tokens"]
        speech_tokens[i, :speech_len] = item["speech_tokens"]
        text_lens[i] = item["text_token_lens"]
        speech_lens[i] = item["speech_token_lens"]

        # Create labels (next token prediction)
        # For text: predict tokens[1:] from tokens[:-1]
        if text_len > 1:
            labels_text[i, :text_len - 1] = item["text_tokens"][1:text_len]

        # For speech: predict tokens[1:] from tokens[:-1]
        if speech_len > 1:
            labels_speech[i, :speech_len - 1] = item["speech_tokens"][1:speech_len]

    return {
        "text_tokens": text_tokens,
        "speech_tokens": speech_tokens,
        "text_token_lens": text_lens,
        "speech_token_lens": speech_lens,
        "t3_cond_speaker_emb": speaker_embs,
        "t3_cond_prompt_speech_tokens": prompt_tokens,
        "t3_cond_emotion_adv": emotion_advs,
        "labels_text": labels_text,
        "labels_speech": labels_speech,
    }


# --- Argument Classes (ModelArguments, DataArguments) ---
@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    model_config: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a json file specifying local paths to models to load."}

    )
    local_model_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to local directory containing ve.safetensors, t3_cfg.safetensors, etc. Overrides model_name_or_path for loading."}
    )

    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    freeze_voice_encoder: bool = field(default=True, metadata={"help": "Freeze the Voice Encoder."})
    freeze_s3gen: bool = field(default=True, metadata={"help": "Freeze the S3Gen model (speech token to waveform)."})
    freeze_text_embeddings: Optional[int] = field(default=None, metadata={
        "help": "Number of original text embedding tokens to freeze (e.g., 704 for original vocab size)."})


@dataclass
class DataArguments:
    dataset_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to the directory containing audio files and text files. Used if dataset_name is not provided."}
    )
    dataset_dirs: List[str] = field(
        default_factory=list,
        metadata={
            "help": "List of paths to multiple dataset directories (e.g., for multi-language training). Each directory should contain JSON and audio files."}
    )
    metadata_file: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a metadata file. Used if dataset_name is not provided."}
    )
    train_csv: Optional[str] = field(
        default=None,
        metadata={"help": "Path to training CSV file with columns 'audio' and 'transcript'."}
    )
    val_csv: Optional[str] = field(
        default=None,
        metadata={"help": "Path to validation CSV file with columns 'audio' and 'transcript'."}
    )
    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the Hugging Face datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={"help": "The configuration name of the dataset to use (via the Hugging Face datasets library)."}
    )
    train_split_name: Optional[str] = field(default="train",
                                            metadata={"help": "The name of the training data set split."})

    train_splits: List[str] = field(
        default_factory=list,
        metadata={"help": "List of language splits to use (e.g., ['de', 'fr'])."}
    )
    eval_split_name: Optional[str] = field(default="validation",
                                           metadata={"help": "The name of the evaluation data set split."})
    text_column_name: str = field(default="text", metadata={"help": "The name of the text column in the HF dataset."})
    audio_column_name: str = field(default="audio",
                                   metadata={"help": "The name of the audio column in the HF dataset."})
    max_text_len: int = field(default=256, metadata={"help": "Maximum length of text tokens (including BOS/EOS)."})
    max_speech_len: int = field(default=800, metadata={"help": "Maximum length of speech tokens (including BOS/EOS)."})
    audio_prompt_duration_s: float = field(
        default=3.0,
        metadata={"help": "Duration of audio (from start) to use for T3 conditioning prompt tokens (in seconds)."}
    )
    min_duration_s: Optional[float] = field(
        default=None,
        metadata={"help": "Minimum audio duration (in seconds) to include in training/eval."}
    )
    max_duration_s: Optional[float] = field(
        default=None,
        metadata={"help": "Maximum audio duration (in seconds) to include in training/eval."}
    )
    assume_language: Optional[str] = field(
        default=None,
        metadata={"help": "Assume a fixed language code (e.g., 'vi', 'fr', 'de') to bypass runtime language detection."}
    )
    audio_cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Directory to cache 16k wav, VE embeddings, S3 tokens, and cond prompt tokens."}
    )
    eval_split_size: float = field(
        default=0.0005, metadata={
            "help": "Fraction of data to use for evaluation if splitting manually. Not used if dataset_name provides eval split."}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    ignore_verifications: bool = field(
        default=False, metadata={"help": "Set to true to ignore dataset verifications."}
    )
    lang_split: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the language split to use."}
    )
    lang_path: Optional[str] = field(
        default=None,
        metadata={"help": "The path to the language split to use."}
    )
    lang_splits: List[str] = field(
        default_factory=list,
        metadata={"help": "List of language splits to use (e.g., ['de', 'fr'])."}
    )
    lang_paths: List[str] = field(
        default_factory=list,
        metadata={"help": "List of paths corresponding to each language split."}
    )
    use_webdataset: bool = field(
        default=False,
        metadata={
            "help": "Use webdataset format for optimized streaming and loading of large datasets like Emilia YODAS."}
    )
    webdataset_urls: Optional[str] = field(
        default=None,
        metadata={
            "help": "URL pattern for webdataset files (e.g., 'https://example.com/data-{000000..001000}.tar'). Used when use_webdataset=True."}
    )
    webdataset_shuffle_buffer: int = field(
        default=1000,
        metadata={
            "help": "Shuffle buffer size for webdataset streaming. Larger values improve randomness but use more memory."}
    )


# --- Dataset Class ---
class SpeechFineTuningDataset(Dataset):
    def __init__(self,
                 data_args: DataArguments,
                 t3_config: T3Config,
                 hf_dataset: Union[datasets.Dataset, List[Dict[str, str]]],
                 is_hf_format: bool,
                 model_dir: str,
                 m_paths: dict = None,
                 device: str = "cpu"):
        # Store raw args
        self.data_args = data_args
        self.chatterbox_t3_config = t3_config
        self.dataset_source = hf_dataset
        self.is_hf_format = is_hf_format
        # Path to model checkpoint directory for lazy loading
        self._model_dir = model_dir
        self.m_paths = m_paths
        self._device = device
        # Placeholders for components, will be initialized lazily
        self.chatterbox_model = None
        self.text_tokenizer = None
        self.speech_tokenizer = None
        self.voice_encoder = None

        # Sampling and conditioning setup
        self.s3_sr = S3_SR
        self.enc_cond_audio_len_samples = int(data_args.audio_prompt_duration_s * self.s3_sr)
        # Immediately load model in main process; workers will reload lazily
        self._init_model()

    def __len__(self):
        return len(self.dataset_source)

    def _load_audio_text_from_item(self, idx):
        if self.is_hf_format:
            item = self.dataset_source[idx]
            # Get text field, with fallback for different column names
            try:
                # HF default
                text = item[self.data_args.text_column_name]
            except KeyError:
                # Emilia Dataset
                if "json" in item and isinstance(item["json"], dict):
                    meta = item["json"]
                    if "text" in meta:
                        text = meta["text"]
                    else:
                        logger.error(
                            f"'text' field not found in JSON metadata. Available JSON keys: {list(meta.keys())}. Skipping.")
                        return None, None
                else:
                    logger.error(
                        f"Text column '{self.data_args.text_column_name}' not found. Available keys: {list(item.keys())}. Skipping.")
                    return None, None
            except Exception as e:
                logger.error(f"Error loading text for item {idx}: {e}. Skipping.")
                return None, None

            # Get audio data, with fallback for different column names
            try:
                # HF default
                audio_data = item[self.data_args.audio_column_name]
            except KeyError:
                # Emilia Dataset
                if "mp3" in item:
                    audio_data = item["mp3"]
                else:
                    for alt in ["audio", "wav"]:
                        if alt in item:
                            logger.warning(
                                f"Column '{self.data_args.audio_column_name}' not found. Using '{alt}' instead.")
                            audio_data = item[alt]
                            break
                    else:
                        logger.error(
                            f"Audio column '{self.data_args.audio_column_name}' not found. Available keys: {list(item.keys())}. Skipping.")
                        return None, None

            # Load audio from bytes (streaming), file path, or pre-loaded dict
            if isinstance(audio_data, (bytes, bytearray)):
                import io
                try:
                    wav_array, original_sr = librosa.load(io.BytesIO(audio_data), sr=None, mono=True)
                except Exception as e:
                    logger.error(f"Error loading audio bytes for item {idx}: {e}. Skipping.")
                    return None, None
            elif isinstance(audio_data, str):
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
            # Ensure model is loaded (in worker)
            self._init_model()
            speaker_emb_np = self.voice_encoder.embeds_from_wavs([wav_16k], sample_rate=self.s3_sr)
            speaker_emb = torch.from_numpy(speaker_emb_np[0])
        except Exception as e:
            logger.error(f"Error getting speaker embedding for item {idx}: {e}. Skipping.")
            return None

        normalized_text = punc_norm(text)
        # Vietnamese-only training: skip language detection and extra tagging
        
        raw_text_tokens = self.text_tokenizer.text_to_tokens(normalized_text).squeeze(0)
        text_tokens = F.pad(raw_text_tokens, (1, 0), value=self.chatterbox_t3_config.start_text_token)
        text_tokens = F.pad(text_tokens, (0, 1), value=self.chatterbox_t3_config.stop_text_token)
        if len(text_tokens) > self.data_args.max_text_len:
            text_tokens = text_tokens[:self.data_args.max_text_len - 1]
            text_tokens = torch.cat(
                [text_tokens, torch.tensor([self.chatterbox_t3_config.stop_text_token], device=text_tokens.device)])
        text_token_len = torch.tensor(len(text_tokens), dtype=torch.long)

        try:
            # Ensure tokenizer is available
            self._init_model()
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
            speech_tokens = speech_tokens[:self.data_args.max_speech_len - 1]
            speech_tokens = torch.cat([speech_tokens, torch.tensor([self.chatterbox_t3_config.stop_speech_token],
                                                                   device=speech_tokens.device)])
        speech_token_len = torch.tensor(len(speech_tokens), dtype=torch.long)

        cond_audio_segment = wav_16k[:self.enc_cond_audio_len_samples]
        if len(cond_audio_segment) == 0:
            cond_prompt_speech_tokens = torch.zeros(self.chatterbox_t3_config.speech_cond_prompt_len, dtype=torch.long)
        else:
            try:
                cond_prompt_tokens_batch, _ = self.speech_tokenizer.forward([cond_audio_segment],
                                                                            max_len=self.chatterbox_t3_config.speech_cond_prompt_len)
                if cond_prompt_tokens_batch is None:
                    cond_prompt_speech_tokens = torch.zeros(self.chatterbox_t3_config.speech_cond_prompt_len,
                                                            dtype=torch.long)
                else:
                    cond_prompt_speech_tokens = cond_prompt_tokens_batch.squeeze(0)
            except Exception as e:
                cond_prompt_speech_tokens = torch.zeros(self.chatterbox_t3_config.speech_cond_prompt_len,
                                                        dtype=torch.long)

        if cond_prompt_speech_tokens.size(0) != self.chatterbox_t3_config.speech_cond_prompt_len:
            current_len = cond_prompt_speech_tokens.size(0)
            target_len = self.chatterbox_t3_config.speech_cond_prompt_len
            if current_len > target_len:
                cond_prompt_speech_tokens = cond_prompt_speech_tokens[:target_len]
            else:
                cond_prompt_speech_tokens = F.pad(cond_prompt_speech_tokens, (0, target_len - current_len), value=0)

        emotion_adv_scalar = 0.5
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

            with tqdm(desc="Loading ChatterboxTTS components", total=1, leave=False) as pbar:
                if self.m_paths:
                    # Helper function to resolve paths correctly
                    def resolve_path(path_str):
                        """Resolve model path, avoiding duplicate chatterbox-project"""
                        if os.path.isabs(path_str):
                            # Absolute path, use as-is
                            return Path(path_str)
                        elif path_str.startswith("chatterbox-project/"):
                            # Already has chatterbox-project prefix, use as-is
                            return Path(path_str)
                        else:
                            # Relative path, add model_dir
                            return Path(self._model_dir) / path_str

                    pbar.set_description("Loading from specified paths...")
                    self.chatterbox_model = ChatterboxTTS.from_specified(
                        voice_encoder_path=resolve_path(self.m_paths["voice_encoder_path"]),
                        t3_path=resolve_path(self.m_paths["t3_path"]),
                        s3gen_path=resolve_path(self.m_paths["s3gen_path"]),
                        tokenizer_path=self.m_paths["tokenizer_path"],
                        conds_path=resolve_path(self.m_paths["conds_path"]),
                        device="cpu"
                    )
                else:
                    pbar.set_description("Loading from local directory...")
                    self.chatterbox_model = ChatterboxTTS.from_local(self._model_dir, device=self._device)

                pbar.set_description("Extracting tokenizers and encoder...")
                self.text_tokenizer = self.chatterbox_model.tokenizer
                self.speech_tokenizer = self.chatterbox_model.s3gen.tokenizer
                self.voice_encoder = self.chatterbox_model.ve
                pbar.update(1)
                pbar.set_description("Model components loaded")

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


# --- CSV Dataset Class for Vietnamese TTS ---
class CSVSpeechDataset(Dataset):
    def __init__(self,
                 csv_file: str,
                 data_args: DataArguments,
                 t3_config: T3Config,
                 model_dir: str,
                 m_paths: dict = None,
                 device: str = "cpu"):
        """
        Dataset class for CSV format with audio paths and transcripts

        Args:
            csv_file: Path to CSV file with columns 'audio' and 'transcript'
            data_args: Data arguments
            t3_config: T3 configuration
            model_dir: Model directory path
            m_paths: Model paths dictionary
            device: Device to load models on
        """
        self.csv_file = csv_file
        self.data_args = data_args
        self.chatterbox_t3_config = t3_config
        self._model_dir = model_dir
        self.m_paths = m_paths
        self._device = device

        # Load CSV data
        logger.info(f"Loading CSV dataset from {csv_file}")
        self.df = pd.read_csv(csv_file)
        logger.info(f"Loaded {len(self.df)} samples from CSV")

        # Validate CSV format
        required_columns = ['audio', 'transcript']
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        if missing_columns:
            raise ValueError(
                f"CSV file missing required columns: {missing_columns}. Found columns: {list(self.df.columns)}")

        # Filter out rows with missing data
        initial_count = len(self.df)
        self.df = self.df.dropna(subset=['audio', 'transcript'])
        self.df = self.df[self.df['transcript'].str.strip() != '']
        # Duration filter (optional)
        if self.data_args.min_duration_s is not None or self.data_args.max_duration_s is not None:
            def _get_duration_seconds(path: str) -> float:
                try:
                    info = sf.info(path)
                    if info and info.frames and info.samplerate:
                        return float(info.frames) / float(info.samplerate)
                except Exception:
                    pass
                try:
                    return float(librosa.get_duration(path=path))
                except Exception:
                    return -1.0

            durations = self.df['audio'].apply(_get_duration_seconds)
            mask = durations >= -0.5
            if self.data_args.min_duration_s is not None:
                mask &= durations >= float(self.data_args.min_duration_s)
            if self.data_args.max_duration_s is not None:
                mask &= durations <= float(self.data_args.max_duration_s)
            removed = int((~mask).sum())
            if removed > 0:
                logger.info(f"Filtered out {removed} rows outside duration bounds (min={self.data_args.min_duration_s}, max={self.data_args.max_duration_s}).")
            self.df = self.df[mask]
            self.df = self.df.reset_index(drop=True)

        final_count = len(self.df)

        if final_count < initial_count:
            logger.warning(f"Filtered out {initial_count - final_count} rows with missing/empty data")

        # Placeholders for components, will be initialized lazily
        self.chatterbox_model = None
        self.text_tokenizer = None
        self.speech_tokenizer = None
        self.voice_encoder = None

        # Audio processing setup
        self.s3_sr = S3_SR
        self.enc_cond_audio_len_samples = int(data_args.audio_prompt_duration_s * self.s3_sr)

        # Runtime cache
        self.cache_dir: Optional[Path] = None
        if self.data_args.audio_cache_dir:
            self.cache_dir = Path(self.data_args.audio_cache_dir)
            (self.cache_dir / 'wav16k').mkdir(parents=True, exist_ok=True)
            (self.cache_dir / 've').mkdir(parents=True, exist_ok=True)
            (self.cache_dir / 's3tok').mkdir(parents=True, exist_ok=True)
            (self.cache_dir / 'condtok').mkdir(parents=True, exist_ok=True)

        # Initialize model in main process
        self._init_model()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx) -> Optional[Dict[str, Union[torch.Tensor, float]]]:
        if idx >= len(self.df):
            return None

        row = self.df.iloc[idx]
        audio_path = row['audio']
        text = row['transcript']

        # Load audio (torchaudio, cache, resample only if needed)
        try:
            cache_key = hashlib.sha1(audio_path.encode('utf-8')).hexdigest() if self.cache_dir else None
            wav_cache = (self.cache_dir / 'wav16k' / f"{cache_key}.npy") if cache_key else None
            wav_16k = None
            if wav_cache and wav_cache.exists():
                wav_16k = np.load(wav_cache)
            else:
                wav, sr = torchaudio.load(audio_path)
                if wav.size(0) > 1:
                    wav = wav.mean(dim=0, keepdim=True)
                if sr != self.s3_sr:
                    wav = AF.resample(wav, sr, self.s3_sr)
                wav_16k = wav.squeeze(0).cpu().numpy().astype(np.float32)
                if wav_cache:
                    try:
                        np.save(wav_cache, wav_16k)
                    except Exception:
                        pass
            if len(wav_16k) == 0:
                logger.warning(f"Empty audio file: {audio_path}")
                return None
            # Duration guard
            dur_s = len(wav_16k) / float(self.s3_sr)
            if (self.data_args.min_duration_s is not None and dur_s < float(self.data_args.min_duration_s)) or \
               (self.data_args.max_duration_s is not None and dur_s > float(self.data_args.max_duration_s)):
                return None
        except Exception as e:
            logger.error(f"Error loading audio {audio_path}: {e}")
            return None

        # Get speaker embedding (cache)
        try:
            self._init_model()
            speaker_emb = None
            spk_cache = None
            if self.cache_dir:
                cache_key = hashlib.sha1(audio_path.encode('utf-8')).hexdigest()
                spk_cache = self.cache_dir / 've' / f"{cache_key}.npy"
                if spk_cache.exists():
                    speaker_emb = torch.from_numpy(np.load(spk_cache))
            if speaker_emb is None:
                speaker_emb_np = self.voice_encoder.embeds_from_wavs([wav_16k], sample_rate=self.s3_sr)
                speaker_emb = torch.from_numpy(speaker_emb_np[0])
                if spk_cache:
                    try:
                        np.save(spk_cache, speaker_emb.numpy())
                    except Exception:
                        pass
        except Exception as e:
            logger.error(f"Error getting speaker embedding for {audio_path}: {e}")
            return None

        # Process text (Vietnamese-only)
        normalized_text = punc_norm(text)

        # Tokenize text
        try:
            raw_text_tokens = self.text_tokenizer.text_to_tokens(normalized_text).squeeze(0)
            text_tokens = F.pad(raw_text_tokens, (1, 0), value=self.chatterbox_t3_config.start_text_token)
            text_tokens = F.pad(text_tokens, (0, 1), value=self.chatterbox_t3_config.stop_text_token)

            if len(text_tokens) > self.data_args.max_text_len:
                text_tokens = text_tokens[:self.data_args.max_text_len - 1]
                text_tokens = torch.cat(
                    [text_tokens, torch.tensor([self.chatterbox_t3_config.stop_text_token], device=text_tokens.device)])
            text_token_len = torch.tensor(len(text_tokens), dtype=torch.long)
        except Exception as e:
            logger.error(f"Error tokenizing text '{text[:50]}...': {e}")
            return None

        # Tokenize speech (cache)
        try:
            self._init_model()
            raw_speech_tokens = None
            s3_cache = None
            if self.cache_dir:
                cache_key = hashlib.sha1(audio_path.encode('utf-8')).hexdigest()
                s3_cache = self.cache_dir / 's3tok' / f"{cache_key}.npz"
                if s3_cache.exists():
                    data = np.load(s3_cache)
                    raw_speech_tokens = torch.from_numpy(data['tokens'])
            if raw_speech_tokens is None:
                raw_speech_tokens_batch, speech_token_lengths_batch = self.speech_tokenizer.forward([wav_16k])
                if raw_speech_tokens_batch is None or speech_token_lengths_batch is None:
                    logger.error(f"S3Tokenizer returned None for {audio_path}")
                    return None
                raw_speech_tokens = raw_speech_tokens_batch.squeeze(0)[:speech_token_lengths_batch.squeeze(0).item()]
                if s3_cache:
                    try:
                        np.savez_compressed(s3_cache, tokens=raw_speech_tokens.numpy())
                    except Exception:
                        pass
        except Exception as e:
            logger.error(f"Error getting speech tokens for {audio_path}: {e}")
            return None

        speech_tokens = F.pad(raw_speech_tokens, (1, 0), value=self.chatterbox_t3_config.start_speech_token)
        speech_tokens = F.pad(speech_tokens, (0, 1), value=self.chatterbox_t3_config.stop_speech_token)
        if len(speech_tokens) > self.data_args.max_speech_len:
            speech_tokens = speech_tokens[:self.data_args.max_speech_len - 1]
            speech_tokens = torch.cat([speech_tokens, torch.tensor([self.chatterbox_t3_config.stop_speech_token],
                                                                   device=speech_tokens.device)])
        speech_token_len = torch.tensor(len(speech_tokens), dtype=torch.long)

        # Conditioning audio segment
        cond_audio_segment = wav_16k[:self.enc_cond_audio_len_samples]
        if len(cond_audio_segment) == 0:
            cond_prompt_speech_tokens = torch.zeros(self.chatterbox_t3_config.speech_cond_prompt_len, dtype=torch.long)
        else:
            try:
                cond_prompt_speech_tokens = None
                cond_cache = None
                if self.cache_dir:
                    cache_key = hashlib.sha1((audio_path + f"_{self.chatterbox_t3_config.speech_cond_prompt_len}").encode('utf-8')).hexdigest()
                    cond_cache = self.cache_dir / 'condtok' / f"{cache_key}.npy"
                    if cond_cache.exists():
                        cond_prompt_speech_tokens = torch.from_numpy(np.load(cond_cache))
                if cond_prompt_speech_tokens is None:
                    cond_prompt_tokens_batch, _ = self.speech_tokenizer.forward(
                        [cond_audio_segment], max_len=self.chatterbox_t3_config.speech_cond_prompt_len
                    )
                    if cond_prompt_tokens_batch is None:
                        cond_prompt_speech_tokens = torch.zeros(self.chatterbox_t3_config.speech_cond_prompt_len,
                                                                dtype=torch.long)
                    else:
                        cond_prompt_speech_tokens = cond_prompt_tokens_batch.squeeze(0)
                        if cond_cache:
                            try:
                                np.save(cond_cache, cond_prompt_speech_tokens.numpy())
                            except Exception:
                                pass
            except Exception:
                cond_prompt_speech_tokens = torch.zeros(self.chatterbox_t3_config.speech_cond_prompt_len,
                                                        dtype=torch.long)

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
        """Lazy-load the ChatterboxTTS model and its components."""
        if self.chatterbox_model is None:
            from chatterbox.tts import ChatterboxTTS

            with tqdm(desc="Loading ChatterboxTTS components", total=1, leave=False) as pbar:
                if self.m_paths:
                    # Helper function to resolve paths correctly
                    def resolve_path(path_str):
                        """Resolve model path, avoiding duplicate chatterbox-project"""
                        if os.path.isabs(path_str):
                            # Absolute path, use as-is
                            return Path(path_str)
                        elif path_str.startswith("chatterbox-project/"):
                            # Already has chatterbox-project prefix, use as-is
                            return Path(path_str)
                        else:
                            # Relative path, add model_dir
                            return Path(self._model_dir) / path_str

                    pbar.set_description("Loading from specified paths...")
                    self.chatterbox_model = ChatterboxTTS.from_specified(
                        voice_encoder_path=resolve_path(self.m_paths["voice_encoder_path"]),
                        t3_path=resolve_path(self.m_paths["t3_path"]),
                        s3gen_path=resolve_path(self.m_paths["s3gen_path"]),
                        tokenizer_path=self.m_paths["tokenizer_path"],
                        conds_path=resolve_path(self.m_paths["conds_path"]),
                        device="cpu"
                    )
                else:
                    pbar.set_description("Loading from local directory...")
                    self.chatterbox_model = ChatterboxTTS.from_local(self._model_dir, device=self._device)

                pbar.set_description("Extracting tokenizers and encoder...")
                self.text_tokenizer = self.chatterbox_model.tokenizer
                self.speech_tokenizer = self.chatterbox_model.s3gen.tokenizer
                self.voice_encoder = self.chatterbox_model.ve
                pbar.update(1)
                pbar.set_description("Model components loaded")

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


# --- Training Callbacks ---
class DetailedProgressCallback(TrainerCallback):
    def __init__(self):
        self.start_time = None
        self.last_log_time = None
        self.samples_processed = 0
        self.step_times = []

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        self.last_log_time = self.start_time
        logger.info("Training started with detailed progress tracking")

    def on_step_end(self, args, state, control, **kwargs):
        current_time = time.time()

        # Track step timing
        if len(self.step_times) > 0:
            step_time = current_time - self.step_times[-1]
        else:
            step_time = current_time - self.last_log_time
        self.step_times.append(current_time)

        # Keep only last 10 step times for moving average
        if len(self.step_times) > 10:
            self.step_times = self.step_times[-10:]

        # Log detailed progress every 10 steps or every 30 seconds
        if state.global_step % 10 == 0 or (current_time - self.last_log_time) >= 30:
            # Calculate average step time
            if len(self.step_times) >= 2:
                recent_times = [self.step_times[i] - self.step_times[i - 1] for i in range(1, len(self.step_times))]
                avg_step_time = sum(recent_times) / len(recent_times)
            else:
                avg_step_time = step_time

            # Estimate samples processed (approximate)
            self.samples_processed = state.global_step * args.per_device_train_batch_size * args.gradient_accumulation_steps
            samples_per_sec = args.per_device_train_batch_size * args.gradient_accumulation_steps / avg_step_time if avg_step_time > 0 else 0

            # Memory usage
            memory_info = psutil.Process().memory_info()

            self.last_log_time = current_time

            # Trigger garbage collection periodically to prevent memory buildup
            if state.global_step % 20 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and 'loss' in logs:
            # Enhanced loss logging with additional context
            current_time = time.time()
            total_time = current_time - self.start_time
            logger.info(f"Training metrics at step {state.global_step}: loss={logs['loss']:.4f}, "
                        f"learning_rate={logs.get('learning_rate', 'N/A')}, "
                        f"total_time={total_time / 60:.1f}min")


# --- Main Training Function ---
CHATTERBOX_PROJECT = "./chatterbox-project"


def run_training(model_args, data_args, training_args, is_local=False):
    # Optional: Login to HuggingFace if needed for model downloads
    # from huggingface_hub import login
    # login(token="your_hf_token_here")  # Replace with your token or use environment variable

    # Enable PyTorch profiler if requested
    use_torch_profiler = getattr(training_args, 'use_torch_profiler', False)
    profiler_output_dir = os.path.join(CHATTERBOX_PROJECT, "profiler_output")

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

    # Ensure all Trainer checkpoints and outputs go to the project directory
    output_dir = training_args.output_dir
    training_args.output_dir = os.path.join(CHATTERBOX_PROJECT, output_dir)
    os.makedirs(training_args.output_dir, exist_ok=True)

    # Auto-detect and resume from the last checkpoint if not explicitly provided
    if training_args.resume_from_checkpoint is None:
        torch.serialization.add_safe_globals([np.core.multiarray._reconstruct])
        last_ckpt = get_last_checkpoint(training_args.output_dir)
        if last_ckpt:
            training_args.resume_from_checkpoint = last_ckpt
            logger.info(f"Found existing checkpoint, resuming from: {last_ckpt}")

    global trainer_instance

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
    repo_home_weights = os.path.join(CHATTERBOX_PROJECT, "chatterbox_weights")

    # Model loading logic
    if model_args.model_config:
        logger.info(f"Loading model from model config file: {model_args.model_config}")
        with open(model_args.model_config, "r") as file:
            m_paths = json.load(file)
        repo_name = "ResembleAI/chatterbox"

        # Add progress bar for model download
        with tqdm(desc="Downloading model from HuggingFace", unit="B", unit_scale=True) as pbar:
            snapshot_download(repo_name, local_dir_use_symlinks=False, local_dir=repo_home_weights,
                              token=os.getenv("HF_TOKEN"))
            pbar.update(1)
            pbar.set_description("Download completed")

        # Helper function to resolve paths correctly
        def resolve_model_path(path_str):
            """Resolve model path, avoiding duplicate chatterbox-project"""
            if os.path.isabs(path_str):
                # Absolute path, use as-is
                return Path(path_str)
            elif path_str.startswith("chatterbox-project/"):
                # Already has chatterbox-project prefix, use as-is
                return Path(path_str)
            else:
                # Relative path, add CHATTERBOX_PROJECT
                return Path(CHATTERBOX_PROJECT) / path_str

        # Add progress bar for model loading
        with tqdm(desc="Loading ChatterboxTTS components", total=4) as pbar:
            pbar.set_description("Loading voice encoder...")
            voice_encoder_path = resolve_model_path(m_paths["voice_encoder_path"])
            pbar.update(1)

            pbar.set_description("Loading T3 model...")
            t3_path = resolve_model_path(m_paths["t3_path"])
            pbar.update(1)

            pbar.set_description("Loading S3Gen model...")
            s3gen_path = resolve_model_path(m_paths["s3gen_path"])
            pbar.update(1)

            pbar.set_description("Initializing complete model...")
            chatterbox_model = ChatterboxTTS.from_specified(
                voice_encoder_path=voice_encoder_path,
                t3_path=t3_path,
                s3gen_path=s3gen_path,
                tokenizer_path=m_paths["tokenizer_path"],
                conds_path=resolve_model_path(m_paths["conds_path"]),
                device="cpu"
            )
            pbar.update(1)
            pbar.set_description("Model loading completed")

        original_model_dir_for_copy = repo_home_weights
    elif model_args.local_model_dir:
        logger.info(f"Loading model from local directory: {model_args.local_model_dir}")
        local_dir_path = Path(model_args.local_model_dir)

        with tqdm(desc="Loading local ChatterboxTTS model", total=1) as pbar:
            chatterbox_model = ChatterboxTTS.from_local(ckpt_dir=str(local_dir_path), device="cpu")
            pbar.update(1)
            pbar.set_description("Local model loading completed")

        original_model_dir_for_copy = local_dir_path
    else:
        repo_to_download = model_args.model_name_or_path or REPO_ID
        logger.info(f"Loading model from Hugging Face Hub: {repo_to_download}")
        download_dir = Path(CHATTERBOX_PROJECT) / "pretrained_model_download"
        download_dir.mkdir(parents=True, exist_ok=True)
        files_to_download = ["ve.safetensors", "t3_cfg.safetensors", "s3gen.safetensors", "tokenizer.json"]

        from huggingface_hub import hf_hub_download as hf_download

        # Add progress bar for file downloads
        with tqdm(desc="Downloading model files", total=len(files_to_download) + 1) as pbar:
            for f in files_to_download:
                try:
                    pbar.set_description(f"Downloading {f}")
                    hf_download(repo_id=repo_to_download, filename=f, local_dir=download_dir,
                                local_dir_use_symlinks=False, cache_dir=model_args.cache_dir)
                    pbar.update(1)
                except Exception as e:
                    logger.warning(f"Could not download {f} from {repo_to_download}: {e}.")
                    pbar.update(1)

            try:
                pbar.set_description("Downloading conds.pt")
                hf_download(repo_id=repo_to_download, filename="conds.pt", local_dir=download_dir,
                            local_dir_use_symlinks=False, cache_dir=model_args.cache_dir)
            except:
                logger.info("conds.pt not found on Hub or failed to download for this model.")
            pbar.update(1)
            pbar.set_description("All downloads completed")

        with tqdm(desc="Loading downloaded model", total=1) as pbar:
            chatterbox_model = ChatterboxTTS.from_local(ckpt_dir=download_dir, device="cpu")
            pbar.update(1)
            pbar.set_description("Model loading completed")

        original_model_dir_for_copy = download_dir

    t3_model = chatterbox_model.t3
    chatterbox_t3_config_instance = t3_model.hp

    # Freeze components as specified
    if model_args.freeze_voice_encoder:
        for param in chatterbox_model.ve.parameters(): param.requires_grad = False
        logger.info("Voice Encoder frozen.")
    if model_args.freeze_s3gen:
        for param in chatterbox_model.s3gen.parameters(): param.requires_grad = False
        logger.info("S3Gen model frozen.")
    for param in t3_model.parameters(): param.requires_grad = True

    logger.info("T3 model set to trainable.")
    logger.info("Loading and processing dataset...")
    verification_mode = VerificationMode.NO_CHECKS if data_args.ignore_verifications else VerificationMode.BASIC_CHECKS

    train_hf_dataset: Union[datasets.Dataset, List[Dict[str, str]]]
    eval_hf_dataset: Optional[Union[datasets.Dataset, List[Dict[str, str]]]] = None
    streaming = None

    # Dataset loading logic (simplified for local training)
    if data_args.train_csv:
        # CSV dataset loading
        logger.info(f"Loading CSV dataset from {data_args.train_csv}")

        # Validate CSV file exists
        if not os.path.exists(data_args.train_csv):
            raise FileNotFoundError(f"Training CSV file not found: {data_args.train_csv}")

        # Create datasets using CSV
        logger.info("Creating training dataset from CSV...")
        train_dataset = CSVSpeechDataset(
            csv_file=data_args.train_csv,
            data_args=data_args,
            t3_config=chatterbox_t3_config_instance,
            model_dir=str(original_model_dir_for_copy) if original_model_dir_for_copy else CHATTERBOX_PROJECT,
            m_paths=m_paths if model_args.model_config else None,
            device="cpu"
        )

        eval_dataset = None
        if training_args.do_eval and data_args.val_csv:
            if os.path.exists(data_args.val_csv):
                logger.info(f"Creating evaluation dataset from {data_args.val_csv}")
                eval_dataset = CSVSpeechDataset(
                    csv_file=data_args.val_csv,
                    data_args=data_args,
                    t3_config=chatterbox_t3_config_instance,
                    model_dir=str(original_model_dir_for_copy) if original_model_dir_for_copy else CHATTERBOX_PROJECT,
                    m_paths=m_paths if model_args.model_config else None,
                    device="cpu"
                )
            else:
                logger.warning(f"Validation CSV file not found: {data_args.val_csv}. Skipping evaluation.")
        elif training_args.do_eval:
            logger.warning("Evaluation requested but no val_csv provided. Skipping evaluation.")

        logger.info(f"Training dataset size: {len(train_dataset)}")
        if eval_dataset:
            logger.info(f"Evaluation dataset size: {len(eval_dataset)}")

        # Continue with training setup
        logger.info("CSV datasets created successfully!")

    elif data_args.dataset_name:
        logger.info(f"Loading dataset '{data_args.dataset_name}' from Hugging Face Hub.")

        # For local training, we'll load the dataset normally (not streaming)
        ds_logging.set_verbosity_info()
        ds_logging.enable_progress_bar()

        download_config = DownloadConfig()

        with tqdm(desc="Loading dataset", total=1) as pbar:
            pbar.set_description("Loading local dataset...")
            raw_datasets_loaded = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                cache_dir=CHATTERBOX_PROJECT,
                num_proc=32,
                download_config=download_config,
                verification_mode=verification_mode,
            )
            pbar.update(1)
            pbar.set_description("Dataset loading completed")

        logger.info("Dataset loaded.")

        if data_args.train_split_name not in raw_datasets_loaded:
            raise ValueError(
                f"Train split '{data_args.train_split_name}' not found. Available: {list(raw_datasets_loaded.keys())}")
        else:
            train_hf_dataset = raw_datasets_loaded[data_args.train_split_name]

        if training_args.do_eval:
            with tqdm(desc="Setting up evaluation dataset", total=1) as pbar:
                if data_args.eval_split_name and data_args.eval_split_name in raw_datasets_loaded:
                    eval_hf_dataset = raw_datasets_loaded[data_args.eval_split_name]
                elif "validation" in raw_datasets_loaded:
                    eval_hf_dataset = raw_datasets_loaded["validation"]
                elif "test" in raw_datasets_loaded:
                    eval_hf_dataset = raw_datasets_loaded["test"]
                elif data_args.eval_split_size > 0 and hasattr(train_hf_dataset, "__len__") and len(
                        train_hf_dataset) > 1:
                    pbar.set_description("Splitting train dataset for evaluation...")
                    logger.info(f"Splitting train dataset for evaluation with ratio {data_args.eval_split_size}")
                    split_dataset = train_hf_dataset.train_test_split(test_size=data_args.eval_split_size,
                                                                      seed=training_args.seed)
                    train_hf_dataset, eval_hf_dataset = split_dataset["train"], split_dataset["test"]
                    logger.info(f"Evaluation set size: {len(eval_hf_dataset)}")
                else:
                    logger.warning(
                        "Evaluation requested but no eval split found/configured or train dataset too small to split. Skipping eval dataset.")
                pbar.update(1)
                pbar.set_description("Evaluation dataset setup completed")

        is_hf_format_train, is_hf_format_eval = True, True
    else:
        # Local dataset processing
        def load_json_dataset_files(dataset_dir: str) -> List[Dict[str, str]]:
            dataset_path = Path(dataset_dir)
            json_files = list(dataset_path.glob("**/*.json"))
            files = []

            with tqdm(desc=f"Loading JSON dataset from {dataset_path.name}", total=len(json_files)) as pbar:
                for json_file in json_files:
                    try:
                        with open(json_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)

                        text = data.get("text", "").strip()
                        audio_filename = json_file.stem + ".mp3"
                        audio_path = json_file.parent / audio_filename

                        if audio_path.exists() and text:
                            files.append({
                                "audio": str(audio_path),
                                "text": text,
                                "language": data.get("language", "unknown"),
                                "speaker": data.get("speaker", "unknown"),
                                "duration": data.get("duration", 0.0)
                            })
                        else:
                            if not audio_path.exists():
                                logger.warning(f"Audio file not found for {json_file}: {audio_path}")
                            if not text:
                                logger.warning(f"Empty text in {json_file}")
                    except Exception as e:
                        logger.warning(f"Error loading {json_file}: {e}")
                    pbar.update(1)

            return files

        if data_args.dataset_dirs:
            # Multiple dataset directories
            all_files = []
            for dataset_dir in data_args.dataset_dirs:
                logger.info(f"Loading dataset from directory: {dataset_dir}")
                files = load_json_dataset_files(dataset_dir)
                all_files.extend(files)
                logger.info(f"Loaded {len(files)} files from {dataset_dir}")
            train_hf_dataset = all_files
        elif data_args.dataset_dir:
            # Single dataset directory
            logger.info(f"Loading dataset from directory: {data_args.dataset_dir}")
            train_hf_dataset = load_json_dataset_files(data_args.dataset_dir)
        else:
            raise ValueError("Either dataset_name or dataset_dir/dataset_dirs must be provided")

        # Split for evaluation if needed
        if training_args.do_eval and data_args.eval_split_size > 0 and len(train_hf_dataset) > 1:
            import random
            random.seed(training_args.seed)
            random.shuffle(train_hf_dataset)
            split_idx = int(len(train_hf_dataset) * (1 - data_args.eval_split_size))
            eval_hf_dataset = train_hf_dataset[split_idx:]
            train_hf_dataset = train_hf_dataset[:split_idx]
            logger.info(f"Split dataset: {len(train_hf_dataset)} train, {len(eval_hf_dataset)} eval")

        is_hf_format_train, is_hf_format_eval = False, False

    # For non-CSV datasets, create SpeechFineTuningDataset
    if not data_args.train_csv:
        logger.info(
            f"Training dataset size: {len(train_hf_dataset) if hasattr(train_hf_dataset, '__len__') else 'streaming'}")
        if eval_hf_dataset:
            logger.info(
                f"Evaluation dataset size: {len(eval_hf_dataset) if hasattr(eval_hf_dataset, '__len__') else 'streaming'}")

        # Create datasets
        logger.info("Creating training dataset...")
        train_dataset = SpeechFineTuningDataset(
            data_args=data_args,
            t3_config=chatterbox_t3_config_instance,
            hf_dataset=train_hf_dataset,
            is_hf_format=is_hf_format_train,
            model_dir=str(original_model_dir_for_copy) if original_model_dir_for_copy else CHATTERBOX_PROJECT,
            m_paths=m_paths if model_args.model_config else None,
            device="cpu"
        )

        eval_dataset = None
        if eval_hf_dataset:
            logger.info("Creating evaluation dataset...")
            eval_dataset = SpeechFineTuningDataset(
                data_args=data_args,
                t3_config=chatterbox_t3_config_instance,
                hf_dataset=eval_hf_dataset,
                is_hf_format=is_hf_format_eval,
                model_dir=str(original_model_dir_for_copy) if original_model_dir_for_copy else CHATTERBOX_PROJECT,
                m_paths=m_paths if model_args.model_config else None,
                device="cpu"
            )

    logger.info("Training dataset created successfully!")

    # Setup Trainer and start training
    logger.info("Setting up Trainer...")

    # Add progress callback
    callbacks = [DetailedProgressCallback()]

    # Create Trainer
    from transformers import Trainer

    # Wrap T3 model for HuggingFace Trainer compatibility
    wrapped_model = ChatterboxT3Wrapper(t3_model)

    # Ensure dataloader settings are valid and respect CLI
    # - prefetch_factor must be an int when num_workers > 0, else None
    # - persistent_workers only makes sense when num_workers > 0
    if getattr(training_args, "dataloader_num_workers", None) is None:
        # Default to a conservative value on Windows
        try:
            cpu_cnt = os.cpu_count() or 4
        except Exception:
            cpu_cnt = 4
        training_args.dataloader_num_workers = min(4, max(2, cpu_cnt // 4))

    if training_args.dataloader_num_workers and training_args.dataloader_num_workers > 0:
        # If user didn't pass a value, set a sane default
        if getattr(training_args, "dataloader_prefetch_factor", None) in (None, True, False):
            training_args.dataloader_prefetch_factor = 2
        training_args.dataloader_persistent_workers = True
    else:
        training_args.dataloader_prefetch_factor = None
        training_args.dataloader_persistent_workers = False

    trainer = Trainer(
        model=wrapped_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=speech_collate_fn,
        callbacks=callbacks,
    )

    logger.info("Starting training...")

    # Start training
    try:
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        logger.info("Training completed successfully!")

        # Save final model
        trainer.save_model()
        logger.info(f"Model saved to {training_args.output_dir}")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


def main():
    debug = False
    if debug:
        # Set debug parameters directly instead of parsing from command line
        model_args = ModelArguments(
            model_config="model_path.json",
            cache_dir=None,
            freeze_voice_encoder=True,
            freeze_s3gen=True,
            freeze_text_embeddings=704
        )

        data_args = DataArguments(
            train_csv="train.csv",
            val_csv="val.csv",
            eval_split_size=0.0002,
            preprocessing_num_workers=4,
            text_column_name="transcript",
            audio_column_name="audio",
            max_text_len=256,
            max_speech_len=800,
            audio_prompt_duration_s=3.0,
            ignore_verifications=False
        )

        training_args = CustomTrainingArguments(
            output_dir="checkpoints/vietnamese_run",
            num_train_epochs=3,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=2,
            learning_rate=5e-5,
            warmup_steps=500,
            logging_steps=50,
            eval_strategy="steps",
            eval_steps=2000,
            save_strategy="steps",
            save_steps=2000,
            save_total_limit=3,
            fp16=True,
            report_to="tensorboard",
            dataloader_num_workers=4,
            do_train=True,
            do_eval=True,
            dataloader_pin_memory=True if torch.cuda.is_available() else False,
            eval_on_start=False,
            use_torch_profiler=False,
            dataloader_persistent_workers=True,
            dataloader_prefetch_factor=8
        )

        # Use preprocessing_num_workers as dataloader_num_workers if set
        if data_args.preprocessing_num_workers is not None:
            training_args.dataloader_num_workers = data_args.preprocessing_num_workers
    else:
        parser = HfArgumentParser((ModelArguments, DataArguments, CustomTrainingArguments))
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

        # Use preprocessing_num_workers as dataloader_num_workers if set
        if data_args.preprocessing_num_workers is not None:
            training_args.dataloader_num_workers = data_args.preprocessing_num_workers

    print("Running training locally...")
    run_training(model_args, data_args, training_args, is_local=True)


if __name__ == "__main__":
    main()
