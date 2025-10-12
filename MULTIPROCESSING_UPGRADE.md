# Multiprocessing Upgrade for Preprocessing

## ❌ Current Status: SINGLE-THREADED

**Speed:** ~32 it/s  
**Time for 2.6M samples:** ~22 hours

## ✅ With Multiprocessing: 4-10x FASTER

| Workers | Expected Speed | Time | Speedup |
|---------|---------------|------|---------|
| 1 (current) | ~32 it/s | 22 hours | 1x |
| **4 workers** | ~120 it/s | **6 hours** | **3.7x** |
| **8 workers** | ~220 it/s | **3.3 hours** | **6.7x** |
| **16 workers** | ~350 it/s | **2.1 hours** | **10.5x** |

---

## 🔧 Implementation Required

File đã được modified externally (silence_padding 300→500ms).  
Cần add multiprocessing support manually:

### Step 1: Add Imports (top of file)

```python
import torch.multiprocessing as mp
from multiprocessing import Queue
```

### Step 2: Add Worker Function (before `main()`)

```python
def worker_process(
    worker_id: int,
    samples_queue: Queue,
    result_queue: Queue,
    model_dir: Path,
    tokenizer_path: Path,
    args
):
    """
    Worker process for parallel preprocessing
    Each worker loads its own models to avoid CUDA issues
    """
    try:
        # Load models for this worker
        device = f"cuda:{worker_id % torch.cuda.device_count()}" if torch.cuda.is_available() else "cpu"
        
        from tokenizers import Tokenizer
        text_tokenizer = Tokenizer.from_file(str(tokenizer_path))
        
        # Load ChatterboxTTS
        tts = ChatterboxTTS.from_local(ckpt_dir=str(model_dir), device=device)
        speech_tokenizer = tts.s3gen.tokenizer
        voice_encoder = tts.ve
        voice_encoder.eval()
        t3_config = tts.t3.hp
        
        # Process samples from queue
        while True:
            item = samples_queue.get()
            if item is None:  # Poison pill
                break
            
            idx, sample = item
            
            try:
                preprocessed = preprocess_sample(
                    audio_path=sample["audio_path"],
                    text=sample["text"],
                    text_tokenizer=text_tokenizer,
                    speech_tokenizer=speech_tokenizer,
                    voice_encoder=voice_encoder,
                    t3_config=t3_config,
                    max_text_len=args.max_text_len,
                    max_speech_len=args.max_speech_len,
                    audio_prompt_duration_s=args.audio_prompt_duration,
                    add_silence=args.add_silence,
                    silence_padding_ms=args.silence_padding_ms,
                )
                
                result_queue.put((idx, preprocessed, sample))
                
            except Exception as e:
                result_queue.put((idx, None, sample))
                
    except Exception as e:
        print(f"Worker {worker_id} error: {e}")
```

### Step 3: Add --num_workers Argument

In `parser.add_argument()` section:

```python
parser.add_argument("--num_workers", type=int, default=1, 
                    help="Number of parallel workers (default: 1, recommend: 4-8)")
```

### Step 4: Replace Processing Loop

Replace the existing `for idx, sample in enumerate(tqdm(samples...` section with:

```python
    # Preprocess all samples
    logger.info("\nPreprocessing samples...")
    logger.info("This will take a while, but only needs to be done once!")
    
    num_workers = args.num_workers
    logger.info(f"Using {num_workers} workers for parallel processing")
    
    successful = 0
    failed = 0
    sample_list = []
    
    if num_workers == 1:
        # Single-threaded (original code)
        for idx, sample in enumerate(tqdm(samples, desc="Preprocessing")):
            output_file = output_dir / f"sample_{idx:06d}.pt"
            
            if output_file.exists():
                successful += 1
                sample_list.append({
                    "idx": idx,
                    "pt_file": f"sample_{idx:06d}.pt",
                    "audio_path": str(sample["audio_path"]),
                    "text": sample["text"]
                })
                continue
            
            preprocessed = preprocess_sample(
                audio_path=sample["audio_path"],
                text=sample["text"],
                text_tokenizer=text_tokenizer,
                speech_tokenizer=speech_tokenizer,
                voice_encoder=voice_encoder,
                t3_config=t3_config,
                max_text_len=args.max_text_len,
                max_speech_len=args.max_speech_len,
                audio_prompt_duration_s=args.audio_prompt_duration,
                add_silence=args.add_silence,
                silence_padding_ms=args.silence_padding_ms,
            )
            
            if preprocessed is not None:
                torch.save(preprocessed, output_file)
                successful += 1
                sample_list.append({
                    "idx": idx,
                    "pt_file": f"sample_{idx:06d}.pt",
                    "audio_path": str(sample["audio_path"]),
                    "text": sample["text"]
                })
            else:
                failed += 1
    
    else:
        # Multi-processing
        mp.set_start_method('spawn', force=True)
        
        # Create queues
        samples_queue = Queue(maxsize=num_workers * 2)
        result_queue = Queue()
        
        # Start workers
        workers = []
        for worker_id in range(num_workers):
            p = mp.Process(
                target=worker_process,
                args=(worker_id, samples_queue, result_queue, model_dir, tokenizer_path, args)
            )
            p.start()
            workers.append(p)
        
        # Filter unprocessed samples
        unprocessed_samples = []
        for idx, sample in enumerate(samples):
            output_file = output_dir / f"sample_{idx:06d}.pt"
            if output_file.exists():
                successful += 1
                sample_list.append({
                    "idx": idx,
                    "pt_file": f"sample_{idx:06d}.pt",
                    "audio_path": str(sample["audio_path"]),
                    "text": sample["text"]
                })
            else:
                unprocessed_samples.append((idx, sample))
        
        # Enqueue unprocessed samples
        for item in unprocessed_samples:
            samples_queue.put(item)
        
        # Send poison pills
        for _ in range(num_workers):
            samples_queue.put(None)
        
        # Collect results with progress bar
        pbar = tqdm(total=len(unprocessed_samples), desc="Preprocessing")
        processed_count = 0
        
        while processed_count < len(unprocessed_samples):
            idx, preprocessed, sample = result_queue.get()
            
            output_file = output_dir / f"sample_{idx:06d}.pt"
            
            if preprocessed is not None:
                torch.save(preprocessed, output_file)
                successful += 1
                sample_list.append({
                    "idx": idx,
                    "pt_file": f"sample_{idx:06d}.pt",
                    "audio_path": str(sample["audio_path"]),
                    "text": sample["text"]
                })
            else:
                failed += 1
            
            processed_count += 1
            pbar.update(1)
        
        pbar.close()
        
        # Wait for workers
        for p in workers:
            p.join()
        
        logger.info(f"All workers finished")
```

---

## 🚀 Usage After Implementation

### Single-threaded (current, 22 hours):
```bash
python3 preprocess_dataset.py \
  --csv metadata.csv \
  --audio_dir wavs \
  --add_silence
```

### Multi-threaded (4x faster, ~6 hours):
```bash
python3 preprocess_dataset.py \
  --csv metadata.csv \
  --audio_dir wavs \
  --add_silence \
  --num_workers 4
```

### Multi-threaded (8x faster, ~3 hours):
```bash
python3 preprocess_dataset.py \
  --csv metadata.csv \
  --audio_dir wavs \
  --add_silence \
  --num_workers 8
```

---

## 💡 Worker Count Recommendations

**Based on your system:**

| CPU Cores | GPU | Recommended Workers | Expected Time |
|-----------|-----|---------------------|---------------|
| 4-8 cores | 1 GPU | 4 workers | ~6 hours |
| 8-16 cores | 1 GPU | 8 workers | ~3.3 hours |
| 16+ cores | 2+ GPUs | 16 workers | ~2.1 hours |

**Check your system:**
```bash
# Check CPU cores
nproc

# Check GPUs
nvidia-smi
```

**Rule of thumb:**
- Start with `num_workers = CPU cores / 2`
- Max useful: `num_workers = CPU cores`
- More workers = more memory usage

---

## ⚠️ Important Notes

### Memory Usage:
- Each worker loads full model (~4GB per worker)
- 8 workers = ~32GB RAM usage
- Monitor with: `watch -n 1 'nvidia-smi; free -h'`

### GPU Distribution:
- Workers auto-distribute across GPUs
- `worker_id % torch.cuda.device_count()`
- 8 workers + 2 GPUs = 4 workers per GPU

### Resume Support:
- ✅ Already skips existing .pt files
- ✅ Safe to stop/restart
- ✅ Progress preserved

---

## 🎯 Implementation Priority

**Option A: Implement Now (Recommended)**
- Add multiprocessing code manually
- Run with `--num_workers 8`
- Finish in ~3 hours instead of 22

**Option B: Run Single-threaded First**
- Keep current code
- Run without `--num_workers`
- Takes 22 hours but simpler

**My recommendation: IMPLEMENT NOW!**
- 19 hours saved
- Code is ready above
- Just copy-paste into file

---

## 📝 Quick Implementation Checklist

- [ ] Add imports (`torch.multiprocessing`, `Queue`)
- [ ] Add `worker_process()` function
- [ ] Add `--num_workers` argument
- [ ] Replace processing loop with conditional (single vs multi)
- [ ] Test with 1 sample: `python test_preprocessing_single.py`
- [ ] Run full: `python preprocess_dataset.py ... --num_workers 8`

---

## 🎉 Expected Results

**Before:**
```
Preprocessing: 100%|██| 2604620/2604620 [22:03:15, 32.81it/s]
```

**After (8 workers):**
```
Preprocessing: 100%|██| 2604620/2604620 [03:18:30, 218.45it/s]
```

**Time saved: 19 hours!** ⚡
