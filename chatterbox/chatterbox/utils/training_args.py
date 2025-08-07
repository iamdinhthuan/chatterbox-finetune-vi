from typing import Optional
from dataclasses import dataclass, field
from transformers.training_args import TrainingArguments as HfTrainingArguments

# --- Custom Training Arguments ---
@dataclass
class CustomTrainingArguments(HfTrainingArguments):
    early_stopping_patience: Optional[int] = field(
        default=None, metadata={"help": "Enable early stopping with specified patience. Default: None (disabled)."}
    )
    use_torch_profiler: bool = field(
        default=False, metadata={"help": "Enable PyTorch profiler and dump traces to TensorBoard."}
    )
    dataloader_persistent_workers: bool = field(
        default=True, metadata={"help": "Use persistent workers for the dataloader."}
    )
    