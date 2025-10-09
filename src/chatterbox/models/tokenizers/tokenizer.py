import logging
import os

import torch
from tokenizers import Tokenizer


# Special tokens
SOT = "[START]"
EOT = "[STOP]"
UNK = "[UNK]"
SPACE = "[SPACE]"
SPECIAL_TOKENS = [SOT, EOT, UNK, SPACE, "[PAD]", "[SEP]", "[CLS]", "[MASK]"]

logger = logging.getLogger(__name__)

class EnTokenizer:
    def __init__(self, vocab_file_path):
        # Import at the top to avoid UnboundLocalError
        from tokenizers import Tokenizer
        
        try:
            self.tokenizer = Tokenizer.from_file(vocab_file_path)
        except Exception as e:
            # If loading fails, we need to manually fix the tokenizer file
            logger.warning(f"Failed to load tokenizer normally: {e}")
            logger.info("Attempting to fix and reload tokenizer...")
            
            import json
            import tempfile
            import shutil
            
            # Load the tokenizer JSON
            with open(vocab_file_path, 'r', encoding='utf-8') as f:
                tokenizer_json = json.load(f)
            
            # The issue is with the 'added_tokens' section having wrong IDs
            # We need to fix the special token IDs to match the vocab
            if 'added_tokens' in tokenizer_json:
                # Fix the special token IDs based on the vocab
                vocab = tokenizer_json['model']['vocab']
                fixed_added_tokens = []
                
                for token_info in tokenizer_json['added_tokens']:
                    content = token_info['content']
                    # Get the correct ID from vocab
                    if content in vocab:
                        correct_id = vocab[content]
                        token_info['id'] = correct_id
                        fixed_added_tokens.append(token_info)
                
                tokenizer_json['added_tokens'] = fixed_added_tokens
            
            # Create a temporary file with the fixed tokenizer
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as tmp_file:
                json.dump(tokenizer_json, tmp_file, ensure_ascii=False, indent=2)
                tmp_path = tmp_file.name
            
            try:
                # Try loading the fixed tokenizer
                self.tokenizer = Tokenizer.from_file(tmp_path)
                logger.info("Successfully loaded fixed tokenizer")
            except Exception as e2:
                logger.error(f"Failed to load fixed tokenizer: {e2}")
                # Fall back to creating a simple BPE tokenizer manually
                from tokenizers import models, pre_tokenizers
                from tokenizers.models import BPE
                
                # Get vocab and merges
                vocab = tokenizer_json['model']['vocab'] 
                merges = tokenizer_json['model'].get('merges', [])
                
                # Create empty BPE model first
                tokenizer = Tokenizer(BPE())
                
                # Set the vocab directly
                tokenizer.model = BPE(vocab, merges, unk_token="[UNK]")
                
                # Set pre-tokenizer 
                from tokenizers.pre_tokenizers import Whitespace
                tokenizer.pre_tokenizer = Whitespace()
                
                # Add special tokens
                if 'added_tokens' in tokenizer_json:
                    from tokenizers import AddedToken
                    for token_info in tokenizer_json['added_tokens']:
                        content = token_info['content']
                        if content in vocab:
                            tokenizer.add_tokens([content])
                
                self.tokenizer = tokenizer
                logger.info("Created fallback BPE tokenizer")
            finally:
                # Clean up temp file
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
        
        self.check_vocabset_sot_eot()

    def check_vocabset_sot_eot(self):
        voc = self.tokenizer.get_vocab()
        assert SOT in voc
        assert EOT in voc

    def text_to_tokens(self, text: str):
        text_tokens = self.encode(text)
        text_tokens = torch.IntTensor(text_tokens).unsqueeze(0)
        return text_tokens

    def encode( self, txt: str, verbose=False):
        """
        clean_text > (append `lang_id`) > replace SPACE > encode text using Tokenizer
        """
        txt = txt.replace(' ', SPACE)
        code = self.tokenizer.encode(txt)
        ids = code.ids
        return ids

    def decode(self, seq):
        if isinstance(seq, torch.Tensor):
            seq = seq.cpu().numpy()

        txt: str = self.tokenizer.decode(seq,
        skip_special_tokens=False)
        txt = txt.replace(' ', '')
        txt = txt.replace(SPACE, ' ')
        txt = txt.replace(EOT, '')
        txt = txt.replace(UNK, '')
        return txt
