from .s3tokenizer import (
    S3_SR,
    S3_HOP,
    S3_TOKEN_HOP,
    S3_TOKEN_RATE,
    SPEECH_VOCAB_SIZE,
    S3Tokenizer,
)


SOS = SPEECH_VOCAB_SIZE
EOS = SPEECH_VOCAB_SIZE + 1



def drop_invalid_tokens(x):
    """Drop SoS and EoS - but be more conservative for Vietnamese"""
    assert len(x.shape) == 1 or (len(x.shape) == 2 and x.shape[0] == 1), "only batch size of one allowed for now"
    
    import torch
    import logging
    
    # Flatten if needed (for cross-platform compatibility)
    if len(x.shape) == 2:
        x = x.squeeze(0)
    logging.debug(f"drop_invalid_tokens input shape: {x.shape}, len: {len(x)}")
    
    # First, filter out all SOS and EOS tokens (they should not be in speech tokens)
    mask = (x != SOS) & (x != EOS)
    result = x[mask]
    logging.debug(f"After filtering SOS/EOS: {len(result)} tokens (from {len(x)} tokens)")
    
    # Validate token range - should be 0 to SPEECH_VOCAB_SIZE-1
    if len(result) > 0:
        min_val = result.min().item()
        max_val = result.max().item()
        if min_val < 0 or max_val >= SPEECH_VOCAB_SIZE:
            logging.error(f"Invalid token values after SOS/EOS removal: min={min_val}, max={max_val}, vocab_size={SPEECH_VOCAB_SIZE}")
            # Clip to valid range as last resort
            result = torch.clamp(result, min=0, max=SPEECH_VOCAB_SIZE-1)
            logging.warning(f"Clipped tokens to valid range [0, {SPEECH_VOCAB_SIZE-1}]")
    
    return result
