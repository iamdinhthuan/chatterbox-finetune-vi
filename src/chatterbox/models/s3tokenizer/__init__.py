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
    
    # Flatten if needed (for cross-platform compatibility)
    if len(x.shape) == 2:
        x = x.squeeze(0)
    
    # Debug logging
    import logging
    logging.debug(f"drop_invalid_tokens input shape: {x.shape}, len: {len(x)}")
    
    # For Vietnamese model, we should be more conservative
    # Only drop SOS at the beginning if it exists
    if len(x) > 0 and x[0] == SOS:
        s = 1
        logging.debug(f"Found SOS at beginning, starting from index 1")
    else:
        s = 0
        logging.debug("No SOS at beginning, starting from 0")
    
    # For EOS, only cut if it's at the end or very close to end
    # Don't cut if EOS appears too early (might be a false positive)
    if EOS in x:
        indices = (x == EOS).nonzero(as_tuple=True)[0]
        if len(indices) > 0:
            # Get the last occurrence of EOS
            if indices.ndim == 0:
                last_eos = int(indices.item())
            else:
                last_eos = int(indices[-1].item())  # Take last EOS
            
            # Only use EOS if it's not too early (at least 10 tokens)
            if last_eos > 10:
                e = last_eos
                logging.debug(f"Found EOS at index {e}, using it")
            else:
                e = None
                logging.debug(f"Found EOS too early at index {last_eos}, ignoring it")
        else:
            e = None
            logging.debug("No EOS found")
    else:
        e = None
        logging.debug("No EOS in tensor")

    result = x[s: e]
    logging.debug(f"drop_invalid_tokens output len: {len(result)} (from {len(x)} tokens)")
    
    # Safety check: if result is too short, return original
    if len(result) < 5 and len(x) > 10:
        logging.warning(f"Result too short ({len(result)}), returning original minus SOS/EOS tokens")
        # Just remove SOS/EOS tokens without position-based cutting
        mask = (x != SOS) & (x != EOS)
        result = x[mask]
        logging.debug(f"After filtering SOS/EOS: {len(result)} tokens")
    
    return result
