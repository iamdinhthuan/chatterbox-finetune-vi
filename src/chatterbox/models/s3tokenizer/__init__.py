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
    """Drop SoS and EoS"""
    assert len(x.shape) == 1 or (len(x.shape) == 2 and x.shape[0] == 1), "only batch size of one allowed for now"
    
    # Debug logging
    import logging
    logging.debug(f"drop_invalid_tokens input shape: {x.shape}, len: {len(x)}")
    
    if SOS in x:
        indices = (x == SOS).nonzero(as_tuple=True)[0]
        if len(indices) > 0:
            # Safely convert to Python int
            if indices.ndim == 0:  # scalar tensor
                s = int(indices.item()) + 1
            else:  # 1D tensor
                s = int(indices[0].item()) + 1
            logging.debug(f"Found SOS at index {s-1}, starting from {s}")
        else:
            s = 0
    else:
        s = 0
        logging.debug("No SOS found, starting from 0")

    if EOS in x:
        indices = (x == EOS).nonzero(as_tuple=True)[0]
        if len(indices) > 0:
            # Safely convert to Python int  
            if indices.ndim == 0:  # scalar tensor
                e = int(indices.item())
            else:  # 1D tensor
                e = int(indices[0].item())
            logging.debug(f"Found EOS at index {e}")
        else:
            e = None
            logging.debug("No EOS found, using full length")
    else:
        e = None
        logging.debug("No EOS found, using full length")

    result = x[s: e]
    logging.debug(f"drop_invalid_tokens output len: {len(result)} (from {len(x)} tokens)")
    
    return result
