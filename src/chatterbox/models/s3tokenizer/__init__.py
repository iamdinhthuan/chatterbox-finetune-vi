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
    if SOS in x:
        indices = (x == SOS).nonzero(as_tuple=True)[0]
        if len(indices) > 0:
            # Use item() to convert 0-d tensor to Python int
            s = indices[0].item() + 1 if indices[0].numel() == 1 else int(indices[0]) + 1
        else:
            s = 0
    else:
        s = 0

    if EOS in x:
        indices = (x == EOS).nonzero(as_tuple=True)[0]
        if len(indices) > 0:
            # Use item() to convert 0-d tensor to Python int
            e = indices[0].item() if indices[0].numel() == 1 else int(indices[0])
        else:
            e = None
    else:
        e = None

    x = x[s: e]
    return x
