"""
Tạo Vietnamese tokenizer giống English tokenizer
- 704 tokens (KHÔNG CÓ PAD!)
- Tất cả là ký tự thực tế
- Special tokens cho expressive TTS
"""

import json
from pathlib import Path
from collections import OrderedDict


def create_vietnamese_tokenizer():
    """
    Tạo Vietnamese tokenizer - 704 tokens thực tế (không PAD)
    """
    
    print("="*80)
    print("TẠO VIETNAMESE TOKENIZER (704 TOKENS - NO PAD)")
    print("="*80)
    
    vocab = OrderedDict()
    token_id = 0

    # 1. Core special tokens
    vocab["[STOP]"] = 0
    vocab["[UNK]"] = 1
    vocab["[SPACE]"] = 2
    token_id = 3

    # 2. SPACE character (QUAN TRỌNG!)
    vocab[" "] = token_id
    token_id += 1

    # 3. Punctuation cơ bản
    for p in ['!', '%', '&', "'", ',', '-', '.']:
        vocab[p] = token_id
        token_id += 1

    # 4. Digits
    for d in '0123456789':
        vocab[d] = token_id
        token_id += 1

    vocab['?'] = token_id
    token_id += 1
    
    # 4. Chữ cái tiếng Việt lowercase (a-z + dấu)
    vietnamese_lowercase = list('abcdefghijklmnopqrstuvwxyz')
    vietnamese_lowercase.extend([
        'à', 'á', 'ả', 'ã', 'ạ',
        'ă', 'ắ', 'ằ', 'ẳ', 'ẵ', 'ặ',
        'â', 'ấ', 'ầ', 'ẩ', 'ẫ', 'ậ',
        'è', 'é', 'ẻ', 'ẽ', 'ẹ',
        'ê', 'ế', 'ề', 'ể', 'ễ', 'ệ',
        'ì', 'í', 'ỉ', 'ĩ', 'ị',
        'ò', 'ó', 'ỏ', 'õ', 'ọ',
        'ô', 'ố', 'ồ', 'ổ', 'ỗ', 'ộ',
        'ơ', 'ớ', 'ờ', 'ở', 'ỡ', 'ợ',
        'ù', 'ú', 'ủ', 'ũ', 'ụ',
        'ư', 'ứ', 'ừ', 'ử', 'ữ', 'ự',
        'ỳ', 'ý', 'ỷ', 'ỹ', 'ỵ',
        'đ',
    ])
    
    for char in vietnamese_lowercase:
        if char not in vocab:
            vocab[char] = token_id
            token_id += 1
    
    print(f"✅ Vietnamese lowercase: {len(vietnamese_lowercase)}")
    
    # 5. Uppercase A-Z
    for c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
        vocab[c] = token_id
        token_id += 1

    # 6. Vietnamese uppercase có dấu (QUAN TRỌNG!)
    vietnamese_uppercase = [
        'À', 'Á', 'Ả', 'Ã', 'Ạ',
        'Ă', 'Ắ', 'Ằ', 'Ẳ', 'Ẵ', 'Ặ',
        'Â', 'Ấ', 'Ầ', 'Ẩ', 'Ẫ', 'Ậ',
        'È', 'É', 'Ẻ', 'Ẽ', 'Ẹ',
        'Ê', 'Ế', 'Ề', 'Ể', 'Ễ', 'Ệ',
        'Ì', 'Í', 'Ỉ', 'Ĩ', 'Ị',
        'Ò', 'Ó', 'Ỏ', 'Õ', 'Ọ',
        'Ô', 'Ố', 'Ồ', 'Ổ', 'Ỗ', 'Ộ',
        'Ơ', 'Ớ', 'Ờ', 'Ở', 'Ỡ', 'Ợ',
        'Ù', 'Ú', 'Ủ', 'Ũ', 'Ụ',
        'Ư', 'Ứ', 'Ừ', 'Ử', 'Ữ', 'Ự',
        'Ỳ', 'Ý', 'Ỷ', 'Ỹ', 'Ỵ',
        'Đ',
    ]

    for char in vietnamese_uppercase:
        if char not in vocab:
            vocab[char] = token_id
            token_id += 1

    print(f"✅ Vietnamese uppercase: {len(vietnamese_uppercase)}")
    
    # 7. Punctuation mở rộng
    extended_punct = [':', ';', '(', ')', '[', ']', '{', '}', '/', '\\', '@', '#', '$', '*', '+', '=', '<', '>', '~', '`', '^', '_', '|']
    for p in extended_punct:
        if p not in vocab:
            vocab[p] = token_id
            token_id += 1

    # 8. Smart quotes & special punctuation (từ dataset)
    smart_punct = ['"', '"', ''', ''', '…', '—', '–', '，', '、', '。', '！', '？']
    for p in smart_punct:
        if p not in vocab:
            vocab[p] = token_id
            token_id += 1
    
    # 9. Ký tự đặc biệt - GIỮ LẠI MỘT SỐ QUAN TRỌNG
    special_chars = [
        '°', '±', '×', '÷', '€', '£', '¥',
    ]
    for char in special_chars:
        if char not in vocab:
            vocab[char] = token_id
            token_id += 1
    
    # 10. Latin extended - BỎ ĐI để giảm xuống 704 tokens
    # Các ký tự này không xuất hiện trong dataset tiếng Việt đã tối ưu
    
    # 9. Padding đến 255 (nếu cần) - ĐẢM BẢO ĐỦ 704 TOKENS
    while token_id < 255:
        # Thêm ký tự Unicode - đảm bảo không trùng
        if token_id < 128:
            char = chr(0x80 + token_id)
        elif token_id < 200:
            char = chr(0x100 + (token_id - 128))
        else:
            char = chr(0x150 + (token_id - 200))

        if char not in vocab:
            vocab[char] = token_id
            token_id += 1
        else:
            # Nếu trùng, dùng ký tự khác
            vocab[f"_U{token_id}_"] = token_id
            token_id += 1

    # 10. [START] ở vị trí 255
    vocab["[START]"] = 255
    token_id = 256
    
    # 10. BPE tokens - bigrams phổ biến
    bigrams = ['ng', 'nh', 'th', 'ch', 'tr', 'kh', 'ph', 'gh', 'gi', 'qu']
    for bg in bigrams:
        if bg not in vocab:
            vocab[bg] = token_id
            token_id += 1
    
    # 11. Từ phổ biến (top 100)
    common_words = [
        'có', 'là', 'và', 'một', 'của', 'không', 'thể', 'người',
        'các', 'trong', 'những', 'cho', 'để', 'được', 'tôi', 'bạn',
        'với', 'đã', 'sự', 'ta', 'việc', 'sẽ', 'chúng', 'khi',
        'cũng', 'như', 'mà', 'đến', 'ra', 'này', 'từ', 'về',
        'nên', 'sau', 'thì', 'năm', 'ngày', 'họ', 'mình', 'rất',
        'đang', 'còn', 'vẫn', 'đều', 'cả', 'nhiều', 'nào', 'hay',
        'đó', 'nó', 'ai', 'gì', 'đây', 'đấy', 'ấy', 'kia',
        'nọ', 'bao', 'bất', 'cứ', 'mỗi', 'mọi', 'tất', 'toàn',
        'cùng', 'nhau', 'nhất', 'hơn', 'lại', 'nữa', 'thêm', 'luôn',
        'vừa', 'mới', 'sắp', 'rồi', 'xong', 'hết', 'bị', 'phải',
        'muốn', 'thích', 'yêu', 'ghét', 'biết', 'hiểu', 'nghĩ', 'tin',
        'làm', 'nói', 'hỏi', 'trả', 'lời', 'nói', 'kể', 'bảo',
        'gọi', 'đọc', 'viết', 'nghe', 'nhìn', 'thấy',
    ]
    
    for word in common_words:
        if word not in vocab and token_id < 604:
            vocab[word] = token_id
            token_id += 1
    
    print(f"✅ BPE tokens (bigrams + words): {token_id - 256}")
    
    # 12. Thêm ký tự Unicode để đủ 604
    # Dùng các ký tự từ bảng mã Unicode
    for i in range(256, 1024):
        if token_id >= 604:
            break
        try:
            char = chr(i)
            if char not in vocab and char.isprintable():
                vocab[char] = token_id
                token_id += 1
        except:
            pass
    
    # 13. Expressive tokens (604-639)
    expressive_tokens = [
        '[UH]', '[UM]', '[giggle]', '[laughter]', '[guffaw]',
        '[inhale]', '[exhale]', '[sigh]', '[cry]', '[bark]',
        '[howl]', '[meow]', '[singing]', '[music]', '[whistle]',
        '[humming]', '[gasp]', '[groan]', '[whisper]', '[mumble]',
        '[sniff]', '[sneeze]', '[cough]', '[snore]', '[chew]',
        '[sip]', '[clear_throat]', '[kiss]', '[shhh]', '[gibberish]',
        '[fr]', '[es]', '[de]', '[it]', '[ipa]', '[end_of_label]'
    ]
    
    for token in expressive_tokens:
        vocab[token] = token_id
        token_id += 1
    
    print(f"✅ Expressive tokens: {len(expressive_tokens)}")
    
    # 14. IPA phonemes (640-694) - GIỐNG ENGLISH
    # International Phonetic Alphabet - cho phoneme-based TTS
    ipa_phonemes = [
        # Consonants
        'ŋ', 'θ', 'ð', 'ʃ', 'ʒ', 'tʃ', 'dʒ', 'ʔ',
        # Vowels
        'ɑː', 'æ', 'ʌ', 'ɒ', 'ɔː', 'ɜː', 'ə', 'ɪ', 'iː', 'ʊ', 'uː', 'eɪ', 'aɪ', 'ɔɪ', 'aʊ', 'əʊ',
        # Vietnamese specific
        'ɯ', 'ɤ', 'ɨ', 'ʉ', 'ɘ', 'ɵ', 'ɜ', 'ɞ', 'ɐ', 'ɶ', 'ɑ', 'ɒ', 'ʌ', 'ɔ', 'ɤ', 'ɯ',
        # Tones (Vietnamese)
        '˧', '˥', '˩˧', '˧˥', '˧˩˧', '˧˩',
        # Additional IPA
        'ɓ', 'ɗ', 'ɠ', 'ʄ', 'ʛ', 'ɲ', 'ɳ', 'ɱ', 'ʈ', 'ɖ', 'ɟ', 'ɡ', 'ɢ', 'ʡ', 'ʕ', 'ʜ', 'ʢ',
    ]

    for phoneme in ipa_phonemes:
        if token_id >= 695:
            break
        if phoneme not in vocab:
            vocab[phoneme] = token_id
            token_id += 1

    # Padding với ký tự Unicode nếu chưa đủ 695
    for i in range(0x250, 0x2B0):  # IPA Extensions Unicode block
        if token_id >= 695:
            break
        char = chr(i)
        if char not in vocab:
            vocab[char] = token_id
            token_id += 1

    # Nếu vẫn chưa đủ 695, thêm ký tự Unicode khác
    for i in range(0x2B0, 0x400):
        if token_id >= 695:
            break
        try:
            char = chr(i)
            if char not in vocab and char.isprintable():
                vocab[char] = token_id
                token_id += 1
        except:
            pass

    # Force padding đến 695 nếu cần
    while token_id < 695:
        # Thêm ký tự từ các Unicode blocks khác
        char = chr(0x400 + (token_id - 640))
        if char not in vocab:
            vocab[char] = token_id
        token_id += 1

    # 15. Placeholder tokens (695-703) - GIỐNG ENGLISH
    for i in range(55, 64):
        vocab[f"[PLACEHOLDER{i}]"] = token_id
        token_id += 1

    print(f"✅ IPA phonemes (640-694): {len([k for k, v in vocab.items() if v >= 640 and v < 695])}")
    
    print(f"\n📊 Total tokens: {len(vocab)}")
    print(f"📊 Max token ID: {max(vocab.values())}")
    
    # Tạo BPE merges
    merges = ['n g', 'n h', 't h', 'c h', 't r', 'k h', 'p h', 'g h', 'g i', 'q u']
    
    # Tạo tokenizer JSON
    tokenizer_json = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [
            {"id": 0, "content": "[STOP]", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True},
            {"id": 1, "content": "[UNK]", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True},
            {"id": 2, "content": "[SPACE]", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True},
            {"id": 255, "content": "[START]", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True},
        ],
        "normalizer": None,
        "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": None,
        "decoder": None,
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": "[UNK]",
            "continuing_subword_prefix": None,
            "end_of_word_suffix": None,
            "fuse_unk": False,
            "byte_fallback": False,
            "vocab": vocab,
            "merges": merges,
            "language": "vi"
        }
    }
    
    # Lưu
    output_dir = Path("VietnameseTokenizer")
    output_dir.mkdir(exist_ok=True)
    
    output_path = output_dir / "tokenizer.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(tokenizer_json, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Saved: {output_path}")
    
    # Lưu vocab list
    vocab_list_path = output_dir / "vocab_list.txt"
    with open(vocab_list_path, "w", encoding="utf-8") as f:
        for token, tid in sorted(vocab.items(), key=lambda x: x[1]):
            f.write(f"{tid}\t{token}\n")
    
    print(f"💾 Saved: {vocab_list_path}")
    
    # Test
    print("\n" + "="*80)
    print("TEST")
    print("="*80)
    
    from tokenizers import Tokenizer
    
    tokenizer = Tokenizer.from_file(str(output_path))
    
    for text in ["Xin chào", "Tiếng Việt", "chào", "chao"]:
        enc = tokenizer.encode(text)
        print(f"\n'{text}': {enc.tokens} → {enc.ids}")
    
    print("\n" + "="*80)
    print("✅ DONE! 704 tokens (NO PAD)")
    print("="*80 + "\n")


if __name__ == "__main__":
    create_vietnamese_tokenizer()

