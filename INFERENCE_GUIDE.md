# Vietnamese TTS Inference Guide

## Giải quyết vấn đề audio ngắn không ổn định

Model TTS có thể tạo ra audio với độ dài khác nhau cho cùng một text do tính ngẫu nhiên trong quá trình sampling. Đây là các tham số để kiểm soát:

## Tham số quan trọng

### 1. **--min_tokens** (mặc định: 10)
Số lượng speech token tối thiểu phải generate trước khi cho phép dừng.

**Khuyến nghị:**
- Text ngắn (< 5 từ): `--min_tokens 20`
- Text trung bình (5-15 từ): `--min_tokens 30-40`
- Text dài (> 15 từ): `--min_tokens 50+`

### 2. **--seed** (mặc định: None)
Cố định random seed để có kết quả nhất quán.

**Sử dụng:**
- Khi cần kết quả giống hệt nhau: `--seed 42`
- Khi test/debug: luôn dùng seed cố định

### 3. **--temperature** (mặc định: 0.8)
Kiểm soát độ ngẫu nhiên của sampling.

**Khuyến nghị:**
- Giá trị thấp (0.5-0.7): Ổn định hơn, ít biến đổi
- Giá trị cao (0.8-1.0): Tự nhiên hơn, nhưng kém ổn định

### 4. **--cfg_weight** (mặc định: 0.5)
Classifier-free guidance - kiểm soát độ tuân thủ với text.

**Khuyến nghị:**
- 0.0: Không dùng CFG (nhanh hơn)
- 0.5-0.8: Cân bằng giữa tốc độ và chất lượng
- 1.0+: Tuân thủ text chặt chẽ hơn

## Ví dụ sử dụng

### Tạo audio ổn định cho production:
```bash
python infer.py \
    --checkpoint ./vietnamese/checkpoint-100000/ \
    --base_model ./vietnamese/pretrained_model_download \
    --voice "path/to/voice.wav" \
    --text "Văn bản của bạn" \
    --seed 42 \
    --min_tokens 40 \
    --temperature 0.7
```

### Debug khi gặp audio quá ngắn:
```bash
python infer.py \
    --checkpoint ./vietnamese/checkpoint-100000/ \
    --base_model ./vietnamese/pretrained_model_download \
    --voice "path/to/voice.wav" \
    --text "Văn bản của bạn" \
    --min_tokens 50 \
    --temperature 0.6 \
    --seed 123
```

### Tạo audio tự nhiên (chấp nhận biến động):
```bash
python infer.py \
    --checkpoint ./vietnamese/checkpoint-100000/ \
    --base_model ./vietnamese/pretrained_model_download \
    --voice "path/to/voice.wav" \
    --text "Văn bản của bạn" \
    --min_tokens 30 \
    --temperature 0.9
```

## Xử lý sự cố

### Vấn đề: Audio quá ngắn (< 1 giây)
**Giải pháp:**
1. Tăng `--min_tokens` lên 40-60
2. Giảm `--temperature` xuống 0.6-0.7
3. Sử dụng `--seed` cố định

### Vấn đề: Audio không nhất quán giữa các lần chạy
**Giải pháp:**
1. Luôn dùng `--seed` cố định (ví dụ: 42)
2. Giảm `--temperature` xuống 0.7

### Vấn đề: Audio có tiếng ồn hoặc lặp lại
**Giải pháp:**
1. Kiểm tra file voice reference
2. Điều chỉnh `--cfg_weight` (thử 0.3-0.7)
3. Giảm `--exaggeration` xuống 0.3-0.4

## Debug Information

Khi chạy inference, model sẽ hiển thị:
- Số speech tokens được generate
- Số tokens sau khi loại bỏ invalid tokens
- Thời lượng audio tạo ra
- Cảnh báo nếu audio quá ngắn

Ví dụ output tốt:
```
Sampling: 7%|▋         | 72/1000 [00:01<00:21, 42.80it/s]
   Generated 77 speech tokens
   After dropping invalid: 76 tokens
✅ SUCCESS!
📁 Audio saved: output.wav
🎵 Sample rate: 24000 Hz
⏱️  Duration: 3.04s
```

## Khuyến nghị cho Production

1. **Luôn sử dụng seed cố định** để đảm bảo reproducibility
2. **Set min_tokens phù hợp** với độ dài text (tối thiểu 30)
3. **Giảm temperature xuống 0.6-0.7** để tăng tính ổn định
4. **Test với nhiều voice reference** khác nhau
5. **Monitor số lượng tokens** được generate để phát hiện vấn đề sớm
