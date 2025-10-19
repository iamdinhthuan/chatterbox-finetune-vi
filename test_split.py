from infer import split_sentences, normalize_text

# Test 1: Normal sentence splitting
print("=" * 80)
print("TEST 1: Normal sentence splitting")
print("=" * 80)
text1 = "Xin chào các bạn. Hôm nay trời đẹp! Bạn có khỏe không?"
print(f"Input: {text1}")
print(f"Length: {len(text1)}")
sents1 = split_sentences(text1)
print(f"\nSplit result ({len(sents1)} sentences):")
for i, s in enumerate(sents1, 1):
    print(f"  {i}. ({len(s)} chars) {s}")

# Test 2: Long sentence with commas (should split)
print("\n" + "=" * 80)
print("TEST 2: Long sentence with commas (should split)")
print("=" * 80)
text2 = "Đây là một câu rất dài với nhiều dấu phẩy, để kiểm tra xem nó có tách không, khi vượt quá giới hạn ký tự, và có đủ từ trước mỗi dấu phẩy."
print(f"Input: {text2}")
print(f"Length: {len(text2)}")
sents2 = split_sentences(text2)
print(f"\nSplit result ({len(sents2)} sentences):")
for i, s in enumerate(sents2, 1):
    print(f"  {i}. ({len(s)} chars) {s}")

# Test 3: Short sentence with commas (should NOT split)
print("\n" + "=" * 80)
print("TEST 3: Short sentence with commas (should NOT split)")
print("=" * 80)
text3 = "Tôi thích ăn phở, bún, và cơm."
print(f"Input: {text3}")
print(f"Length: {len(text3)}")
sents3 = split_sentences(text3)
print(f"\nSplit result ({len(sents3)} sentences):")
for i, s in enumerate(sents3, 1):
    print(f"  {i}. ({len(s)} chars) {s}")

# Test 4: Multiple sentences + long with commas
print("\n" + "=" * 80)
print("TEST 4: Multiple sentences + long with commas")
print("=" * 80)
text4 = "Xin chào! Đây là câu dài với nhiều phần, bao gồm nhiều thông tin, để test việc tách câu theo dấu phẩy khi quá dài. Câu cuối ngắn."
print(f"Input: {text4}")
print(f"Length: {len(text4)}")
sents4 = split_sentences(text4)
print(f"\nSplit result ({len(sents4)} sentences):")
for i, s in enumerate(sents4, 1):
    print(f"  {i}. ({len(s)} chars) {s}")

# Test 5: Normalization + splitting
print("\n" + "=" * 80)
print("TEST 5: Normalization + splitting")
print("=" * 80)
text5 = "Xin   chào ! Đây  là  câu dài  với  khoảng   trắng thừa ,  và  dấu phẩy ,  để test normalize."
print(f"Input: {text5}")
norm5 = normalize_text(text5)
print(f"Normalized: {norm5}")
sents5 = split_sentences(norm5)
print(f"\nSplit result ({len(sents5)} sentences):")
for i, s in enumerate(sents5, 1):
    print(f"  {i}. ({len(s)} chars) {s}")
