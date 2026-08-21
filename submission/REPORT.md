# Lab 21 — Evaluation Report

**Họ tên**: Nguyễn Thanh Phúc  **MSSV**: 2A202601345  **Ngày**: 2026-08-21
**Tier**: `CPU` cho NB1, `T4` cho NB2 trở đi  **Base model**: `Qwen/Qwen3.5-0.8B` (NB1), `unsloth/Qwen3.5-4B` (NB2)  **GPU thực tế**: Không có GPU cục bộ; Tesla T4 trên Colab cho NB2

> Mọi con số dưới đây phải khớp với file trong `results/`. Grader kiểm tra chéo.

---

## 1. Setup

| | |
|---|---|
| Dataset | 250 ticket CSKH tiếng Việt → JSON triage |
| Train / val | 225 / 25 (seed 42) |
| `max_length` | `512` — p95 đo được là `98` *(results/token_stats.json)* |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | NB3 đã train với `max_steps=30` |

**Template có giữ khối `<think>` không?** Có — kết quả kiểm tra là
`reasoning preserved — safe to train on traces` *(results/template_check.json)*.

`max_length=512` lớn hơn giá trị đề xuất 256 (p95 = 98 token, làm tròn lên),
nhưng vẫn đủ an toàn cho mẫu dài nhất 101 token trong CPU tier.

---

## 2. Mask proof (NB1)

| | |
|---|---|
| `supervised_fraction` | `0.3936` |
| Câu trả lời nằm trong loss | `true` |
| Câu hỏi KHÔNG nằm trong loss | `true` |

Dán 3–5 dòng đầu của đoạn được tính loss:

```
{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0.000 | 0.7578 | 0.000 | 3371.77 |
| (b) base + optimized prompt | 0.765 | 0.7578 | 1.000 | 1050.19 |
| (c) LoRA fine-tune | Chưa đo (NB5) | Chưa đo (NB5) | Chưa đo (NB5) | Chưa đo (NB5) |

**(b) có thật sự mạnh hơn (a) không?** Có. Baseline (b) đạt target `0.765`,
cao hơn (a) `0.000`, đồng thời format tăng từ `0.000` lên `1.000`.
Regression của hai baseline giống nhau ở `0.7578`. Vì (b) đã khá mạnh, fine-tune
chỉ có ý nghĩa nếu vượt được mốc `0.765` mà không làm suy giảm các nhóm còn lại.
Tôi không sửa `OPTIMIZED_PROMPT` sau khi đo baseline; prompt được giữ nguyên và
đóng băng bằng SHA `719e74d3b6232053`.
Bạn có sửa `OPTIMIZED_PROMPT` không? Nếu có: **làm mạnh lên hay yếu đi**, và vì sao?

---

## 4. Giải phẫu cấu hình sai (NB4)

| Run | vị trí | r | trainable | LR | train loss (NB4) | **target (NB5 §4)** | s | VRAM GB |
|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear | 16 | 32,464,896 | 0.0001 | 0.6259 | Chưa đo (NB5) | 30 | 12.01 |
| `attn_only` | q,v | 283 (matched) | 32,456,704 | 0.0001 | 0.5376 | Chưa đo (NB5) | 30 | 12.02 |
| `wrong_lr` | text-linear | 16 | 32,464,896 | 0.00001 | 1.5704 | Chưa đo (NB5) | 30 | 12.01 |
| `qlora` | text-linear | 16 | 32,464,896 | 0.0001 | 0.7058 | Chưa đo (NB5) | 30 | 7.09 |

> Xếp hạng bằng cột **target**, không bằng cột train loss — chấm bằng chỉ số thay thế
> chính là Lỗi #3. Nếu hai cột cho hai thứ tự khác nhau, nói thẳng điều đó ở 4.1: đó là
> kết quả đáng giá nhất bạn đo được trong lab này.

Trả lời ba câu (mỗi câu ≥3 câu văn):

**4.1 — `attn_only` có cùng số tham số huấn luyện với `correct`.** `attn_only` có
32,456,704 tham số, còn `correct` có 32,464,896, lệch khoảng 0.025%, nên đây là
đối chứng công bằng về ngân sách. Điểm target và thứ hạng cuối cùng sẽ được xác định
ở NB5; không dùng `final_loss` để kết luận. NB4 cho thấy `attn_only` có loss 0.5376,
thấp hơn `correct` là 0.6259, nhưng loss thấp hơn chưa chứng minh target cao hơn.

**4.2 — `wrong_lr` chỉ khác đúng một con số.** `wrong_lr` dùng learning rate
0.00001 thay vì 0.0001; final loss là 1.5704 so với 0.6259 của `correct`, cao hơn
0.9445. Kết quả này cho thấy learning rate thấp hơn 10 lần khiến model học kém trong
cùng 30 steps. Tuy nhiên, thứ hạng trên task vẫn phải kiểm tra bằng target ở NB5,
không được suy ra chỉ từ training loss.

**4.3 — `qlora` tiết kiệm bao nhiêu VRAM, trả giá bằng gì?** `qlora` dùng 7.09 GB
VRAM, so với 12.01 GB của `correct`, tiết kiệm 4.92 GB, tương đương khoảng 41.0%.
Đổi lại, final loss của QLoRA là 0.7058, cao hơn `correct` 0.0799. Kết luận về chất
lượng task và việc có ủng hộ khuyến nghị không sẽ chờ các điểm target, regression và
format được đo ở NB5.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: `<PASSED | FAILED>`
`target Δ = <+0.xxx>` · `regression Δ = <+0.xxx>` · `valid_trace_rate = <0.xx>`

Diễn giải (≥100 từ). Nếu FAILED: **vì sao**, và điều đó nói gì về bài toán của bạn?
(Một FAILED được phân tích tốt ăn điểm cao hơn một PASSED không giải thích được.)

---

## 6. Định tính — bắt buộc có cả ca THUA

| # | Ticket (rút gọn) | Nhãn đúng | (b) prompt | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
| 1 | | | | | ✅ FT thắng |
| 2 | | | | | ✅ FT thắng |
| 3 | | | | | ❌ **FT thua** |
| 4 | | | | | ❌ **FT thua** |
| 5 | | | | | |

Có mẫu chung nào ở các ca FT thua không?

---

## 7. Kết luận & điều tôi học được

**Kết luận (≥150 từ).** Bạn có nên deploy bản fine-tune này không, và vì sao? Đâu là đòn
bẩy thật sự trong lab này — vị trí adapter, learning rate, chất lượng dữ liệu, hay mask?

**Ba điều tôi học được** (cụ thể, không generic):
1.
2.
3.

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:**

---

## Phụ lục — thưởng đã làm

- [ ] B1 NB6 merge + hot-swap
- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [ ] B3 reasoning-trace collapse (hai `MASK_MODE`, kèm `valid_trace_rate`)
- [ ] B4 quét rank có kiểm soát
- [ ] B5 HuggingFace Hub — link:
