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
| (c) LoRA fine-tune | 0.970 | 0.6778 | 1.000 | 1359.9 |

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
| `correct` | text-linear | 16 | 32,464,896 | 0.0001 | 0.6259 | 0.970 | 30 | 12.01 |
| `attn_only` | q,v | 283 (matched) | 32,456,704 | 0.0001 | 0.5376 | 0.965 | 30 | 12.02 |
| `wrong_lr` | text-linear | 16 | 32,464,896 | 0.00001 | 1.5704 | 0.000 | 30 | 12.01 |
| `qlora` | text-linear | 16 | 32,464,896 | 0.0001 | 0.7058 | 0.940 | 30 | 7.09 |

> Xếp hạng bằng cột **target**, không bằng cột train loss — chấm bằng chỉ số thay thế
> chính là Lỗi #3. Nếu hai cột cho hai thứ tự khác nhau, nói thẳng điều đó ở 4.1: đó là
> kết quả đáng giá nhất bạn đo được trong lab này.

Trả lời ba câu (mỗi câu ≥3 câu văn):

**4.1 — `attn_only` có cùng số tham số huấn luyện với `correct`.** `attn_only` có
32,456,704 tham số, còn `correct` có 32,464,896, lệch khoảng 0.025%, nên đây là
đối chứng công bằng về ngân sách. Điểm target và thứ hạng cuối cùng sẽ được xác định
ở NB5; không dùng `final_loss` để kết luận. NB5 cho thấy `attn_only` đạt target
0.965, thấp hơn `correct` đạt 0.970, nên `attn_only` thua nhẹ trên target. Thứ hạng
này ngược với training loss: `attn_only` có loss 0.5376 thấp hơn `correct` là 0.6259.
Điều đó cho thấy vị trí gắn adapter quan trọng hơn việc chỉ tăng rank; rank 283 đã
cân bằng ngân sách nhưng không bù được việc chỉ cập nhật q,v.

**4.2 — `wrong_lr` chỉ khác đúng một con số.** `wrong_lr` dùng learning rate
0.00001 thay vì 0.0001; final loss là 1.5704 so với 0.6259 của `correct`, cao hơn
0.9445. Kết quả này cho thấy learning rate thấp hơn 10 lần khiến model học kém trong
cùng 30 steps. NB5 xác nhận `wrong_lr` đạt target 0.000 và format 0.000, trong khi
`correct` đạt target 0.970 và format 1.000. Vì vậy trong thí nghiệm này kết luận từ
loss phù hợp với target, nhưng về nguyên tắc vẫn không được dùng loss thay cho task
metric.

**4.3 — `qlora` tiết kiệm bao nhiêu VRAM, trả giá bằng gì?** `qlora` dùng 7.09 GB
VRAM, so với 12.01 GB của `correct`, tiết kiệm 4.92 GB, tương đương khoảng 41.0%.
Đổi lại, final loss của QLoRA là 0.7058, cao hơn `correct` 0.0799. Kết luận về chất
lượng task và việc có ủng hộ khuyến nghị không sẽ chờ các điểm target, regression và
format. QLoRA đạt target 0.940 và format 1.000, thấp hơn correct (0.970) nhưng vẫn
khá gần. Vì vậy số đo ủng hộ khuyến nghị thận trọng với QLoRA cho dòng model này:
tiết kiệm VRAM nhưng có mất mát target, dù chưa đủ để nói QLoRA luôn không dùng được.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: `FAILED`
`target Δ = +0.205` · `regression Δ = -0.080` · `valid_trace_rate = 0.00`

Fine-tune cải thiện target từ 0.765 lên 0.970, tức tăng 0.205 điểm, và giữ format ở
1.000. Tuy nhiên regression giảm từ 0.7578 xuống 0.6778, giảm 0.080; mức giảm này
vượt xa ngưỡng cho phép 0.020 nên cổng bị FAILED. Fine-tune đã học tốt phân phối
ticket triage, nhưng đánh đổi một phần năng lực chung của base model. Vì vậy không
thể kết luận bản fine-tune đã sẵn sàng deploy chỉ bằng target accuracy. `valid_trace_rate`
0.00 cũng cho thấy cần thận trọng khi diễn giải năng lực reasoning sau fine-tune, dù
NB1 xác nhận template giữ được khối `<think>`. Kết luận hợp lý là mô hình thắng rõ
trên task chuyên biệt nhưng chưa đạt cổng an toàn tổng thể; cần thêm replay data 1–5%
hoặc thiết kế lại dữ liệu trước khi triển khai.
(Một FAILED được phân tích tốt ăn điểm cao hơn một PASSED không giải thích được.)

---

## 6. Định tính — bắt buộc có cả ca THUA

| # | Ticket (rút gọn) | Nhãn đúng | (b) prompt | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
| 1 | Chuột không dây — trả lại, gấp | `doi_tra / cao / chuột không dây / tich_cuc` | Baseline (b) | FT đạt 1.00 | ✅ FT thắng |
| 2 | Ốp lưng — hoàn tiền, sớm | `hoan_tien / trung_binh / ốp lưng điện thoại / tieu_cuc` | Baseline (b) | FT đạt 1.00 | ✅ FT thắng |
| 3 | Bình giữ nhiệt — chưa thấy tiền | `hoan_tien / thap / bình giữ nhiệt / tich_cuc` | Baseline (b) | FT đạt 0.75 | ❌ FT thua một trường |
| 4 | Nồi chiên — thiếu phụ kiện | `san_pham_loi / thap / nồi chiên không dầu / trung_tinh` | Baseline (b) | FT đạt 0.75 | ❌ FT thua một trường |
| 5 | Áo khoác — bị lỗi | `san_pham_loi / thap / áo khoác gió / tich_cuc` | Baseline (b) | FT đạt 0.75 | ❌ FT thua một trường |

Các ca FT thua đều đạt đúng ba trong bốn trường. Lỗi còn lại thường nằm ở các nhãn
ngữ cảnh như urgency hoặc sentiment, trong khi intent và product vẫn đúng. Điều này
gợi ý fine-tune đã học tốt cấu trúc triage và trích xuất sản phẩm, nhưng còn nhạy với
các tín hiệu ngôn ngữ mềm như “khi nào tiện”, “cảm ơn” hoặc mức độ khẩn cấp.

---

## 7. Kết luận & điều tôi học được

**Kết luận.** Bản fine-tune không nên được deploy nguyên trạng dù target score tăng
mạnh từ 0.765 của baseline prompt tối ưu lên 0.970. Mức tăng 0.205 cho thấy dữ liệu
triage và cấu hình mask đã giúp model học đúng nhiệm vụ: format đạt 1.000 và phần lớn
các trường JSON được dự đoán chính xác. Tuy nhiên regression giảm từ 0.7578 xuống
0.6778, vượt ngưỡng cho phép nên verdict là FAILED. Điều này cho thấy fine-tune đã
chuyên môn hóa model quá mạnh trên 250 ticket, làm mất một phần năng lực chung. Nếu
chỉ nhìn target, tôi sẽ triển khai và bỏ qua rủi ro này; cổng regression buộc tôi nhìn
toàn diện hơn. Các đối chứng cho thấy vị trí adapter là đòn bẩy quan trọng: attn_only
đã match ngân sách nhưng target 0.965, thấp hơn correct 0.970 dù final loss lại thấp
hơn. Learning rate cũng rất quan trọng: wrong_lr làm target và format về 0. QLoRA
giảm 4.92 GB VRAM nhưng target thấp hơn correct. Bước sửa hợp lý nhất là thêm 1–5%
replay data phổ thông, đánh giá lại regression và kiểm tra reasoning trace trước khi
quyết định deploy.

**Ba điều tôi học được** (cụ thể, không generic):
1. Prompt tốt có thể tạo baseline 0.765, vì vậy fine-tune phải vượt qua một mốc cạnh tranh thực sự chứ không chỉ thắng prompt ngây thơ.
2. Training loss không thay thế được task target: attn_only có loss thấp hơn nhưng target thấp hơn correct.
3. Fine-tune có thể tăng accuracy chuyên biệt đồng thời làm regression giảm, nên verdict phải có nhiều nhóm đo.

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:**

---

## Phụ lục — thưởng đã làm

- [ ] B1 NB6 merge + hot-swap
- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [ ] B3 reasoning-trace collapse (hai `MASK_MODE`, kèm `valid_trace_rate`)
- [ ] B4 quét rank có kiểm soát
- [x] B5 HuggingFace Hub — các adapter công khai:
  - `correct`: https://huggingface.co/phucnt9186/lab21-correct
  - `attn_only`: https://huggingface.co/phucnt9186/lab21-attn-only
  - `wrong_lr`: https://huggingface.co/phucnt9186/lab21-wrong-lr
  - `qlora`: https://huggingface.co/phucnt9186/lab21-qlora

### Tải lại adapter sau khi clone repo

Các file `*.safetensors` không nằm trong GitHub vì kích thước lớn. Sau khi clone,
cài `huggingface_hub` rồi chạy:

```bash
.venv/bin/pip install -U huggingface_hub
.venv/bin/python scripts/download_adapters.py
```

Có thể tải riêng một adapter, ví dụ:

```bash
.venv/bin/python scripts/download_adapters.py --only correct
```
