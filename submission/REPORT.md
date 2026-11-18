# Báo cáo Day 09 — Vũ Tiến Vinh, 2A202603269

## 1. Kết luận trước
Pipeline kết luận: **Hold**. 
Tổng số ảnh đầu vào là 224 ảnh (`data/raw`). Sau khi qua Bước 1 (Nhận dữ liệu - Intake), có 21 ảnh bị loại bỏ (gồm 3 ảnh sai kích thước `BAD_SIZE`, 2 ảnh hỏng `CORRUPT_IMAGE`, 8 ảnh trùng tuyệt đối `DUP_IMAGE_EXACT`, 3 ảnh gần trùng `DUP_IMAGE_NEAR`, và 5 ảnh rò rỉ tập test `LEAK_TEST`), còn lại 203 ảnh đi vào pipeline và được xuất ra dataset `submission/export/clean_v1`. 
Lý do chính kết luận **Hold**: Sau khi máy tự động sửa các lỗi cú pháp và mô phỏng người duyệt 80 ảnh, pipeline không còn bất kỳ lỗi bắt buộc nào (0 errors). Tuy nhiên, tỷ lệ ảnh còn cảnh báo nghi ngờ chưa được kiểm duyệt là **21.67%** (44/203 ảnh), vượt quá ngưỡng quy định 10.00% (`warn_ratio_hold = 0.10`). Theo quy tắc dự án, lô dữ liệu phải tạm giữ (Hold) để tiếp tục duyệt phần còn lại trước khi chính thức phát hành.

## 2. Lỗi và phân việc
Tổng hợp các lỗi phát hiện được từ `02_error_list_v1` (trước khi sửa):

| Mã lỗi | Số ảnh/dòng | Ai sửa (máy/người) | Vì sao |
|---|---:|---|---|
| `BAD_CLASS_NAME` | 2 lỗi | Máy (autofix `rename_classes`) | Tên lớp trong `data.yaml` viết hoa sai ('Person', 'Bus'), máy chuẩn hoá tự động theo danh sách 4 lớp chuẩn `[car, person, truck, bus]`. |
| `BAD_SIZE` | 3 ảnh | Máy (intake `remove_image`) | Ảnh có cạnh ngắn < 250px không đủ chuẩn kích thước tối thiểu của bài toán. |
| `CORRUPT_IMAGE` | 2 ảnh | Máy (intake `remove_image`) | File ảnh hỏng cấu trúc (`UnidentifiedImageError`, `OSError`), không thể load được. |
| `DUP_IMAGE_EXACT` | 8 ảnh | Máy (intake `remove_image`) | Trùng khớp 100% từng byte (MD5) với ảnh xuất hiện trước trong lô. |
| `DUP_IMAGE_NEAR` | 3 ảnh | Máy (intake `remove_image`) | Trùng gần đúng qua Perceptual Hash (khoảng cách Hamming phash $\le 8$, vd. ảnh thu nhỏ, đổi tỉ lệ). |
| `LEAK_TEST` | 5 ảnh | Máy (intake `remove_image`) | Ảnh trùng với kho dữ liệu kiểm thử `data/test/` hoặc `gold_val/`, bắt buộc phải loại để tránh rò rỉ dữ liệu (data leakage). |
| `BAD_CLASS_ID` | 6 dòng | Máy (autofix `drop_line`) | ID lớp nằm ngoài phạm vi 0..3 (-1, 4, 7, 93), không hợp lệ theo định dạng YOLO của dự án. |
| `BAD_FORMAT` | 3 dòng | Máy (autofix `drop_line`) | Dòng nhãn hỏng (chứa giá trị `nan` hoặc thiếu toạ độ) không thể phân tích cú pháp. |
| `BOX_ZERO` | 4 dòng | Máy (autofix `drop_line`) | Box rỗng có chiều rộng hoặc chiều cao $\le 0$, không mang thông tin hình học. |
| `BOX_OUT_OF_IMAGE` | 8 dòng | Máy (autofix `clip_box`) | Mép box bị tràn nhẹ ra ngoài khung hình $[0, 1]$ vượt dung sai 0.002, máy cắt toạ độ về lại $[0, 1]$ an toàn. |
| `DUP_BOX` | 7 dòng | Máy (autofix `drop_line`) | Box bị gán nhân đôi (trùng cùng lớp và IoU > 0.9 với box có trước trong cùng một ảnh). |
| `MISSING_LABEL_FILE` | 6 ảnh | Người (hàng đợi `needs_label`) | Ảnh hoàn toàn chưa có file nhãn; máy không được tự ý bịa nhãn mà bắt buộc phải đưa lên đầu hàng đợi để người gán. |
| `EMPTY_LABEL` | 12 ảnh | Người (hàng đợi duyệt) | File nhãn rỗng; người duyệt cần xác định xem đây là ảnh nền (background không có đối tượng) hay annotator bỏ quên. |
| `BOX_TINY` | 6 dòng | Người (hàng đợi duyệt) | Cạnh ngắn của box (px) < 12px; cần người kiểm tra xem có phải nhiễu hoặc vật thể quá nhỏ không nên gán. |
| `BOX_HUGE` | 3 dòng | Người (hàng đợi duyệt) | Diện tích box chiếm > 95% ảnh; cần người kiểm tra xem box có bị kéo tràn toàn màn hình do lỗi thao tác không. |
| `CLASS_RARE` | 4 lớp | Máy cảnh báo | Thống kê số lượng box của các lớp hiếm (cảnh báo mất cân bằng dữ liệu cấp dataset). |
| `POSSIBLE_WRONG_CLASS` | 18 ảnh/lỗi | Người (hàng đợi duyệt) | Model checker (yolo26l) nghi ngờ nhãn bị nhầm lớp (IoU $\ge 0.5$ với dự đoán nhưng khác lớp, vd. car vs truck/bus). |
| `POSSIBLE_MISSING` | 68 ảnh/lỗi | Người (hàng đợi duyệt) | Model checker phát hiện đối tượng rõ ràng (conf $\ge 0.50$, diện tích $\ge 600\text{px}^2$) mà nhãn người bị bỏ sót. |
| `POSSIBLE_LOOSE_BOX` | 31 ảnh/lỗi | Người (hàng đợi duyệt) | Cùng lớp nhưng box nhãn bị vẽ lệch/lỏng so với đối tượng (IoU trong khoảng $[0.3, 0.5)$). |
| `POSSIBLE_FALSE_POS` | 29 dòng | Người (hàng đợi duyệt) | Box nhãn không khớp với bất kỳ đối tượng nào model checker nhìn thấy (nghi ngờ vẽ thừa/vẽ nhầm vào nền). |

**Vì sao `05_review_queue.csv` xếp ảnh đầu hàng như vậy?**
Thuật toán xếp hàng tính điểm ưu tiên (`priority`) dựa trên tổng trọng số của các lỗi xuất hiện trên ảnh:
- Các ảnh dính lỗi thiếu nhãn `NEEDS_LABEL` (6 ảnh: `img_0031`, `img_0053`, `img_0126`, `img_0135`, `img_0170`, `img_0219`) được gán trọng số cao nhất (10.0) vì đây là lỗi bắt buộc (`ERROR`), nếu không có người gán thì pipeline sẽ rơi vào **Rework**.
- Các ảnh vừa bị máy sửa tự động (`FIXED_DROP_LINE` trọng số 2, `FIXED_CLIP_BOX` trọng số 1) đồng thời có nhiều cảnh báo ngữ nghĩa nghiêm trọng (`POSSIBLE_WRONG_CLASS` trọng số 3, `POSSIBLE_MISSING` trọng số 2 có nhân trọng số với $1 + \text{conf}$) như `img_0020` (priority 14.528), `img_0185` (priority 10.052) được xếp lên các vị trí đầu tiên.
- **Hiệu quả ngân sách 80 ảnh**: Với ngân sách 80 ảnh, người duyệt xem được toàn bộ 6 ảnh thiếu nhãn (chuyển thành có nhãn sạch) và giải quyết được 80/130 ảnh trong hàng đợi. Nhờ đó, số lỗi ngữ nghĩa giảm rõ rệt: `POSSIBLE_WRONG_CLASS` giảm từ 18 xuống còn 4 lỗi, `POSSIBLE_MISSING` giảm từ 68 xuống còn 26 lỗi (giải quyết được 62% lỗi missing và 78% lỗi wrong class).

## 3. Calibration (2 ngưỡng)
| Tham số | Khởi đầu | Đã chọn | Precision / Recall (trước → sau) | Lý do (đánh đổi) |
|---|---|---|---|---|
| `duplicates.phash_dist` | 2 | **8** | DUP_IMAGE_NEAR: $1.00 / 0.50 \rightarrow \mathbf{1.00 / 0.67}$ (F1: $0.67 \rightarrow \mathbf{0.80}$) | Ngưỡng khởi đầu 2 quá chặt, chỉ bắt được 50% ảnh gần trùng trên tập calib (Recall 0.50). Nâng lên 8 giúp tăng Recall lên 0.67 mà vẫn duy trì Precision tuyệt đối 1.00. Trên tập `raw`, ngưỡng 8 phát hiện chính xác các ảnh bị biến đổi như `img_0223` (thu nhỏ 75% từ `img_0077`, dist = 6) mà không bị xóa nhầm ảnh khác. |
| `second_opinion.conf` | 0.25 | **0.50** | POSSIBLE_MISSING: $0.50 / 1.00 \rightarrow \mathbf{0.57 / 1.00}$ (F1: $0.67 \rightarrow \mathbf{0.73}$) | Ở ngưỡng 0.25, model checker sinh ra rất nhiều dự đoán độ tin cậy thấp, khiến 50% cảnh báo `POSSIBLE_MISSING` là báo động giả (FP = 8 trên calib), làm hàng đợi duyệt bị phình to vượt quá ngân sách 80 ảnh. Tăng ngưỡng lên 0.50 giúp tăng Precision lên 0.57, loại bớt báo động giả nhưng vẫn giữ trọn vẹn 100% Recall (8/8 ca thiếu nhãn thật đều được bắt). Không tăng lên $\ge 0.70$ vì Recall bắt đầu sụt giảm (bỏ sót lỗi). |

**Một chỗ calibration có thể đánh lừa bạn:**
Tập calibration chỉ có **39 ảnh** với các lỗi chèn nhân tạo đơn giản. Nếu chỉ máy móc nhìn vào bảng sweep để chọn F1 cao nhất:
- Với `phash_dist`: Trên calib, giá trị 16 đạt F1 = 0.91 (Recall 0.83). Tuy nhiên, khi áp dụng lên dữ liệu thật `data/raw`, khoảng cách Hamming 16/64 bit quá lỏng khiến thuật toán gom nhầm các ảnh hoàn toàn khác nhau vào cùng mã băm, làm số ảnh bị coi là rò rỉ `LEAK_TEST` nhảy vọt từ 5 lên 11 ảnh (xóa oan 6 ảnh sạch).
- Với `second_opinion.conf`: Đa số lỗi `WRONG_CLASS` trong calib là person $\leftrightarrow$ car nhân tạo, trong khi trên dữ liệu thật lỗi chủ yếu nằm ở các cặp lớp nhập nhằng như car $\leftrightarrow$ truck $\leftrightarrow$ bus. Nếu tin tưởng tuyệt đối vào độ tin cậy của calib mà đẩy `conf` lên 0.80 để triệt tiêu báo động giả, model sẽ bị "mù" trước các đối tượng thật bị bỏ sót trên `data/raw`.

## 4. Cửa chặn và điều pipeline KHÔNG bắt được
(a) **Chứng minh gate chặn đúng**:
Khi chạy pipeline với ngân sách duyệt bằng 0:
`python -m pipeline.run --budget 0 --reports work/rep_nb --export-dir work/exp_nb`
Do không có người duyệt, 6 ảnh thiếu nhãn (`MISSING_LABEL_FILE`) không được gán nhãn thủ công và vẫn còn nguyên trong `07_error_list_v3.csv`. Cửa chặn `gate` đã hoạt động chính xác:
- Kết luận: **REWORK — Còn 6 lỗi bắt buộc (error) chưa được khắc phục.**
- Mã thoát trả về: **20** (tương ứng với trạng thái Rework trong hệ thống CI).
- Kết quả xuất: Thư mục `work/exp_nb/clean_v1` **hoàn toàn không được tạo ra** (chặn triệt để việc xuất dữ liệu lỗi ra môi trường production).

(b) **Điểm mù pipeline không bắt được (kèm bằng chứng)**:
1. **Box lỏng nhẹ hoặc lệch góc**: Ở check `second_opinion`, ngưỡng `iou_loose` được ghim ở mức 0.50. Nếu một box nhãn bị vẽ rộng ra hoặc lệch vị trí nhưng IoU với đối tượng thật vẫn đạt $\ge 0.50$, pipeline coi là nhãn hợp lệ và hoàn toàn không đưa ra cảnh báo `POSSIBLE_LOOSE_BOX`.
2. **Nhầm lớp đối tượng mà cả model checker (yolo26l) cũng nhầm**: Khi một đối tượng bị che khuất một phần hoặc có hình dạng lai (ví dụ xe bán tải, SUV lớn dễ nhập nhằng giữa `car` và `truck`), nếu annotator gán là `car` và model yolo26l cũng dự đoán là `car`, pipeline sẽ coi như có sự đồng thuận (`agreement`) và bỏ qua lỗi nhầm lớp này.
3. **Ảnh trùng bị biến đổi hình học / cắt xén mạnh**: Thuật toán Perceptual Hash (phash) nhạy cảm với việc crop khung hình lớn. Nếu một ảnh bị cắt xén mất > 50% diện tích hoặc xoay góc rồi chèn lại, khoảng cách Hamming phash sẽ vượt quá ngưỡng 8 và lọt qua được check `duplicates`.

## 5. Train: raw vs clean (tùy chọn, chấm riêng)
Chưa chạy (đây là phần tùy chọn chấm riêng 15 điểm trên Google Colab / GPU; học viên tập trung tối ưu phần pipeline cốt lõi và kiểm soát chất lượng dữ liệu theo yêu cầu chuẩn của bài lab).

## 6. Quyết định và ghi chú AI
(a) **Quyết định cuối**: **Hold**. 
Lô dữ liệu chưa thể phát hành (`Release`) ngay vì vẫn còn 44 ảnh mang cảnh báo nghi ngờ chưa được kiểm duyệt (chiếm 21.67% lô).
Để chuyển thành **Release**, cần:
- Mở rộng ngân sách duyệt của con người thêm ít nhất 50 ảnh (tổng ngân sách $\ge 130$ ảnh) để đội ngũ annotator duyệt nốt 44 ảnh còn tồn đọng trong `05_review_queue.csv`.
- Sau khi 44 ảnh này được người kiểm tra và xác nhận/sửa đổi, tỷ lệ cảnh báo chưa duyệt sẽ giảm về 0% ($< 10\%$), và `gate` sẽ chính thức đưa ra kết luận **Release**.

(b) **Ghi chú sử dụng AI**:
- **Prompt chính**: "Xây dựng 4 module cho pipeline AutoQC theo phương pháp TDD: (1) `label_format.py` bắt lỗi cú pháp định dạng YOLO, tràn viền và trùng box; (2) `gate.py` phân nhánh quyết định Release/Hold/Rework theo tỷ lệ cảnh báo; (3) `autofix.py` sửa an toàn các lỗi cú pháp, cắt box tràn viền và ghi log truy vết; (4) `second_opinion.py` so sánh ground-truth với model checker yolo26l. Yêu cầu đọc kỹ các docstring, tuân thủ chặt chẽ cấu hình `cfg` và vượt qua toàn bộ test suite trong `tests/`."
- **Cách kiểm code AI**: Chạy bộ unit test `pytest tests/test_<module>.py -q` theo từng module; chủ động thử nghiệm các trường hợp biên (dòng trống, toạ độ âm, box tràn viền, nhiều lỗi trên cùng 1 dòng); đối chiếu file sinh ra bằng lệnh kiểm tra tự động `scripts/check_submission.py`.
- **1 lần AI sai**: Ở module `pipeline/steps/autofix.py`, ban đầu khi một dòng nhãn vừa dính lỗi `BOX_OUT_OF_IMAGE` vừa dính lỗi `DUP_BOX`, code do AI sinh ra suýt thực hiện cả hai thao tác (vừa cắt box vừa xóa dòng) khiến số lượng hành động ghi nhận trong log bị trùng lặp. Đã sửa lại theo nguyên tắc: thao tác xóa dòng (`drop_line`) luôn được ưu tiên tuyệt đối (`clip_lines = clip_lines - drop_lines`) để chỉ ghi duy nhất 1 lần `drop_line`.
