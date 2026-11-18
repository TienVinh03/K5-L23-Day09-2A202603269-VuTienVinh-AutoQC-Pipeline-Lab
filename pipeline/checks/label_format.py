"""CHECK 2 — Dòng nhãn YOLO: sai định dạng, ID lớp lạ, box tràn ảnh, box rỗng, box lặp.   (Script 2 trong slide)   ★★

Test:  pytest tests/test_label_format.py -q

Mỗi dòng nhãn: `id cx cy w h` (5 số, toạ độ 0–1). Trả về list[Issue]; `Issue.line` = số dòng (đếm từ 1).

| code             | severity | khi nào                                                                  |
|------------------|----------|--------------------------------------------------------------------------|
| BAD_FORMAT       | ERROR    | không đúng 5 cột số (dùng `parse_yolo_line`, bắt `ValueError`)           |
| BAD_CLASS_ID     | ERROR    | id lớp ngoài 0..len(ds.classes)-1                                        |
| BOX_ZERO         | ERROR    | w <= 0 hoặc h <= 0                                                       |
| BOX_OUT_OF_IMAGE | ERROR    | mép box vượt [0,1] quá `cfg["label_format"]["overflow_tol"]`; extra={"overflow": mức tràn} |
| DUP_BOX          | ERROR    | cùng lớp với một dòng TRƯỚC ĐÓ và IoU > `dup_box_iou`; báo ở dòng SAU    |
| BOX_TINY         | WARN     | cạnh ngắn của box (px) < `min_box_px`  (cần `ds.image_size(stem)`)       |
| BOX_HUGE         | WARN     | w*h > `max_box_frac`                                                     |

Gợi ý
- Mép trái = x - w/2, mép phải = x + w/2 (xem câu hỏi "dòng YOLO nào lỗi?" trong slide).
- Một dòng hỏng (BAD_FORMAT, BAD_CLASS_ID, BOX_ZERO) thì `continue`, đừng kiểm tiếp hình học của dòng đó.
- Dòng thiếu cột làm script của slide dừng giữa chừng: đây là lý do phải `try/except`.
- `iou_xywh(a, b)` có sẵn trong common.py.
- Ngưỡng nằm trong `cfg["label_format"]` — đừng viết cứng số vào code.
"""
from ..common import ERROR, WARN, Issue, iou_xywh, parse_yolo_line  # noqa: F401


def run(ds, cfg):
    issues = []
    c_fmt = cfg.get("label_format", {})
    overflow_tol = c_fmt.get("overflow_tol", 0.001)
    min_box_px = c_fmt.get("min_box_px", 12)
    max_box_frac = c_fmt.get("max_box_frac", 0.95)
    dup_box_iou = c_fmt.get("dup_box_iou", 0.9)
    num_classes = len(ds.classes)

    for lf in ds.label_files():
        stem = lf.stem
        lines = ds.read_label_lines(stem)
        if lines is None:
            continue
        im_sz = ds.image_size(stem)
        prev_boxes = []

        for line_no, text in lines:
            try:
                c, cx, cy, w, h = parse_yolo_line(text)
            except ValueError:
                issues.append(Issue("label_format", ERROR, "BAD_FORMAT", stem, f"sai định dạng dòng YOLO: {text!r}", line=line_no))
                continue

            if c < 0 or c >= num_classes:
                issues.append(Issue("label_format", ERROR, "BAD_CLASS_ID", stem, f"id lớp {c} không hợp lệ (hợp lệ: 0..{num_classes - 1})", line=line_no, extra={"class_id": c}))
                continue

            if w <= 0 or h <= 0:
                issues.append(Issue("label_format", ERROR, "BOX_ZERO", stem, f"kích thước box <= 0 (w={w}, h={h})", line=line_no, extra={"w": w, "h": h}))
                continue

            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2
            overflow = max(0.0, -x1, -y1, x2 - 1.0, y2 - 1.0)
            if overflow > overflow_tol:
                issues.append(Issue("label_format", ERROR, "BOX_OUT_OF_IMAGE", stem, f"box tràn viền {overflow:.4f} > tol {overflow_tol}", line=line_no, extra={"overflow": overflow}))

            box = (cx, cy, w, h)
            for prev_no, prev_c, prev_box in prev_boxes:
                if prev_c == c:
                    iou = iou_xywh(box, prev_box)
                    if iou > dup_box_iou:
                        issues.append(Issue("label_format", ERROR, "DUP_BOX", stem, f"trùng box với dòng {prev_no} (IoU {iou:.2f})", line=line_no, extra={"dup_with": prev_no, "iou": iou}))
                        break
            prev_boxes.append((line_no, c, box))

            if im_sz is not None:
                img_w, img_h = im_sz
                short_side_px = min(w * img_w, h * img_h)
                if short_side_px < min_box_px:
                    issues.append(Issue("label_format", WARN, "BOX_TINY", stem, f"cạnh ngắn của box {short_side_px:.1f}px < {min_box_px}px", line=line_no, extra={"short_side_px": short_side_px}))

            if w * h > max_box_frac:
                issues.append(Issue("label_format", WARN, "BOX_HUGE", stem, f"diện tích box {w * h:.3f} > max {max_box_frac}", line=line_no, extra={"area_frac": w * h}))

    return issues
