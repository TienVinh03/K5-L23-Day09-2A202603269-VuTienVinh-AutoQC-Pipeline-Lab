"""CHECK 6 — Ý kiến thứ hai: so nhãn với một model LỚN HƠN (yolo26l).   ★★★ (khó nhất, đáng điểm nhất)

Test:  pytest tests/test_second_opinion.py -q

Máy kiểm quy tắc, người kiểm ý nghĩa (slide). Các lỗi như thiếu nhãn / gán nhầm lớp hợp lệ / box lỏng
KHÔNG bắt được bằng quy tắc cú pháp. Ta nhờ một model mạnh hơn "nhìn lại" và CHỈ CẢNH BÁO (model cũng có thể sai):
những ảnh bị cờ sẽ vào hàng đợi cho NGƯỜI duyệt.

`preds[stem]` = list `[id_lớp, conf, cx, cy, w, h]` của model (lấy qua `get_preds(ds, cfg, "checker")` nếu preds=None).
`L = ds.valid_boxes(stem)` = list `(id_lớp, cx, cy, w, h)` của nhãn (đã bỏ dòng hỏng).
Ngưỡng ở `cfg["second_opinion"]`: conf, min_area_px, fp_conf, iou_missing, iou_match, iou_loose.

Thuật toán gợi ý, với từng ảnh (bỏ qua ảnh không có trong preds hoặc không có file nhãn):
  P = dự đoán có conf >= conf  VÀ diện tích px (w*W * h*H) >= min_area_px
  Với mỗi dự đoán r trong P: lấy các nhãn có IoU(r, nhãn) >= iou_missing  (cands)
   - không có cands                            -> POSSIBLE_MISSING     (model thấy vật mà nhãn không có box)
   - có cands cùng lớp, tốt nhất IoU < iou_loose -> POSSIBLE_LOOSE_BOX  (đúng vật, box lỏng/lệch)
   - không có cands cùng lớp, nhưng cands tốt nhất IoU >= iou_match (khác lớp) -> POSSIBLE_WRONG_CLASS
  Với mỗi dòng nhãn hợp lệ: nếu IoU với MỌI dự đoán (conf >= fp_conf, không lọc diện tích) đều < iou_missing
                                             -> POSSIBLE_FALSE_POS  (box nhãn không khớp vật nào model thấy; Issue.line = số dòng)

Chi tiết các ca biên (test kiểm đúng các ca này)
- `cands` gồm nhãn MỌI lớp có IoU >= iou_missing; "tốt nhất cùng lớp" = cands cùng lớp có IoU lớn nhất; "tốt nhất khác lớp" tương tự với cands khác lớp.
- Thứ tự quyết định cho mỗi dự đoán: (1) cands rỗng -> MISSING. (2) có cands cùng lớp: IoU tốt nhất < iou_loose -> LOOSE_BOX, ngược lại KHÔNG cờ
  (nhãn đúng lớp đã phủ vật này, kể cả khi còn nhãn khác lớp chồng lên). (3) không có cands cùng lớp: nếu IoU tốt nhất khác lớp >= iou_match -> WRONG_CLASS;
  nếu chỉ có khác lớp với IoU trong [iou_missing, iou_match) -> không cờ gì (mơ hồ: để cho FALSE_POS / lần duyệt khác).
- Một dự đoán sinh tối đa MỘT issue (MISSING / LOOSE_BOX / WRONG_CLASS); một ảnh có thể có nhiều issue.
- Thứ tự issue: theo `ds.image_files()` rồi theo thứ tự dự đoán trong `preds[stem]`, cuối cùng là các FALSE_POS theo số dòng; `Issue.image` = stem, `Issue.line` chỉ đặt cho FALSE_POS.

Lưu ý số dòng: dùng `ds.read_label_lines(stem)` (giữ số dòng gốc, kể cả dòng hỏng/trống) để lấy `Issue.line`; `valid_boxes` KHÔNG giữ số dòng.

Tất cả severity WARN; `Issue.extra` nên có {"conf": ...} hoặc {"iou": ...} (hàng đợi duyệt dùng conf để xếp ưu tiên).

Gợi ý
- `iou_xywh` có sẵn. Dòng nhãn hỏng không được làm script chết (dùng `ds.valid_boxes` và `parse_yolo_line` trong try).
- Kết quả phụ thuộc mạnh vào ngưỡng: calibrate chúng ở Bước 3 thay vì đoán.
"""
from ..common import WARN, Issue, iou_xywh, parse_yolo_line  # noqa: F401
from ..model_cache import get_preds  # noqa: F401


def run(ds, cfg, preds=None):
    if preds is None:
        preds = get_preds(ds, cfg, "checker")

    c_so = cfg.get("second_opinion", {})
    conf_thresh = c_so.get("conf", 0.25)
    min_area_px = c_so.get("min_area_px", 600)
    fp_conf = c_so.get("fp_conf", 0.1)
    iou_missing = c_so.get("iou_missing", 0.3)
    iou_match = c_so.get("iou_match", 0.5)
    iou_loose = c_so.get("iou_loose", 0.5)
    num_classes = len(ds.classes)

    issues = []

    for img_path in ds.image_files():
        stem = img_path.stem
        if stem not in preds or not ds.label_path(stem).exists():
            continue

        im_sz = ds.image_size(stem)
        if im_sz is not None:
            w_px, h_px = im_sz
        else:
            w_px, h_px = 1, 1

        all_preds = preds.get(stem, [])

        # P: Các dự đoán có conf >= conf VÀ diện tích px >= min_area_px
        p_filtered = []
        for r in all_preds:
            c, conf, pcx, pcy, pw, ph = r
            area_px = (pw * w_px) * (ph * h_px)
            if conf >= conf_thresh and area_px >= min_area_px:
                p_filtered.append(r)

        # Lấy các nhãn hợp lệ kèm số dòng gốc
        valid_labels = []
        raw_lines = ds.read_label_lines(stem) or []
        for line_no, text in raw_lines:
            try:
                c, lcx, lcy, lw, lh = parse_yolo_line(text)
            except ValueError:
                continue
            if 0 <= c < num_classes:
                valid_labels.append((line_no, c, lcx, lcy, lw, lh))

        # 1. So sánh từng dự đoán r trong P với các nhãn hợp lệ
        for r in p_filtered:
            pred_c, pred_conf, pcx, pcy, pw, ph = r
            pred_box = (pcx, pcy, pw, ph)

            cands = []
            for line_no, lbl_c, lcx, lcy, lw, lh in valid_labels:
                lbl_box = (lcx, lcy, lw, lh)
                iou = iou_xywh(pred_box, lbl_box)
                if iou >= iou_missing:
                    cands.append((lbl_c, iou))

            if not cands:
                issues.append(Issue("second_opinion", WARN, "POSSIBLE_MISSING", stem, f"model dự đoán lớp {pred_c} ({pred_conf:.2f}) nhưng nhãn thiếu", line=None, extra={"conf": pred_conf, "iou": 0.0}))
            else:
                same_class = [cand for cand in cands if cand[0] == pred_c]
                if same_class:
                    best_same_iou = max(cand[1] for cand in same_class)
                    if best_same_iou < iou_loose:
                        issues.append(Issue("second_opinion", WARN, "POSSIBLE_LOOSE_BOX", stem, f"nhãn lớp {pred_c} lệch so với dự đoán (IoU {best_same_iou:.2f} < {iou_loose})", line=None, extra={"iou": best_same_iou, "conf": pred_conf}))
                else:
                    best_diff_iou = max(cand[1] for cand in cands)
                    if best_diff_iou >= iou_match:
                        issues.append(Issue("second_opinion", WARN, "POSSIBLE_WRONG_CLASS", stem, f"nhãn khác lớp nhưng khớp vị trí (IoU {best_diff_iou:.2f} >= {iou_match})", line=None, extra={"iou": best_diff_iou, "conf": pred_conf}))

        # 2. Kiểm tra POSSIBLE_FALSE_POS cho từng dòng nhãn hợp lệ
        fp_preds = [p for p in all_preds if p[1] >= fp_conf]
        for line_no, lbl_c, lcx, lcy, lw, lh in valid_labels:
            lbl_box = (lcx, lcy, lw, lh)
            match_found = False
            for p in fp_preds:
                pred_box = (p[2], p[3], p[4], p[5])
                if iou_xywh(lbl_box, pred_box) >= iou_missing:
                    match_found = True
                    break
            if not match_found:
                issues.append(Issue("second_opinion", WARN, "POSSIBLE_FALSE_POS", stem, f"nhãn không khớp dự đoán nào của model", line=line_no, extra={"iou": 0.0}))

    return issues
