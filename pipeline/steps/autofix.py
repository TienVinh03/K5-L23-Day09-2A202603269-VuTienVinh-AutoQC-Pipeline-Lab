"""BƯỚC 2/4 — Sửa tự động những lỗi bắt buộc mà máy sửa được AN TOÀN.   ★★

Test:  pytest tests/test_autofix.py -q

`fix(ds, issues, cfg) -> list[dict]`: duyệt `issues` (chỉ lỗi severity ERROR), sửa thẳng vào thư mục `ds`,
và trả về danh sách thay đổi; mỗi phần tử có khoá: image, line, action, before, after
(đây chính là file `03_fixes.csv` — bằng chứng pipeline đã làm gì, người có thể kiểm lại).

| code lỗi                                       | cách sửa                                    | action           |
|------------------------------------------------|---------------------------------------------|------------------|
| BAD_FORMAT, BAD_CLASS_ID, BOX_ZERO, DUP_BOX    | xoá dòng đó khỏi file nhãn                  | "drop_line"      |
| BOX_OUT_OF_IMAGE                               | cắt box về trong [0,1] (giữ nguyên lớp)     | "clip_box"       |
| MISSING_LABEL_FILE                             | KHÔNG sửa file (máy không thể bịa nhãn): ghi 1 dòng fix, before="(không có file)", after="chuyển người gán" | "needs_label" |
| BAD_CLASS_NAME                                 | ghi lại `names` trong data.yaml đúng `cfg["classes"]` | "rename_classes" |
| DUP_IMAGE_*, LEAK_TEST, CORRUPT_IMAGE, BAD_SIZE| xoá ảnh + nhãn khỏi `ds`                    | "remove_image"   |

Bẫy cần nghĩ tới (có test cho từng cái)
- Một file có NHIỀU dòng lỗi: số dòng trong `issues` là số dòng GỐC. Nếu xoá từng dòng một thì số dòng lệch.
  Gom theo file, rồi ghi lại MỘT lần.
- Sau khi cắt, box có thể không còn diện tích (nằm hẳn ngoài ảnh): lúc đó xoá dòng thay vì ghi box rỗng.
- Một dòng có 2 lỗi (vd. BOX_OUT_OF_IMAGE + DUP_BOX): xoá dòng thắng, chỉ ghi 1 fix `drop_line`, không ghi thêm `clip_box`. Dòng trống ở giữa file vẫn đếm vào số dòng gốc.
- Ảnh vừa bị xoá (DUP_IMAGE_*...) mà còn issue dòng của chính nó: bỏ qua êm, không được báo lỗi/chết.
- Không động tới lỗi WARN (cảnh báo là việc của người). Không sửa file ở `data/raw/` — chỉ sửa `ds`.
- Chạy lần 2 trên dữ liệu đã sửa phải không làm gì thêm (idempotent) — NGOẠI LỆ duy nhất: `needs_label` được ghi lại mỗi lần còn lỗi `MISSING_LABEL_FILE` (vì ảnh vẫn chưa có nhãn).

- `needs_label` KHÔNG tạo file nhãn rỗng (rỗng = "ảnh nền", khác với "chưa gán"). `review_queue` sẽ đẩy ảnh này lên đầu hàng đợi cho người.

Gợi ý: `yaml.safe_load / safe_dump`.
"""
from collections import defaultdict  # noqa: F401

import yaml  # noqa: F401

from ..common import ERROR, parse_yolo_line  # noqa: F401


def fix(ds, issues, cfg):
    error_issues = [i for i in issues if i.severity == ERROR or i.severity == "error"]
    if not error_issues:
        return []

    fixes = []
    removed_images = set()

    # 1. Sửa BAD_CLASS_NAME trong data.yaml
    if any(i.code == "BAD_CLASS_NAME" for i in error_issues):
        if ds.yaml_path.exists():
            try:
                d = yaml.safe_load(ds.yaml_path.read_text(encoding="utf-8")) or {}
            except Exception:
                d = {}
            old_names = d.get("names", {})
            d["names"] = list(cfg["classes"])
            ds.yaml_path.write_text(yaml.safe_dump(d, sort_keys=False), encoding="utf-8")
            fixes.append({
                "image": "data.yaml",
                "line": "",
                "action": "rename_classes",
                "before": str(old_names),
                "after": str(cfg["classes"]),
            })

    # 2. Xóa ảnh + nhãn đối với các lỗi ảnh/trùng/hỏng/rò rỉ
    remove_codes = {"DUP_IMAGE_EXACT", "DUP_IMAGE_NEAR", "LEAK_TEST", "CORRUPT_IMAGE", "BAD_SIZE"}
    for iss in error_issues:
        if iss.code in remove_codes or iss.code.startswith("DUP_IMAGE_"):
            stem = iss.image
            if stem and stem not in removed_images and stem not in ("data.yaml", "<dataset>"):
                removed_images.add(stem)
                for p in ds.image_files():
                    if p.stem == stem:
                        try:
                            p.unlink(missing_ok=True)
                        except Exception:
                            pass
                for ext in (".jpg", ".jpeg", ".png", ".JPG", ".PNG"):
                    p = ds.images_dir / f"{stem}{ext}"
                    if p.exists():
                        try:
                            p.unlink(missing_ok=True)
                        except Exception:
                            pass
                lp = ds.label_path(stem)
                if lp.exists():
                    try:
                        lp.unlink(missing_ok=True)
                    except Exception:
                        pass
                if hasattr(ds, "_size_cache"):
                    ds._size_cache.pop(stem, None)
                fixes.append({
                    "image": stem,
                    "line": "",
                    "action": "remove_image",
                    "before": "giữ",
                    "after": "xoá khỏi dataset",
                })

    # 3. Ghi nhận ảnh thiếu nhãn (MISSING_LABEL_FILE) -> chuyển người gán, không tạo file
    seen_missing = set()
    for iss in error_issues:
        if iss.code == "MISSING_LABEL_FILE":
            stem = iss.image
            if stem and stem not in removed_images and stem not in seen_missing:
                seen_missing.add(stem)
                fixes.append({
                    "image": stem,
                    "line": "",
                    "action": "needs_label",
                    "before": "(không có file)",
                    "after": "chuyển người gán",
                })

    # 4. Gom các lỗi dòng nhãn theo từng ảnh
    drop_codes = {"BAD_FORMAT", "BAD_CLASS_ID", "BOX_ZERO", "DUP_BOX"}
    clip_codes = {"BOX_OUT_OF_IMAGE"}

    file_issues = defaultdict(list)
    for iss in error_issues:
        if iss.image and iss.image not in removed_images and iss.image not in ("data.yaml", "<dataset>"):
            if iss.line is not None and iss.code in (drop_codes | clip_codes):
                file_issues[iss.image].append(iss)

    for stem in sorted(file_issues.keys()):
        lbl_path = ds.label_path(stem)
        if not lbl_path.exists():
            continue

        drop_lines = set()
        clip_lines = set()
        for iss in file_issues[stem]:
            if iss.line is not None:
                if iss.code in drop_codes:
                    drop_lines.add(iss.line)
                elif iss.code in clip_codes:
                    clip_lines.add(iss.line)

        clip_lines = clip_lines - drop_lines

        raw_lines = lbl_path.read_text(encoding="utf-8", errors="replace").splitlines()
        new_lines = []

        for line_idx, line_str in enumerate(raw_lines, 1):
            if line_idx in drop_lines:
                fixes.append({
                    "image": stem,
                    "line": line_idx,
                    "action": "drop_line",
                    "before": line_str,
                    "after": "",
                })
            elif line_idx in clip_lines:
                try:
                    c, cx, cy, w, h = parse_yolo_line(line_str)
                    x1 = cx - w / 2
                    y1 = cy - h / 2
                    x2 = cx + w / 2
                    y2 = cy + h / 2

                    x1_c = max(0.0, min(1.0, x1))
                    y1_c = max(0.0, min(1.0, y1))
                    x2_c = max(0.0, min(1.0, x2))
                    y2_c = max(0.0, min(1.0, y2))

                    new_w = x2_c - x1_c
                    new_h = y2_c - y1_c

                    if new_w <= 0.0 or new_h <= 0.0:
                        fixes.append({
                            "image": stem,
                            "line": line_idx,
                            "action": "drop_line",
                            "before": line_str,
                            "after": "",
                        })
                    else:
                        new_cx = (x1_c + x2_c) / 2
                        new_cy = (y1_c + y2_c) / 2
                        new_line_str = f"{c} {new_cx:.6f} {new_cy:.6f} {new_w:.6f} {new_h:.6f}"
                        new_lines.append(new_line_str)
                        fixes.append({
                            "image": stem,
                            "line": line_idx,
                            "action": "clip_box",
                            "before": line_str,
                            "after": new_line_str,
                        })
                except ValueError:
                    fixes.append({
                        "image": stem,
                        "line": line_idx,
                        "action": "drop_line",
                        "before": line_str,
                        "after": "",
                    })
            else:
                new_lines.append(line_str)

        if new_lines:
            lbl_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
        else:
            lbl_path.write_text("", encoding="utf-8")

    return fixes
