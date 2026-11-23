"""
Script download & chuẩn hóa dataset từ HuggingFace — TV1 EXP-00/EXP-01.

Dataset chính: UIT-ViHSD (ura-hcmut/UIT-ViHSD)
  - Nhãn: CLEAN=0, OFFENSIVE=1, HATE=2 — khớp contract v4 trực tiếp
  - Đã có official train/dev/test split

Đầu ra:
  - data/raw/vihsd_*.jsonl          (bản gốc giữ nguyên schema HF)
  - data/processed/vihsd.jsonl      (schema contract v4: id,text,label,split)
  - data/processed/manifest.json    (id→split mapping)
  - data/processed/dataset_info.json (provenance, hashes, statistics)

Chạy: python -m src.data.download_dataset
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

# Thêm root vào sys.path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_records(records: list[dict]) -> str:
    h = hashlib.sha256()
    for r in records:
        h.update(json.dumps(r, ensure_ascii=False, sort_keys=True).encode())
    return h.hexdigest()


def download_vihsd(raw_dir: Path, processed_dir: Path) -> None:
    """Download ViHSD và xuất theo contract v4."""
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        print("ERROR: datasets chưa được cài. Chạy: pip install datasets huggingface-hub")
        sys.exit(1)

    print("Đang tải UIT-ViHSD từ HuggingFace...")
    ds = load_dataset("ura-hcmut/UIT-ViHSD", trust_remote_code=True)
    print(f"  Splits có sẵn: {list(ds.keys())}")
    print(f"  Features: {ds['train'].features}")

    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Kiểm tra schema gốc
    sample = ds["train"][0]
    print(f"  Sample record: {sample}")

    # Xác định tên cột từ dataset
    # ViHSD thường có: 'free_label' hoặc 'label_id', 'comment' hoặc 'sentence'
    features = ds["train"].features
    text_col = None
    label_col = None

    for col in ["free_label", "label_id", "label"]:
        if col in features:
            label_col = col
            break

    for col in ["comment", "sentence", "text"]:
        if col in features:
            text_col = col
            break

    if text_col is None or label_col is None:
        print(f"WARNING: Không nhận ra cột chuẩn. Columns: {list(features.keys())}")
        print("Các cột có sẵn:", list(features.keys()))
        # Dùng cột đầu tiên có kiểu string và integer
        str_cols = [k for k, v in features.items() if str(v.dtype) in ("string", "str")]
        int_cols = [k for k, v in features.items() if str(v.dtype) in ("int64", "int32", "ClassLabel")]
        if str_cols:
            text_col = str_cols[0]
        if int_cols:
            label_col = int_cols[0]
        print(f"  Dùng text_col={text_col}, label_col={label_col}")

    # Ánh xạ split name HF → contract v4
    split_map = {
        "train": "train",
        "validation": "val",
        "dev": "val",
        "test": "test",
    }

    all_records_dev = []   # train + val (không có test)
    manifest = {}
    split_stats: dict[str, dict] = {}
    raw_hashes = {}

    for hf_split, contract_split in split_map.items():
        if hf_split not in ds:
            continue
        split_ds = ds[hf_split]
        rows = []
        for i, item in enumerate(split_ds):
            text = item.get(text_col, "")
            if not isinstance(text, str):
                text = str(text)
            text = text.strip()

            raw_label = item.get(label_col, -1)
            # Đảm bảo label là int trong {0,1,2}
            if hasattr(raw_label, "item"):
                raw_label = raw_label.item()
            label = int(raw_label)
            if label not in (0, 1, 2):
                print(f"  WARNING: nhãn lạ {label} ở {hf_split}[{i}], bỏ qua")
                continue
            if not text:
                print(f"  WARNING: text rỗng ở {hf_split}[{i}], bỏ qua")
                continue

            # ID: dùng f"{split_prefix}_{i}" để ổn định
            record_id = f"{hf_split}_{i}"
            rows.append({"id": record_id, "text": text, "label": label, "split": contract_split})

        # Lưu raw JSONL
        raw_path = raw_dir / f"vihsd_{hf_split}.jsonl"
        with open(raw_path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        raw_hash = _sha256_file(raw_path)
        raw_hashes[hf_split] = raw_hash
        print(f"  {hf_split} → {len(rows)} records, hash={raw_hash[:16]}...")

        # Thống kê phân phối nhãn
        label_counts = {0: 0, 1: 0, 2: 0}
        for r in rows:
            label_counts[r["label"]] += 1
        split_stats[contract_split] = {
            "n": len(rows),
            "CLEAN": label_counts[0],
            "OFFENSIVE": label_counts[1],
            "HATE": label_counts[2],
        }

        # Chỉ đưa train/val vào development set
        if contract_split in ("train", "val"):
            for r in rows:
                manifest[r["id"]] = r["split"]
                all_records_dev.append(r)

    # Xuất processed JSONL (chỉ train+val, locked test không phát hành cho EXP)
    processed_path = processed_dir / "vihsd.jsonl"
    with open(processed_path, "w", encoding="utf-8") as f:
        for r in all_records_dev:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    processed_hash = _sha256_file(processed_path)

    # Lưu manifest
    manifest_path = processed_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    # Lưu dataset_info (provenance)
    info = {
        "dataset_name": "UIT-ViHSD",
        "source": "ura-hcmut/UIT-ViHSD",
        "huggingface_url": "https://huggingface.co/datasets/ura-hcmut/UIT-ViHSD",
        "label_mapping": {"0": "CLEAN", "1": "OFFENSIVE", "2": "HATE"},
        "text_column_used": text_col,
        "label_column_used": label_col,
        "split_map": split_map,
        "split_stats": split_stats,
        "contract_version": "v4",
        "processed_jsonl_hash": processed_hash,
        "raw_split_hashes": raw_hashes,
        "note": (
            "Official split từ HuggingFace được giữ nguyên. "
            "Locked test không phát hành cho development. "
            "Chỉ fit TF-IDF/vocabulary trên split=train."
        ),
    }
    info_path = processed_dir / "dataset_info.json"
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    print("\n=== Dataset summary ===")
    for split_name, stats in split_stats.items():
        total = stats["n"]
        print(
            f"  {split_name:12s}: {total:5d} records | "
            f"CLEAN={stats['CLEAN']} ({stats['CLEAN']/total*100:.1f}%) | "
            f"OFFENSIVE={stats['OFFENSIVE']} ({stats['OFFENSIVE']/total*100:.1f}%) | "
            f"HATE={stats['HATE']} ({stats['HATE']/total*100:.1f}%)"
        )
    print(f"\n  Processed JSONL: {processed_path}")
    print(f"  Manifest:        {manifest_path}")
    print(f"  Dataset info:    {info_path}")
    print(f"  Processed hash:  {processed_hash}")
    print("\nDone. Dữ liệu sẵn sàng cho EDA và training.")


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[2]
    download_vihsd(
        raw_dir=repo_root / "data" / "raw",
        processed_dir=repo_root / "data" / "processed",
    )
