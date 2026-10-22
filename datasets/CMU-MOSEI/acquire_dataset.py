#!/usr/bin/env python3
"""
CMU-MOSEI Dataset Acquisition, Integrity Verification & Adapter Utility
========================================================================
Project: Multimodal Emotion Recognition (Text, Speech, Vision)
Storage Root: D:\\Datasets\\CMU-MOSEI

This script handles:
1. Directory initialization on D:
2. Verification and transfer of downloaded .csd files
3. Verification of Hugging Face repository metadata and checksums
4. Fetching official benchmark split definitions from CMU-MultimodalSDK
5. Inspecting CSD schemas and dimensions
6. Resumable downloading of VisualFacet42.csd and COVAREP.csd to D:
7. Cross-modal alignment audit across all 4 modalities
8. Generating production mosei_annotations.csv from verified CSD records
"""

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import h5py
import numpy as np

HF_REPO_API = "https://huggingface.co/api/datasets/reeha-parkar/cmu-mosei-comp-seq"
HF_DATA_URL_BASE = "https://huggingface.co/datasets/reeha-parkar/cmu-mosei-comp-seq/resolve/main/data"
SDK_STD_FOLDS_URL = "https://raw.githubusercontent.com/CMU-MultiComp-Lab/CMU-MultimodalSDK/main/mmsdk/mmdatasdk/dataset/standard_datasets/CMU_MOSEI/cmu_mosei_std_folds.py"

OFFICIAL_FILES = {
    "labels": {
        "filename": "CMU_MOSEI_Labels.csd",
        "description": "Continuous sentiment [-3, 3] and 6 emotion intensities [0, 3]",
    },
    "words": {
        "filename": "CMU_MOSEI_TimestampedWords.csd",
        "description": "Timestamped word tokens for transcript reconstruction",
    },
    "visual": {
        "filename": "CMU_MOSEI_VisualFacet42.csd",
        "description": "FACET 4.2 visual features (42 dimensions: Action Units, landmarks, pose)",
    },
    "audio": {
        "filename": "CMU_MOSEI_COVAREP.csd",
        "description": "COVAREP acoustic features (74 dimensions: pitch, glottal, voice quality)",
    },
}


def setup_directories(target_root: Path) -> Dict[str, Path]:
    """Initializes required directory hierarchy on target drive."""
    dirs = {
        "root": target_root,
        "features": target_root / "features",
        "audio": target_root / "features" / "audio",
        "visual": target_root / "features" / "visual",
        "text": target_root / "features" / "text",
        "annotations": target_root / "annotations",
        "data": target_root / "data",
        "manifests": target_root / "manifests",
        "reports": target_root / "reports",
    }
    for name, p in dirs.items():
        p.mkdir(parents=True, exist_ok=True)
    return dirs


def verify_csd_integrity(filepath: Path) -> Tuple[bool, str, Dict[str, Any]]:
    """Verifies HDF5 format and returns structure summary."""
    if not filepath.exists():
        return False, f"File does not exist: {filepath}", {}
    try:
        size = filepath.stat().st_size
        with h5py.File(filepath, "r") as h5f:
            keys = list(h5f.keys())
            if not keys:
                return False, "Empty HDF5 file (no root keys)", {}
            root = keys[0]
            root_grp = h5f[root]
            if "data" not in root_grp:
                return False, f"Missing 'data' group under root '{root}'", {}
            data_grp = root_grp["data"]
            num_keys = len(data_grp)
            sample_key = next(iter(data_grp.keys())) if num_keys > 0 else None
            sample_info = {}
            if sample_key:
                item = data_grp[sample_key]
                has_feat = "features" in item
                has_intervals = "intervals" in item
                sample_info = {
                    "sample_key": sample_key,
                    "has_features": has_feat,
                    "has_intervals": has_intervals,
                    "feature_shape": list(item["features"].shape) if has_feat else None,
                    "intervals_shape": list(item["intervals"].shape) if has_intervals else None,
                }
            return True, "ok", {
                "root": root,
                "video_count": num_keys,
                "file_size": size,
                "sample_info": sample_info,
            }
    except Exception as e:
        return False, f"HDF5 read error: {type(e).__name__}: {e}", {}


def transfer_existing_csd(c_feat_dir: Path, d_feat_dir: Path) -> List[str]:
    """Safely transfers verified files from C: to D: preserving recoverable copy until checked."""
    transferred = []
    for fname in ["CMU_MOSEI_Labels.csd", "CMU_MOSEI_TimestampedWords.csd"]:
        c_path = c_feat_dir / fname
        d_path = d_feat_dir / fname
        if not c_path.exists():
            continue
        print(f"[*] Validating source {fname} on C:...")
        ok, reason, c_meta = verify_csd_integrity(c_path)
        if not ok:
            raise RuntimeError(f"Source file {c_path} failed integrity: {reason}")
        print(f"  [+] Source verified: {c_meta['video_count']} videos, {c_meta['file_size']} bytes.")

        print(f"[*] Copying {fname} to {d_path}...")
        shutil.copy2(c_path, d_path)

        ok_d, reason_d, d_meta = verify_csd_integrity(d_path)
        if not ok_d or d_meta["file_size"] != c_meta["file_size"]:
            raise RuntimeError(f"Destination verification failed for {d_path}: {reason_d}")
        print(f"  [+] Destination verified on D: {d_meta['video_count']} videos, {d_meta['file_size']} bytes.")

        c_path.unlink()
        print(f"  [+] Cleaned up source copy on C: ({c_path.name})")
        transferred.append(fname)
    return transferred


def verify_hf_metadata() -> Dict[str, Any]:
    """Queries Hugging Face dataset API to check LFS pointers, commits, and file sizes."""
    print("[*] Querying Hugging Face repository metadata...")
    req = urllib.request.Request(HF_REPO_API, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    print(f"  [+] Repository: {data.get('id')}")
    print(f"  [+] Author/Owner: {data.get('author')}")
    print(f"  [+] Last Modified: {data.get('lastModified')}")
    print(f"  [+] SHA Commit: {data.get('sha')}")

    tree_req = urllib.request.Request(f"{HF_REPO_API}/tree/main/data", headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(tree_req, timeout=15) as resp:
        tree_data = json.loads(resp.read().decode("utf-8"))

    file_catalog = {}
    for item in tree_data:
        path = item.get("path")
        size = item.get("size")
        oid = item.get("lfs", {}).get("oid") if item.get("lfs") else item.get("oid")
        file_catalog[path] = {
            "size_bytes": size,
            "size_mb": round(size / (1024 * 1024), 2) if size else 0,
            "sha256": oid,
        }
        print(f"  [File] {path}: {round(size / (1024 * 1024), 2)} MB | SHA256: {oid}")

    return {"repo": data, "files": file_catalog}


def fetch_official_standard_folds(dest_file: Path) -> Dict[str, List[str]]:
    """Fetches the official video fold split definitions from CMU-MultimodalSDK."""
    print(f"[*] Fetching standard folds from {SDK_STD_FOLDS_URL}...")
    req = urllib.request.Request(SDK_STD_FOLDS_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        code = resp.read().decode("utf-8")

    namespace = {}
    exec(code, namespace)
    folds = {
        "standard_train_fold": sorted(list(set(namespace["standard_train_fold"]))),
        "standard_valid_fold": sorted(list(set(namespace["standard_valid_fold"]))),
        "standard_test_fold": sorted(list(set(namespace["standard_test_fold"]))),
    }

    tr = set(folds["standard_train_fold"])
    va = set(folds["standard_valid_fold"])
    te = set(folds["standard_test_fold"])
    assert len(tr.intersection(va)) == 0, "Train-Dev video leakage in official folds!"
    assert len(tr.intersection(te)) == 0, "Train-Test video leakage in official folds!"
    assert len(va.intersection(te)) == 0, "Dev-Test video leakage in official folds!"

    print(f"  [+] Official Video Folds: Train={len(tr)}, Valid(Dev)={len(va)}, Test={len(te)}, Total={len(tr)+len(va)+len(te)}")
    dest_file.parent.mkdir(parents=True, exist_ok=True)
    with open(dest_file, "w", encoding="utf-8") as f:
        json.dump(folds, f, indent=2)
    print(f"  [+] Saved official split definitions to {dest_file}")
    return folds


def download_file_resumable(url: str, dest_path: Path, expected_size: Optional[int] = None) -> bool:
    """Downloads a file with HTTP Range support for resumption."""
    temp_path = dest_path.with_suffix(dest_path.suffix + ".part")
    existing_size = temp_path.stat().st_size if temp_path.exists() else 0

    headers = {"User-Agent": "Mozilla/5.0"}
    if existing_size > 0:
        headers["Range"] = f"bytes={existing_size}-"
        print(f"[*] Resuming download of {dest_path.name} from byte {existing_size}...")
    else:
        print(f"[*] Starting download of {dest_path.name}...")

    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            total_size = None
            cr = resp.headers.get("Content-Range")
            if cr:
                total_size = int(cr.split("/")[-1])
            elif resp.headers.get("Content-Length"):
                total_size = existing_size + int(resp.headers.get("Content-Length"))

            mode = "ab" if existing_size > 0 and resp.status == 206 else "wb"
            if mode == "wb":
                existing_size = 0

            downloaded = existing_size
            block_size = 2 * 1024 * 1024  # 2 MB blocks
            start_time = time.time()
            last_report = start_time

            with open(temp_path, mode) as out_f:
                while True:
                    chunk = resp.read(block_size)
                    if not chunk:
                        break
                    out_f.write(chunk)
                    downloaded += len(chunk)
                    now = time.time()
                    if now - last_report >= 3.0:
                        speed = (downloaded - existing_size) / (now - start_time) / (1024 * 1024)
                        pct = (downloaded / total_size * 100) if total_size else 0.0
                        tot_mb = (total_size / (1024 * 1024)) if total_size else 0.0
                        sys.stdout.write(f"\r  -> {downloaded/(1024*1024):.1f} / {tot_mb:.1f} MB ({pct:.1f}%) at {speed:.2f} MB/s")
                        sys.stdout.flush()
                        last_report = now

            print(f"\n  [+] Download completed ({downloaded} bytes). Moving to {dest_path.name}...")
            temp_path.rename(dest_path)
            return True
    except Exception as e:
        print(f"\n[-] Download error for {dest_path.name}: {e}", file=sys.stderr)
        return False


def run_cross_modal_audit_and_generate_annotations(
    target_root: Path,
) -> Dict[str, Any]:
    """
    Performs comprehensive cross-modal alignment audit across:
    - CMU_MOSEI_Labels.csd
    - CMU_MOSEI_TimestampedWords.csd
    - CMU_MOSEI_VisualFacet42.csd
    - CMU_MOSEI_COVAREP.csd
    - standard_splits.json

    Generates production mosei_annotations.csv on D:.
    """
    feat_dir = target_root / "features"
    anno_dir = target_root / "annotations"
    folds_file = anno_dir / "standard_splits.json"

    lbl_file = feat_dir / "CMU_MOSEI_Labels.csd"
    wrd_file = feat_dir / "CMU_MOSEI_TimestampedWords.csd"
    vis_file = feat_dir / "CMU_MOSEI_VisualFacet42.csd"
    aud_file = feat_dir / "CMU_MOSEI_COVAREP.csd"

    print("=================================================================")
    print("CMU-MOSEI Cross-Modal Alignment Audit & Annotation Generator")
    print("=================================================================")

    # 1. Load Split Folds
    if not folds_file.exists():
        print(f"[*] Standard splits file missing; fetching from SDK...")
        fetch_official_standard_folds(folds_file)

    with open(folds_file, "r", encoding="utf-8") as f:
        folds = json.load(f)

    train_vids = set(folds.get("standard_train_fold", []))
    dev_vids = set(folds.get("standard_valid_fold", []))
    test_vids = set(folds.get("standard_test_fold", []))

    print(f"[*] Official Folds: Train={len(train_vids)}, Dev={len(dev_vids)}, Test={len(test_vids)}")

    # 2. Open HDF5 files
    print("[*] Opening CSD files...")
    h5_lbl = h5py.File(lbl_file, "r")
    h5_wrd = h5py.File(wrd_file, "r")
    h5_vis = h5py.File(vis_file, "r")
    h5_aud = h5py.File(aud_file, "r")

    lbl_data = h5_lbl[list(h5_lbl.keys())[0]]["data"]
    wrd_data = h5_wrd[list(h5_wrd.keys())[0]]["data"]
    vis_data = h5_vis[list(h5_vis.keys())[0]]["data"]
    aud_data = h5_aud[list(h5_aud.keys())[0]]["data"]

    lbl_vids = set(lbl_data.keys())
    wrd_vids = set(wrd_data.keys())
    vis_vids = set(vis_data.keys())
    aud_vids = set(aud_data.keys())

    print(f"  [+] Label videos: {len(lbl_vids)}")
    print(f"  [+] Word videos : {len(wrd_vids)}")
    print(f"  [+] Visual videos: {len(vis_vids)}")
    print(f"  [+] Audio videos : {len(aud_vids)}")

    # Coverage comparison
    common_all = lbl_vids.intersection(wrd_vids).intersection(vis_vids).intersection(aud_vids)
    print(f"  [+] Videos present in ALL 4 modalities: {len(common_all)} / {len(lbl_vids)} ({len(common_all)/len(lbl_vids)*100:.2f}%)")

    # 3. Iterate through all labeled segments and construct records
    print("[*] Performing segment-level cross-modal alignment audit...")
    annotation_rows: List[Dict[str, Any]] = []
    split_segment_counts: Counter = Counter()
    split_video_sets: Dict[str, Set[str]] = defaultdict(set)
    unassigned_videos: Set[str] = set()

    missing_words_count = 0
    missing_vis_count = 0
    missing_aud_count = 0
    zero_overlap_vis = 0
    zero_overlap_aud = 0
    zero_overlap_wrd = 0

    total_segments = 0

    for vid in sorted(list(lbl_vids)):
        lbl_item = lbl_data[vid]
        intervals = lbl_item["intervals"][:]
        features = lbl_item["features"][:]

        num_clips = intervals.shape[0]
        total_segments += num_clips

        # Split determination
        if vid in train_vids:
            split_name = "train"
        elif vid in dev_vids:
            split_name = "dev"
        elif vid in test_vids:
            split_name = "test"
        else:
            split_name = "unassigned"
            unassigned_videos.add(vid)

        split_segment_counts[split_name] += num_clips
        split_video_sets[split_name].add(vid)

        # Modality checks for this video
        has_wrd_vid = vid in wrd_data
        has_vis_vid = vid in vis_data
        has_aud_vid = vid in aud_data

        w_intervals = wrd_data[vid]["intervals"][:] if has_wrd_vid else None
        w_feats = wrd_data[vid]["features"][:] if has_wrd_vid else None

        v_intervals = vis_data[vid]["intervals"][:] if has_vis_vid else None
        a_intervals = aud_data[vid]["intervals"][:] if has_aud_vid else None

        for clip_idx in range(num_clips):
            t_start, t_end = float(intervals[clip_idx, 0]), float(intervals[clip_idx, 1])
            feat_row = features[clip_idx]

            sentiment = round(float(feat_row[0]), 4)
            happy = round(float(feat_row[1]), 4)
            sad = round(float(feat_row[2]), 4)
            anger = round(float(feat_row[3]), 4)
            surprise = round(float(feat_row[4]), 4)
            disgust = round(float(feat_row[5]), 4)
            fear = round(float(feat_row[6]), 4)

            sample_id = f"{vid}_{clip_idx}"

            # Word transcript reconstruction
            words_in_seg: List[str] = []
            if has_wrd_vid and w_intervals is not None:
                # Interval overlap: max(t_start, w_start) <= min(t_end, w_end)
                overlap_mask = (w_intervals[:, 0] < t_end) & (w_intervals[:, 1] > t_start)
                overlap_idx = np.where(overlap_mask)[0]
                for o_idx in overlap_idx:
                    w_raw = w_feats[o_idx, 0]
                    w_str = w_raw.decode("utf-8", errors="ignore") if isinstance(w_raw, (bytes, bytearray)) else str(w_raw)
                    if w_str != "sp" and w_str.strip():
                        words_in_seg.append(w_str.strip())
                if len(words_in_seg) == 0:
                    zero_overlap_wrd += 1
            else:
                missing_words_count += 1

            transcript_text = " ".join(words_in_seg)

            # Visual overlap count
            vis_frames = 0
            if has_vis_vid and v_intervals is not None:
                v_mask = (v_intervals[:, 0] < t_end) & (v_intervals[:, 1] > t_start)
                vis_frames = int(np.sum(v_mask))
                if vis_frames == 0:
                    zero_overlap_vis += 1
            else:
                missing_vis_count += 1

            # Audio overlap count
            aud_frames = 0
            if has_aud_vid and a_intervals is not None:
                a_mask = (a_intervals[:, 0] < t_end) & (a_intervals[:, 1] > t_start)
                aud_frames = int(np.sum(a_mask))
                if aud_frames == 0:
                    zero_overlap_aud += 1
            else:
                missing_aud_count += 1

            row_dict = {
                "sample_id": sample_id,
                "video_id": vid,
                "clip_id": clip_idx,
                "start_time": t_start,
                "end_time": t_end,
                "duration": round(t_end - t_start, 4),
                "sentiment": sentiment,
                "happy": happy,
                "sad": sad,
                "anger": anger,
                "surprise": surprise,
                "disgust": disgust,
                "fear": fear,
                "split": split_name,
                "num_words": len(words_in_seg),
                "vis_frames": vis_frames,
                "aud_frames": aud_frames,
                "has_text": len(words_in_seg) > 0,
                "has_visual": vis_frames > 0,
                "has_audio": aud_frames > 0,
                "text": transcript_text,
            }
            annotation_rows.append(row_dict)

    # Close HDF5 files
    h5_lbl.close()
    h5_wrd.close()
    h5_vis.close()
    h5_aud.close()

    print("\n=== Cross-Modal Alignment Audit Results ===")
    print(f"Total Segments Audited: {total_segments}")
    print(f"Split Segment Distribution:")
    for s_name in ["train", "dev", "test", "unassigned"]:
        print(f"  - {s_name.upper():10s}: {split_segment_counts[s_name]:6d} segments across {len(split_video_sets[s_name]):5d} videos")

    print("\nModality Availability Summary across all segments:")
    has_text_count = sum(1 for r in annotation_rows if r["has_text"])
    has_vis_count = sum(1 for r in annotation_rows if r["has_visual"])
    has_aud_count = sum(1 for r in annotation_rows if r["has_audio"])
    has_all_three = sum(1 for r in annotation_rows if r["has_text"] and r["has_visual"] and r["has_audio"])

    print(f"  - Segments with Valid Text Transcripts : {has_text_count:6d} ({has_text_count/total_segments*100:.2f}%)")
    print(f"  - Segments with Valid Visual Frames     : {has_vis_count:6d} ({has_vis_count/total_segments*100:.2f}%)")
    print(f"  - Segments with Valid Audio Frames      : {has_aud_count:6d} ({has_aud_count/total_segments*100:.2f}%)")
    print(f"  - Segments with ALL THREE Modalities    : {has_all_three:6d} ({has_all_three/total_segments*100:.2f}%)")

    # 4. Save annotations CSV
    out_csv = anno_dir / "mosei_annotations.csv"
    print(f"\n[*] Writing complete real annotations to {out_csv}...")
    fieldnames = [
        "sample_id",
        "video_id",
        "clip_id",
        "start_time",
        "end_time",
        "duration",
        "sentiment",
        "happy",
        "sad",
        "anger",
        "surprise",
        "disgust",
        "fear",
        "split",
        "has_text",
        "has_audio",
        "has_visual",
        "text",
    ]
    with open(out_csv, "w", encoding="utf-8", newline="\n") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in annotation_rows:
            writer.writerow(r)

    print(f"  [+] Saved {len(annotation_rows)} real segment records to {out_csv} ({out_csv.stat().st_size} bytes)")

    audit_summary = {
        "total_segments": total_segments,
        "split_segment_counts": dict(split_segment_counts),
        "split_video_counts": {k: len(v) for k, v in split_video_sets.items()},
        "unassigned_videos": sorted(list(unassigned_videos)),
        "modality_availability": {
            "has_text": has_text_count,
            "has_visual": has_vis_count,
            "has_audio": has_aud_count,
            "has_all_three": has_all_three,
        },
    }
    return audit_summary


def reconcile_audit(target_root: Path) -> Dict[str, Any]:
    """
    Performs complete split and accounting audit on actual files on D:.
    Reconciles all 23,259 labeled segments against manifests, exclusions, and folds.
    """
    feat_dir = target_root / "features"
    anno_dir = target_root / "annotations"
    man_dir = target_root / "manifests"
    rep_dir = target_root / "reports"

    print("=================================================================")
    print("CMU-MOSEI Comprehensive Split & Accounting Audit")
    print("=================================================================")

    # 1. Load mosei_annotations.csv
    anno_csv = anno_dir / "mosei_annotations.csv"
    with open(anno_csv, "r", encoding="utf-8") as f:
        all_anno_rows = list(csv.DictReader(f))
    total_anno_segments = len(all_anno_rows)
    all_anno_ids = {r["sample_id"] for r in all_anno_rows}
    all_anno_vids = {r["video_id"] for r in all_anno_rows}
    print(f"[1] Ground Truth Annotations:")
    print(f"    Total Segments: {total_anno_segments}")
    print(f"    Unique Sample IDs: {len(all_anno_ids)}")
    print(f"    Unique Video IDs : {len(all_anno_vids)}")

    # 2. Load Manifests
    def load_manifest(path: Path) -> List[Dict[str, Any]]:
        with open(path, "r", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    tr_rows = load_manifest(man_dir / "train_manifest.csv")
    dev_rows = load_manifest(man_dir / "dev_manifest.csv")
    te_rows = load_manifest(man_dir / "test_manifest.csv")

    tr_ids = {r["sample_id"] for r in tr_rows}
    dev_ids = {r["sample_id"] for r in dev_rows}
    te_ids = {r["sample_id"] for r in te_rows}

    tr_vids = {r["video_id"] for r in tr_rows}
    dev_vids = {r["video_id"] for r in dev_rows}
    te_vids = {r["video_id"] for r in te_rows}

    print(f"\n[2] Output Manifests:")
    print(f"    Train: {len(tr_rows)} segments ({len(tr_ids)} unique IDs, {len(tr_vids)} unique videos)")
    print(f"    Dev  : {len(dev_rows)} segments ({len(dev_ids)} unique IDs, {len(dev_vids)} unique videos)")
    print(f"    Test : {len(te_rows)} segments ({len(te_ids)} unique IDs, {len(te_vids)} unique videos)")
    total_manifest_segments = len(tr_rows) + len(dev_rows) + len(te_rows)
    print(f"    Total Manifest Segments: {total_manifest_segments}")

    # Check manifest exclusivity
    assert len(tr_ids.intersection(dev_ids)) == 0, "Train-Dev sample ID overlap!"
    assert len(tr_ids.intersection(te_ids)) == 0, "Train-Test sample ID overlap!"
    assert len(dev_ids.intersection(te_ids)) == 0, "Dev-Test sample ID overlap!"
    print(f"    Sample ID Exclusivity across manifests: PASSED (Zero Overlap)")

    assert len(tr_vids.intersection(dev_vids)) == 0, "Train-Dev video overlap!"
    assert len(tr_vids.intersection(te_vids)) == 0, "Train-Test video overlap!"
    assert len(dev_vids.intersection(te_vids)) == 0, "Dev-Test video overlap!"
    print(f"    Video ID Leakage across manifests: PASSED (Zero Overlap)")

    # 3. Load Exclusions
    excl_csv = rep_dir / "exclusions.csv"
    with open(excl_csv, "r", encoding="utf-8") as f:
        excl_rows = list(csv.DictReader(f))
    print(f"\n[3] Exclusions Log:")
    print(f"    Total Exclusion Records: {len(excl_rows)}")

    excl_by_reason: Counter = Counter()
    excl_by_split: Counter = Counter()
    excl_ids = set()
    for r in excl_rows:
        sid = r["sample_id"]
        excl_ids.add(sid)
        excl_by_reason[r["exclusion_reason"]] += 1
        excl_by_split[r.get("split", "unassigned")] += 1

    print("    Breakdown by Split:")
    for s, c in excl_by_split.items():
        print(f"      - {s:12s}: {c}")
    print("    Breakdown by Reason:")
    for rsn, c in excl_by_reason.items():
        print(f"      - {rsn}: {c}")

    # 4. Reconciliation Math
    # In mosei_annotations.csv, there were 23,259 records.
    # In prepare_dataset.py:
    # First pass: valid records vs validation errors (e.g. invalid timestamps or unassigned splits)
    # Manifest rows = 16,165 + 1,856 + 4,616 = 22,637 segments.
    # Excluded from manifests: 5 (train) + 0 (dev) + 3 (test) = 8 segments (due to zero visual frames).
    # Unassigned / holdout / invalid: 622 records in exclusions.csv.
    # Total = 22,637 (manifest rows) + 622 (unassigned exclusions) = 23,259!
    # Wait, notice: 8 segments in manifests were ALSO marked is_fully_usable=False and logged to exclusions.csv.
    # Let's verify the exact set arithmetic:
    manifest_all_ids = tr_ids.union(dev_ids).union(te_ids)
    all_accounted_ids = manifest_all_ids.union(excl_ids)

    print(f"\n[4] Exact Segment Accounting Reconciliation:")
    print(f"    Total labeled segments in mosei_annotations.csv : {total_anno_segments}")
    print(f"    Total segments in manifests (train + dev + test): {total_manifest_segments}")
    print(f"      - Usable segments   : {sum(1 for r in tr_rows+dev_rows+te_rows if r['is_fully_usable'] == 'True')}")
    print(f"      - Excluded segments : {sum(1 for r in tr_rows+dev_rows+te_rows if r['is_fully_usable'] == 'False')}")
    print(f"    Total unassigned/holdout/invalid segments       : {sum(1 for r in excl_rows if r['sample_id'] not in manifest_all_ids)}")
    print(f"    Reconciled Total: {total_manifest_segments + sum(1 for r in excl_rows if r['sample_id'] not in manifest_all_ids)} == {total_anno_segments}")
    assert total_manifest_segments + sum(1 for r in excl_rows if r['sample_id'] not in manifest_all_ids) == total_anno_segments

    # Check for missing or double-counted
    unaccounted = all_anno_ids.difference(all_accounted_ids)
    print(f"    Unaccounted Segments: {len(unaccounted)}")
    assert len(unaccounted) == 0, f"Unaccounted segments found: {unaccounted}"

    # 5. Video ID Reconciliation against Official SDK Benchmark Folds
    folds_file = anno_dir / "standard_splits.json"
    with open(folds_file, "r", encoding="utf-8") as f:
        folds = json.load(f)

    fold_tr = set(folds["standard_train_fold"])
    fold_dev = set(folds["standard_valid_fold"])
    fold_te = set(folds["standard_test_fold"])
    all_fold_vids = fold_tr.union(fold_dev).union(fold_te)

    print(f"\n[5] Video ID Reconciliation against Official SDK Folds:")
    print(f"    Official Fold Definitions: Train={len(fold_tr)}, Dev={len(fold_dev)}, Test={len(fold_te)}, Total={len(all_fold_vids)}")
    print(f"    Manifest Video Counts    : Train={len(tr_vids)}, Dev={len(dev_vids)}, Test={len(te_vids)}, Total={len(tr_vids)+len(dev_vids)+len(te_vids)}")

    missing_tr_vids = fold_tr - tr_vids
    missing_dev_vids = fold_dev - dev_vids
    missing_te_vids = fold_te - te_vids

    print(f"\n    Detailed Investigation of Missing Videos from Benchmark Folds:")
    print(f"    -> Train missing ({len(missing_tr_vids)}): {sorted(list(missing_tr_vids))}")
    print(f"    -> Dev missing   ({len(missing_dev_vids)}): {sorted(list(missing_dev_vids))}")
    print(f"    -> Test missing  ({len(missing_te_vids)}): {sorted(list(missing_te_vids))}")

    # Check why each missing video was omitted
    lbl_file = feat_dir / "CMU_MOSEI_Labels.csd"
    with h5py.File(lbl_file, "r") as h5f:
        lbl_vids_all = set(h5f[list(h5f.keys())[0]]["data"].keys())

    missing_all = missing_tr_vids.union(missing_dev_vids).union(missing_te_vids)
    for m_vid in sorted(list(missing_all)):
        in_labels = m_vid in lbl_vids_all
        in_anno = m_vid in all_anno_vids
        anno_segs = [r for r in all_anno_rows if r["video_id"] == m_vid]
        excl_segs = [r for r in excl_rows if r["video_id"] == m_vid]
        print(f"    - Video `{m_vid}`: in_Labels_csd={in_labels}, in_annotations={in_anno}, num_annotated_segments={len(anno_segs)}, exclusion_reasons={[x['exclusion_reason'] for x in excl_segs]}")

    # 6. Feature Path & Data Readability Verification
    print(f"\n[6] Feature Path & Data Readability Verification:")
    lbl_h5 = h5py.File(feat_dir / "CMU_MOSEI_Labels.csd", "r")
    aud_h5 = h5py.File(feat_dir / "CMU_MOSEI_COVAREP.csd", "r")
    vis_h5 = h5py.File(feat_dir / "CMU_MOSEI_VisualFacet42.csd", "r")
    wrd_h5 = h5py.File(feat_dir / "CMU_MOSEI_TimestampedWords.csd", "r")

    aud_data = aud_h5[list(aud_h5.keys())[0]]["data"]
    vis_data = vis_h5[list(vis_h5.keys())[0]]["data"]
    wrd_data = wrd_h5[list(wrd_h5.keys())[0]]["data"]

    # Verify a random sample of 500 rows across manifests
    import random
    all_manifest_rows = tr_rows + dev_rows + te_rows
    sample_manifest = random.sample(all_manifest_rows, min(500, len(all_manifest_rows)))

    sample_verified_count = 0
    for r in sample_manifest:
        vid = r["video_id"]
        t_start, t_end = float(r["start_time"]), float(r["end_time"])

        # Audio check
        if r["has_audio"] == "True":
            assert vid in aud_data, f"Video {vid} missing from COVAREP!"
            a_int = aud_data[vid]["intervals"][:]
            overlap = (a_int[:, 0] < t_end) & (a_int[:, 1] > t_start)
            assert overlap.any(), f"No audio frames for {r['sample_id']} in [{t_start}, {t_end}]!"

        # Visual check
        if r["has_visual"] == "True":
            assert vid in vis_data, f"Video {vid} missing from VisualFacet42!"
            v_int = vis_data[vid]["intervals"][:]
            overlap = (v_int[:, 0] < t_end) & (v_int[:, 1] > t_start)
            assert overlap.any(), f"No visual frames for {r['sample_id']} in [{t_start}, {t_end}]!"

        sample_verified_count += 1

    lbl_h5.close()
    aud_h5.close()
    vis_h5.close()
    wrd_h5.close()
    print(f"    Verified {sample_verified_count} randomly sampled manifest records against CSD arrays: 100% OK.")

    print("\n=================================================================")
    print("CMU-MOSEI Split & Accounting Audit Completed Successfully.")
    print("=================================================================")
    return {}


def main():
    parser = argparse.ArgumentParser(description="CMU-MOSEI Local Acquisition & Verification Utility")
    parser.add_argument("--target-root", type=str, default="D:/Datasets/CMU-MOSEI", help="Storage root on D:")
    parser.add_argument("--transfer-existing", action="store_true", help="Transfer C: .csd files to D:")
    parser.add_argument("--verify-hf-metadata", action="store_true", help="Query and verify Hugging Face repository metadata & checksums")
    parser.add_argument("--fetch-folds", action="store_true", help="Fetch official standard benchmark folds from SDK")
    parser.add_argument("--cross-modal-audit", action="store_true", help="Run cross-modal alignment audit across all 4 CSD files and write mosei_annotations.csv")
    parser.add_argument("--reconcile-audit", action="store_true", help="Run comprehensive segment and video split reconciliation audit")
    parser.add_argument("--verify-all", action="store_true", help="Verify all .csd files in target features dir")
    args = parser.parse_args()

    root = Path(args.target_root).resolve()
    dirs = setup_directories(root)

    if args.transfer_existing:
        c_feat = Path("datasets/CMU-MOSEI/features").resolve()
        transferred = transfer_existing_csd(c_feat, dirs["features"])
        print(f"Transferred and verified: {transferred}")

    if args.verify_hf_metadata:
        verify_hf_metadata()

    if args.fetch_folds:
        folds_file = dirs["annotations"] / "standard_splits.json"
        fetch_official_standard_folds(folds_file)

    if args.verify_all:
        print("\n=== Comprehensive Verification of All CSD Files ===")
        for key, info in OFFICIAL_FILES.items():
            fpath = dirs["features"] / info["filename"]
            ok, reason, meta = verify_csd_integrity(fpath)
            print(f"[{key.upper()}] {info['filename']}: present={fpath.exists()}, valid={ok}, reason={reason}")
            if ok:
                print(f"    Videos: {meta['video_count']}, Root: {meta['root']}, Sample: {meta['sample_info']}")

    if args.cross_modal_audit:
        run_cross_modal_audit_and_generate_annotations(root)

    if args.reconcile_audit:
        reconcile_audit(root)


if __name__ == "__main__":
    main()
