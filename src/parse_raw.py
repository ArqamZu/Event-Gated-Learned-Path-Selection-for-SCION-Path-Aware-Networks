"""Step 2 - Data collection: parse the public ScionPathML raw JSON archives into tables.

Reads the three per-lab raw_data.zip archives (copied from github.com/keshvadi/ScionPathML,
AnalysisResults/LabN/Data/raw_data.zip) directly, without extracting, and writes:

  data/processed/pings.parquet     one row per (probe file, path): SCMP ping statistics
  data/processed/pathsets.parquet  one row per (path-discovery snapshot, path)
  data/processed/bandwidth.parquet one row per (bandwidth test, path, direction)

Scope (Phase 2, Step 2): clean window from 2025-07-16 00:00 UTC onward; the Taiwan AS
(22-ffaa:1:11ee) is dropped because it was unreachable for the whole campaign.
"""
import hashlib
import json
import platform
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from config import (DATASET_BLOBS, DATASET_COMMIT, DATASET_REPO, DATASET_ZIP_COMMIT, PARSER_NAME,
                    PARSER_VERSION)

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"

CLEAN_START = pd.Timestamp("2025-07-16 00:00")
EXCLUDED_AS = {"22-ffaa:1:11ee"}
LAB_SOURCE = {"Lab1": "19-ffaa:1:11de", "Lab2": "17-ffaa:1:11e4", "Lab3": "18-ffaa:1:11e5"}
VANTAGE_NAME = {"19-ffaa:1:11de": "EU", "17-ffaa:1:11e4": "CH", "18-ffaa:1:11e5": "NA"}

TS_RE = re.compile(r"(\d{4}-\d{2}-\d{2}T\d{2}[:-]\d{2})")


def file_ts(name: str) -> pd.Timestamp:
    s = TS_RE.search(name).group(1)
    return pd.Timestamp(s[:13] + ":" + s[14:16])


def slot_of(ts: pd.Timestamp) -> pd.Timestamp:
    return ts.floor("30min")


def seq_ases(sequence: str):
    """'19-ffaa:1:11de#0,1 19-ffaa:0:1303#280,1 ...' -> list of ISD-AS strings."""
    if " -> " in sequence:
        return [h.strip() for h in sequence.split("->")]
    return [h.split("#")[0] for h in sequence.split()]


def path_structure(sequence: str):
    ases = seq_ases(sequence)
    return len(ases), len({a.split("-")[0] for a in ases})


def hop_count(probe: dict, n_as: int):
    """Inter-AS hops of the path. The ping result lists each hop as an (ingress, egress) interface
    pair, so hops = len(interfaces) / 2. Probes without a ping result (skipped) fall back to the
    AS sequence (n_as - 1), which is what the interface list gives on every probed path."""
    hops = ((probe.get("ping_result") or {}).get("path") or {}).get("hops")
    return len(hops) // 2 if hops else n_as - 1


def parse_pings(z, lab, src):
    rows = []
    for name in z.namelist():
        base = name.rsplit("/", 1)[-1]
        if not base.startswith("prober_") or not base.endswith(".json"):
            continue
        ts = file_ts(base)
        d = json.loads(z.read(name))
        dst = d.get("ia")
        for p in d.get("probes", []):
            seq = p.get("sequence", "")
            n_as, n_isd = path_structure(seq)
            r = dict(lab=lab, src=src, dst=dst, file_ts=ts, slot=slot_of(ts),
                     fp=p.get("fingerprint", "")[:16], sequence=seq, n_as=n_as, n_isd=n_isd,
                     hop_count=hop_count(p, n_as),
                     n_hop_ifaces=len(((p.get("ping_result") or {}).get("path") or {}).get("hops") or []))
            if "ping_result" in p:
                st = p["ping_result"].get("statistics") or {}
                replies = p["ping_result"].get("replies") or []
                rtts = [x["round_trip_time"] for x in replies if x.get("state") == "success"]
                r.update(status="ok", sent=st.get("sent"), received=st.get("received"),
                         loss=st.get("packet_loss"),
                         rtt_avg=st.get("avg_rtt") if rtts else None,
                         rtt_min=st.get("min_rtt") if rtts else None,
                         rtt_max=st.get("max_rtt") if rtts else None,
                         rtt_mdev=st.get("mdev_rtt") if rtts else None,
                         rtt_med=float(pd.Series(rtts).median()) if rtts else None)
            elif p.get("status") == "skipped":
                r.update(status="skipped")
            else:
                r.update(status="error", note=str(p.get("error", ""))[:200])
            rows.append(r)
    return rows


def parse_pathsets(z, lab, src):
    rows = []
    for name in z.namelist():
        base = name.rsplit("/", 1)[-1]
        if not base.startswith("AS-") or not base.endswith(".json"):
            continue
        ts = file_ts(base)
        d = json.loads(z.read(name))
        dst = d.get("destination")
        paths = d.get("paths")
        if not paths:  # empty discovery result: record as an empty snapshot
            rows.append(dict(lab=lab, src=src, dst=dst, file_ts=ts, slot=slot_of(ts),
                             fp=None, sequence=None, status="none", n_as=None, n_isd=None))
            continue
        for p in paths:
            seq = p.get("sequence", "")
            n_as, n_isd = path_structure(seq)
            rows.append(dict(lab=lab, src=src, dst=dst, file_ts=ts, slot=slot_of(ts),
                             fp=p.get("fingerprint", "")[:16], sequence=seq,
                             status=p.get("status"), n_as=n_as, n_isd=n_isd, mtu=p.get("mtu")))
    return rows


def _num(s, unit_re):
    m = re.search(unit_re, s or "")
    return float(m.group(1)) if m else None


def parse_bandwidth(z, lab, src):
    rows = []
    for name in z.namelist():
        base = name.rsplit("/", 1)[-1]
        if not base.startswith("BW_") or not base.endswith(".json"):
            continue
        ts = file_ts(base)
        d = json.loads(z.read(name))
        for p in d.get("paths", []):
            res = p.get("result") or {}
            for direction in ("S->C results", "C->S results"):
                r = res.get(direction)
                rows.append(dict(
                    lab=lab, src=src, server=d.get("as"), file_ts=ts, slot=slot_of(ts),
                    tier=d.get("target_mbps"), fp=(p.get("fingerprint") or "")[:16],
                    sequence=p.get("sequence"), direction=direction[:4],
                    ok=bool(r) and p.get("return_code") == 0 and not res.get("invalid_format"),
                    achieved_mbps=_num(r.get("achieved_bps") if r else None, r"/\s*([\d.]+)\s*Mbps"),
                    loss=_num(r.get("loss_rate") if r else None, r"([\d.]+)%")))
    return rows


def git_blob_sha1(path: Path) -> str:
    b = path.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()


def provenance(counts):
    """Step 2 reproducibility record: dataset commit, archive hashes, parser version, row counts."""
    files = {}
    for lab in LAB_SOURCE:
        f = RAW / f"{lab}.zip"
        blob = git_blob_sha1(f)
        files[lab] = dict(repo_path=f"AnalysisResults/{lab}/Data/raw_data.zip", bytes=f.stat().st_size,
                          sha256=hashlib.sha256(f.read_bytes()).hexdigest(), git_blob_sha1=blob,
                          matches_repo_commit=blob == DATASET_BLOBS[lab])
    rec = dict(dataset_repo=DATASET_REPO, dataset_commit=DATASET_COMMIT, archives_added_in_commit=DATASET_ZIP_COMMIT,
               archives=files, all_archives_match=all(v["matches_repo_commit"] for v in files.values()),
               parser=PARSER_NAME, parser_version=PARSER_VERSION,
               parser_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               python=platform.python_version(), pandas=pd.__version__,
               clean_start=str(CLEAN_START), excluded_as=sorted(EXCLUDED_AS), row_counts=counts,
               parsed_at_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    (OUT / "provenance.json").write_text(json.dumps(rec, indent=2))
    if not rec["all_archives_match"]:
        print("WARNING: local archives differ from the recorded dataset commit")
    return rec


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pings, sets, bws = [], [], []
    counts = {}
    for lab, src in LAB_SOURCE.items():
        with zipfile.ZipFile(RAW / f"{lab}.zip") as z:
            pings += parse_pings(z, lab, src)
            sets += parse_pathsets(z, lab, src)
            bws += parse_bandwidth(z, lab, src)
        print(f"{lab}: parsed")
    for rows, fname, dcol in ((pings, "pings", "dst"), (sets, "pathsets", "dst"), (bws, "bandwidth", None)):
        df = pd.DataFrame(rows)
        n0 = len(df)
        df = df[df.file_ts >= CLEAN_START]
        if dcol:
            df = df[~df[dcol].isin(EXCLUDED_AS)]
        df = df.sort_values([c for c in ("src", "dst", "server", "slot", "fp") if c in df]).reset_index(drop=True)
        df.to_parquet(OUT / f"{fname}.parquet", index=False)
        counts[fname] = dict(raw_rows=n0, clean_rows=len(df))
        print(f"{fname}: {n0} raw rows -> {len(df)} in clean scope")
    rec = provenance(counts)
    print(f"dataset commit {rec['dataset_commit'][:12]}, archives match: {rec['all_archives_match']}, "
          f"parser {rec['parser_version']}")


if __name__ == "__main__":
    main()
