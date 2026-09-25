"""Fetch LocalLLaMA/typed-decisions parquet shards and convert to local JSONL.

Avoids the `datasets` library (pandas DLL is blocked by local app-control policy)
and the datasets-server REST API (rate-limited). Parquet rows are decoded with
the stdlib-free `pyarrow` only if present, else `fastparquet`; no pandas.
"""
import json
import os
import sys

from huggingface_hub import list_repo_files, hf_hub_download

REPO = "LocalLLaMA/typed-decisions"
SPLIT = sys.argv[1] if len(sys.argv) > 1 else "train"
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else 0  # 0 = all rows in the split
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "data_local", "typed_decisions_%s.jsonl" % SPLIT)


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    files = [f for f in list_repo_files(REPO, repo_type="dataset")
             if f.startswith("all/%s" % SPLIT) and f.endswith(".parquet")]
    if not files:
        sys.exit("no parquet files under all/%s in %s" % (SPLIT, REPO))
    print("shards:", files)

    try:
        import pyarrow.parquet as pq
    except ImportError:
        sys.exit("pip install pyarrow")

    n = 0
    with open(OUT, "w", encoding="utf-8") as out:
        for fname in files:
            local = hf_hub_download(REPO, fname, repo_type="dataset")
            tbl = pq.read_table(local, columns=["state", "questions", "gold"])
            cols = {c: tbl.column(c).to_pylist() for c in ("state", "questions", "gold")}
            for i in range(tbl.num_rows):
                out.write(json.dumps({"state": cols["state"][i],
                                      "questions": cols["questions"][i],
                                      "gold": cols["gold"][i]}, ensure_ascii=False) + "\n")
                n += 1
                if LIMIT and n >= LIMIT:
                    break
            if LIMIT and n >= LIMIT:
                break
    print("wrote %d rows -> %s" % (n, OUT))


if __name__ == "__main__":
    main()
