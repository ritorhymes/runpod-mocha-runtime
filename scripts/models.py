"""Official BF16 MoCha model manifest, downloader, and offline verifier.

No persistent RunPod resources; downloads go to the disposable Pod's local disk.
"""
from __future__ import annotations
import argparse
import json
import os
import shutil
from pathlib import Path

MANIFEST = Path(__file__).resolve().parents[1] / "model-manifest.json"
DEFAULT_HOME = Path(os.getenv("MOCHA_HOME", "/workspace/mocha"))


def get_manifest(path: Path = MANIFEST) -> dict:
    info = json.loads(path.read_text(encoding="utf-8"))
    assert info["officialPrecision"] == "bfloat16"
    assert info["officialImageSize"] == [832, 480]
    assert info["officialFrames"] == 81
    return info


def repo_dir(root: Path, repo: str) -> Path:
    if repo == "Wan-AI/Wan2.1-T2V-14B":
        return root / "weights" / "wan"
    if repo == "Orange-3DV-Team/MoCha":
        return root / "weights" / "mocha"
    raise ValueError(f"Unapproved model repository: {repo}")


def expected_files(info: dict, root: Path):
    for model in info["hfRepositories"]:
        for f in model["files"]:
            yield repo_dir(root, model["repo"]) / f["path"], f["size"], model


def verify(info: dict, root: Path, *, quiet: bool = False) -> bool:
    ok = True
    for filename, expected_size, _ in expected_files(info, root):
        actual = filename.stat().st_size if filename.is_file() else None
        matched = actual == expected_size
        ok &= matched
        if not quiet:
            print(("OK " if matched else "MISSING/MISMATCH ") + str(filename) +
                  f" ({actual if actual is not None else 'none'} / {expected_size} bytes)")
    return bool(ok)


def download(info: dict, root: Path):
    from huggingface_hub import snapshot_download

    root.mkdir(parents=True, exist_ok=True)
    expected = list(expected_files(info, root))
    absent = sum(sz for path, sz, _ in expected if not path.is_file() or path.stat().st_size != sz)
    remaining_disk = shutil.disk_usage(root).free
    if remaining_disk < absent + 20 * 1024 ** 3:
        raise SystemExit(
            f"Not enough ephemeral disk: require at least {(absent + 20*1024**3)/1e9:.1f} GB "
            f"free, have {remaining_disk/1e9:.1f} GB. Increase --container-disk-in-gb."
        )
    for model in info["hfRepositories"]:
        directory = repo_dir(root, model["repo"])
        directory.mkdir(parents=True, exist_ok=True)
        print(f"Downloading official {model['repo']} at {model['revision']}", flush=True)
        snapshot_download(
            repo_id=model["repo"], revision=model["revision"],
            allow_patterns=[f["path"] for f in model["files"]],
            local_dir=str(directory), max_workers=8
        )
    if not verify(info, root):
        raise SystemExit("CHECKPOINT VERIFICATION FAILED: not safe to start inference")


def main():
    parser = argparse.ArgumentParser(description="Official MoCha/Wan BF16 checkpoint manager")
    parser.add_argument("action", choices=("plan", "download", "verify"))
    parser.add_argument("--root", type=Path, default=DEFAULT_HOME)
    args = parser.parse_args()
    info = get_manifest()
    root = args.root.expanduser().resolve()
    if args.action == "plan":
        total = sum(s for _, s, _ in expected_files(info, root))
        print(f"Precision: {info['officialPrecision']} (unquantized official checkpoints)")
        print(f"Required download: {total:,} bytes ({total/1e9:.1f} GB)")
        print(f"Ephemeral destination: {root} (deleted with Pod)")
        for model in info["hfRepositories"]:
            print(f"  {model['repo']} commit {model['revision']}")
    elif args.action == "download":
        download(info, root)
    else:
        raise SystemExit(0 if verify(info, root) else 1)


if __name__ == "__main__":
    main()
