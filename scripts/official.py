"""Fetch pinned official MoCha source and run its unquantized model.

The upstream inference script has literal /path/to/... placeholders. We derive
a path-only copy, verifying the exact known placeholders first. There is no
change to model architecture, precision, 50 steps, 81 frames, or output fps.
"""
from __future__ import annotations
import argparse
import csv
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from models import DEFAULT_HOME, get_manifest, repo_dir, verify

REQUIRED_COLUMNS = ("source_video", "source_mask", "reference_1", "reference_2")


def call(*args: str, cwd: Path | None = None):
    print("$", " ".join(args), flush=True)
    subprocess.run(args, cwd=cwd, check=True)


def fetch(root: Path, manifest: dict):
    target = root / "upstream"
    expected = manifest["officialGitRevision"]
    if (target / ".git").exists():
        actual = subprocess.check_output(["git", "-C", str(target), "rev-parse", "HEAD"],
                                         text=True).strip()
        if actual != expected:
            raise SystemExit(f"Upstream exists at wrong commit ({actual}); refusing overwrite")
        return target
    if target.exists() and any(target.iterdir()):
        raise SystemExit(f"Target is nonempty, refusing to overwrite: {target}")
    target.mkdir(parents=True, exist_ok=True)
    call("git", "init", "-q", str(target))
    call("git", "-C", str(target), "remote", "add", "origin", manifest["officialGitRepo"])
    call("git", "-C", str(target), "sparse-checkout", "init", "--cone")
    call("git", "-C", str(target), "sparse-checkout", "set", "diffsynth", "inference_mocha.py", "requirements.txt")
    call("git", "-C", str(target), "fetch", "--depth", "1", "origin", expected)
    call("git", "-C", str(target), "checkout", "-q", "--detach", "FETCH_HEAD")
    actual = subprocess.check_output(["git", "-C", str(target), "rev-parse", "HEAD"],
                                     text=True).strip()
    if actual != expected:
        raise SystemExit(f"Unexpected source SHA {actual}, expected {expected}")
    return target


def official_paths(manifest: dict, root: Path):
    wan = repo_dir(root, "Wan-AI/Wan2.1-T2V-14B")
    return {
        "/path/to/diffusion_pytorch_model-": str(wan / "diffusion_pytorch_model-"),
        "/path/to/Wan2.1-T2V-14B/": str(wan) + "/",
    }


def path_only_patch(source: str, manifest: dict, root: Path) -> str:
    replacements = official_paths(manifest, root)
    expected_counts = {"/path/to/diffusion_pytorch_model-": 6,
                       "/path/to/Wan2.1-T2V-14B/": 2}
    for original, replacement in replacements.items():
        count = source.count(original)
        if count != expected_counts[original]:
            raise ValueError(f"Official inference changed: placeholder {original} count={count}, "
                             f"expected={expected_counts[original]}. Stop and inspect upstream.")
        source = source.replace(original, replacement)
    if "/path/to/" in source:
        raise ValueError("Unresolved placeholder in official inference source")
    return source


def validate_csv(csv_path: Path):
    if not csv_path.is_file():
        raise SystemExit(f"Missing job CSV: {csv_path}")
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or any(f not in reader.fieldnames for f in REQUIRED_COLUMNS):
            raise SystemExit(f"CSV must include {', '.join(REQUIRED_COLUMNS)}")
        rows = list(reader)
    if len(rows) != 1:
        raise SystemExit("One scene per official-inference job is required")
    row = rows[0]
    for col in REQUIRED_COLUMNS:
        if col == "reference_2" and (not row[col] or row[col] == "None"):
            continue
        f = Path(row[col])
        if not f.is_file():
            raise SystemExit(f"Missing {col}: {f}")
    return row


def render(root: Path, csv_path: Path, output: Path):
    info = get_manifest()
    source_dir = fetch(root, info)
    if not verify(info, root, quiet=True):
        raise SystemExit("Official checkpoints missing/invalid. Run mocha-runtime download first.")
    validate_csv(csv_path)
    upstream = (source_dir / "inference_mocha.py").read_text(encoding="utf-8")
    script = path_only_patch(upstream, info, root)
    checkpoint = repo_dir(root, "Orange-3DV-Team/MoCha") / "preview/step18500.ckpt"
    output.mkdir(parents=True, exist_ok=True)
    temp_dir = root / "generated"
    temp_dir.mkdir(exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".py",
                                     prefix="inference-mocha-", dir=temp_dir, delete=False) as tmp:
        tmp.write(script)
        script_path = Path(tmp.name)
    try:
        env = dict(os.environ)
        env["PYTHONPATH"] = str(source_dir) + os.pathsep + env.get("PYTHONPATH", "")
        command = [
            sys.executable, str(script_path), "--data_path", str(csv_path),
            "--ckpt_path", str(checkpoint), "--output_dir", str(output),
            "--dataloader_num_workers", "0"
        ]
        print("OFFICIAL baseline: BF16, 832x480, 81 frames, 50 steps, 30 fps", flush=True)
        subprocess.run(command, check=True, cwd=source_dir, env=env)
    finally:
        script_path.unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser(description="Pinned official MoCha BF16 inference wrapper")
    ap.add_argument("action", choices=("source", "run", "patch-check"))
    ap.add_argument("--root", type=Path, default=DEFAULT_HOME)
    ap.add_argument("--csv", type=Path)
    ap.add_argument("--output", type=Path)
    ns = ap.parse_args()
    root = ns.root.expanduser().resolve()
    manifest = get_manifest()
    if ns.action == "source":
        print("Pinned official source:", fetch(root, manifest))
    elif ns.action == "patch-check":
        src = fetch(root, manifest) / "inference_mocha.py"
        path_only_patch(src.read_text(encoding="utf-8"), manifest, root)
        print("PASS: Official model path placeholders replace safely.")
    else:
        if ns.csv is None or ns.output is None:
            ap.error("run requires --csv and --output")
        render(root, ns.csv.expanduser().resolve(), ns.output.expanduser().resolve())


if __name__ == "__main__":
    main()
