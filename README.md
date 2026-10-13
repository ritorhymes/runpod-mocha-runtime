# Official MoCha BF16 on disposable RunPod GPUs

Reusable, **code-only** GPU runtime for [Orange-3DV-Team/MoCha](https://github.com/Orange-3DV-Team/MoCha) using **official full-precision BF16 checkpoints**, without a quantized community conversion.

This project is a companion to the private/local controller at ~/techstuff/runpod (NotCodex). **No user footage, character images, RunPod credentials, or model weights are committed to GitHub or included in the container.** At Pod startup, the tools fetch the pinned official upstream source and model checkpoints directly from GitHub and Hugging Face.

## Verified upstream inputs

- Official MoCha source: see model-manifest.json for **exact commit**
- Official MoCha checkpoint: Orange-3DV-Team/MoCha, preview/step18500.ckpt
- Official Wan 2.1 T2V 14B: 6 BF16 DiT shards, official T5 encoder, VAE
- Total model bytes: **97,600,801,337 (~97.6 decimal GB)**
- Official baseline: **832x480**, 81 frames, 50 steps, BF16, 30 fps.
- No changes to the official architecture, denoising, checkpoints, or frame settings.
- The upstream inference script literally contains /path/to/ placeholders. Our verified path-only wrapper substitutes those paths in a temporary copy at runtime.

The upstream MoCha source is **not bundled into the published image**; only original orchestration scripts and ordinary runtime dependencies are included. Consult upstream repository and checkpoint licenses before redistributing either.

## GitHub build (no RunPod GPU)

Workflow: .github/workflows/build.yml (manual workflow_dispatch only, no automatic push builds).

On public repositories, standard GitHub-hosted Actions runners and public GHCR packages are free under GitHub's published terms. The first build can fail if the large CUDA base image consumes the hosted runner's disk; the workflow removes unused runner software first. No paid runner is configured.

The workflow:
1. Runs local unit tests.
2. Checks out the pinned MoCha source sparsely and checks that all official path substitutions match.
3. Builds the Docker image from RunPod's official PyTorch 2.2.0 / Python 3.10 / CUDA 12.1 base.
4. Installs exact compatible research dependencies and runs import smoke tests plus pip check.
5. Publishes to ghcr.io/ritorhymes/runpod-mocha-runtime:main and an immutable commit tag.

No GPU inference is claimed until separately tested on an authorized paid Pod.

## Container commands (when running on a Pod)

    mocha-runtime help
    mocha-runtime plan
    mocha-runtime smoke
    mocha-runtime source
    mocha-runtime patch-check
    mocha-runtime download
    mocha-runtime verify
    mocha-runtime run --csv /workspace/job/shot.csv --output /workspace/results

The download step verifies checkpoint sizes; it does NOT quantize or alter them. Expect ~97.6 GB downloads into the Pod's local ephemeral disk, plus extra working space. **A container disk of at least 180 GB is recommended for the first trial**. The image contains dependencies only; no model download during image build.

The CSV has the official columns:
source_video,source_mask,reference_1,reference_2
Use absolute remote paths. Each job should be a single shot, ideally exactly **81 video frames at 30 fps**, with the first-frame mask and two reference images (second image may be "None"). Because the upstream output is 30 fps, audio and original timing must be handled separately by FFmpeg after generation.

## Interfaces

- **CLI**: runpodctl to create/delete the Pod and SSH into it; mocha-runtime to fetch official source, verify weights, and run.
- **Browser UI**: the official RunPod PyTorch image includes **JupyterLab**. Use RunPod's Pod Connect URL (protected access) to view scripts, media, logs and notebooks, or connect through an SSH tunnel.
- **Local UI**: the companion rgpu controller provides a loopback-only browser gallery for reviewing results, without needing the remote GPU.

This official research repo does **not** provide a MoCha-specific visual node editor out of the box. We are intentionally not substituting a ComfyUI community implementation for the official BF16 reference.

## Billing safety

This repository will **never** create a paid Pod from a GitHub Action. There is no GPU creation in the workflow and no scheduled trigger.

The authenticated local rgpu controller has rgpu status and rgpu kill. When a user explicitly authorizes a paid launch, use an A100 80GB-class GPU for the first reference run, test with a small shot, retrieve output before deleting the Pod, then check RunPod account spend.

No active external compute remains after deleting the disposable Pod. Fresh pods must download their models again, and initial cold start can take significant time.

## Known limitations / gates

- Software image build and CPU import checks do NOT prove that a 14B BF16 model fits VRAM or produces good replacement footage.
- One first-frame mask may still allow identity bleed in shots containing multiple people.
- Native 720p is *not* the official baseline; don't silently change size to 1280x720.
- The official loader requires **BF16-compatible GPU**, adequate CPU RAM and approximately 100 GB of local weight files, before GPU inference can be tested.
- The official pipeline is initially tested on short shots only; extended video and high-quality target masks remain separate steps.

## Local offline checks

    python3 -m unittest discover -s tests -v
    python3 scripts/models.py plan
    python3 scripts/official.py patch-check --root /tmp/mocha-smoke-source
