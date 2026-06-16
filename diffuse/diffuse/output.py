"""Output helpers — resolve paths, save metadata, debrief, and open images."""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from diffuse.paths import OUTPUTS_DIR

log = logging.getLogger("diffuse")


def resolve_output_path(model_name: str, seed: int, output_arg: str | None, cwd: Path | None = None) -> Path:
    """Determine output PNG path."""
    if output_arg is not None:
        return Path(output_arg).expanduser().resolve()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Sanitize model name for filesystem (replace : and other problematic chars)
    safe_model = model_name.replace(":", "_").replace("/", "_")
    base_dir = cwd if cwd else Path.cwd()
    out = base_dir / f"diffuse_{safe_model}_{ts}_seed{seed}.png"
    out.parent.mkdir(parents=True, exist_ok=True)

    # Also write to internal log dir for metadata
    internal_dir = OUTPUTS_DIR / model_name
    internal_dir.mkdir(parents=True, exist_ok=True)

    return out


def save_metadata(
    model_name: str, prompt: str, seed: int, width: int, height: int,
    steps: int, load_time: float, diffusion_time: float, wall_time: float,
    peak_hbm: float, output_path: Path, enhanced_prompt: str | None = None,
) -> Path:
    """Append generation record to JSON log."""
    meta_dir = OUTPUTS_DIR / model_name
    meta_dir.mkdir(parents=True, exist_ok=True)
    meta_path = meta_dir / "generations.json"

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": model_name,
        "prompt": prompt,
        "seed": seed,
        "width": width,
        "height": height,
        "steps": steps,
        "load_seconds": round(load_time, 3),
        "diffusion_seconds": round(diffusion_time, 3),
        "wall_seconds": round(wall_time, 3),
        "peak_hbm_mib": round(peak_hbm, 1),
        "output": str(output_path),
    }
    if enhanced_prompt:
        record["enhanced_prompt"] = enhanced_prompt
        record["prompt_enhanced"] = True

    existing = []
    if meta_path.exists():
        try:
            existing = json.loads(meta_path.read_text())
            if not isinstance(existing, list):
                existing = []
        except json.JSONDecodeError:
            existing = []

    existing.append(record)
    meta_path.write_text(json.dumps(existing, indent=2) + "\n")
    return meta_path


def print_debrief(
    model_name: str, model_info: dict, prompt: str, seed: int,
    width: int, height: int, steps: int, load_time: float,
    diffusion_time: float, wall_time: float, peak_hbm: float,
    output_path: Path, enhanced_prompt: str | None = None,
    original_prompt: str | None = None,
) -> None:
    """Print generation stats report."""
    print()
    print("═══ diffuse — Generation Report ═══")
    print(f"  Model:       {model_name} ({model_info['bits']})")
    if original_prompt and original_prompt != prompt:
        print(f"  Prompt:      \"{original_prompt}\"")
        print(f"  Enhanced:    Yes → Ideogram 4 JSON")
    else:
        print(f"  Prompt:      \"{prompt}\"")
    print(f"  Seed:        {seed}")
    print(f"  Resolution:  {width} × {height}")
    print(f"  Steps:       {steps}")
    if original_prompt and original_prompt != prompt and enhanced_prompt:
        print()
        print("  Enhanced prompt (JSON):")
        try:
            parsed = json.loads(enhanced_prompt)
            for key, val in parsed.items():
                if isinstance(val, dict):
                    print(f"    {key}:")
                    for k, v in val.items():
                        if isinstance(v, list):
                            print(f"      {k}:")
                            for item in v:
                                print(f"        {item}")
                        else:
                            print(f"      {k}: {v}")
                else:
                    print(f"    {key}: {val}")
        except json.JSONDecodeError:
            for line in enhanced_prompt.split("\n"):
                print(f"    {line[:120]}")
    print()
    print("  Timings:")
    print(f"    Setup:      {load_time:7.2f} s   (imports + model load)")
    print(f"    Diffusion: {diffusion_time:7.2f} s   (denoising + VAE decode)")
    print(f"    ─────────────────────")
    print(f"    Wall:       {wall_time:7.2f} s")
    print()
    if peak_hbm > 0:
        print("  Memory:")
        print(f"    Peak HBM:  {peak_hbm:,.0f} MiB")
        print()
    print(f"  Output: {output_path}")
    print("══════════════════════════════════════")


def get_previous_runs(model_name: str, width: int, height: int) -> list[float]:
    """Check historical generation times at this resolution."""
    meta_path = OUTPUTS_DIR / model_name / "generations.json"
    if not meta_path.exists():
        return []
    try:
        entries = json.loads(meta_path.read_text())
    except (json.JSONDecodeError, OSError):
        return []
    return [
        float(e["wall_seconds"])
        for e in entries
        if isinstance(e, dict)
        and e.get("width") == width
        and e.get("height") == height
        and isinstance(e.get("wall_seconds"), (int, float))
    ]


def open_image(path: Path) -> None:
    """Open the generated image with feh in a background process."""
    viewer = shutil.which("feh")
    if not viewer:
        log.warning("feh not found — skipping image viewer")
        return

    subprocess.Popen(
        [viewer, "--scale-down", str(path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    log.info("Opened in feh: %s", path)