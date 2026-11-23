"""diffuse sheet — compiled character reference sheets on the local H3-Image.

Standalone port of the H3 Character Sheet Designer compiler (ComfyUI custom
node, MIT, ukr8b3g-cmyk). The command compiles a structured prompt plus an
optional layout guide image, then hands everything to the h3img early path
with ref_mode "I" (exact compiler canvas; no ref-derived sizing).

Local changes vs upstream:
  - merged part roles on one file (--part hands,feet:img.png), deduped by path
  - guide assets live in assets/ (package-relative)
  - T2I mode (--desc): no character reference; Picture tags renumbered and
    the character citation replaced by the description
  - reference order for WanGP conditioning: base -> guide -> extras, matching
    the compiled <Picture N> numbering exactly
"""
from __future__ import annotations

import argparse
import json
import os
import secrets
import sys
from pathlib import Path

from diffuse.sheet.minimax_image_compiler import compile_minimax_image_state
from diffuse.sheet.guide_13 import render_reference13_guide
from diffuse.sheet.compiler import StateValidationError
from diffuse.sheet.state_13 import SETTINGS as SETTING_CHOICES
from diffuse.sheet.view_catalog_13 import VIEW_IDS

PRESET_ALIASES = {
    "standard5": "five", "five": "five",
    "basic4": "basic", "basic": "basic",
    "all13": "detail", "detail": "detail",
    "head6": "head", "head": "head",
    "turnaround": "turnaround",
    "single": "single",
    "outfit": "clothing", "clothing": "clothing",
}

PROFILE_CHOICES = ("current", "high", "max_8mp", "max_16mp")
PART_ROLE_CHOICES = ("same_character", "face", "head", "clothing",
                     "clothing_front", "clothing_back", "hands", "footwear")
# friendly spellings accepted by --part
PART_ROLE_ALIASES = {"feet": "footwear", "hand": "hands"}
CLOTHING_VIEW_IDS = ("clothing_front", "clothing_back")

# Upstream canvases land around 4-6.2 MP. Our pipeline shows regular vertical
# banding at >= 8 MP (decode-level, not fixable by steps): keep the work window
# near 4 MP and upscale with Real-ESRGAN when a bigger sheet is needed.
BANDING_LIMIT_MP = 8.0

VIEW_NAMES = {
    "face_front": "face, front", "face_left": "face, profile (image-right)",
    "face_right": "face, profile (image-left)", "face_3q_left": "face, 3/4 (image-right)",
    "face_3q_right": "face, 3/4 (image-left)", "face_back": "face, from behind",
    "body_front": "full body, front", "body_left": "full body, profile",
    "body_back": "full body, back", "clothing_front": "outfit, front (flat)",
    "clothing_back": "outfit, back (flat)", "hands": "hands close-up", "feet": "feet close-up",
}

EPILOG = f"""\
examples:
  diffuse sheet --desc "a horned wolf warrior, blue robe, gray fur"
  diffuse sheet --ref lobo.png --style anime
  diffuse sheet --ref lobo.png --ref rosto_alternativo.png --views all13
  diffuse sheet --ref lobo.png --part hands,feet:details.png --layout on
  diffuse sheet --ref body.png --part face:face.png --no-clothing --style photo

--part roles (role of a supplementary reference; repeatable):
  same_character     general supplement of the character (default)
  face / head        facial appearance / head + hair
  clothing           the outfit (front and back)
  clothing_front / clothing_back
  hands / footwear
  The first --ref is the character base (Picture 1). Remaining --ref files
  are general supplements. A --part file used as the base attaches its roles
  to Picture 1 instead of taking a new slot. The same file may carry several
  roles; duplicate paths are merged; max 3 supplementary files.

--views values (preset or CSV of view ids):
  standard5 (default) face front + profile, body front, profile, back
  basic4    face front, body front, profile, back
  head6     six head angles
  all13     everything: 6 head + 3 body + outfit front/back + hands + feet
  turnaround, single, outfit
  view ids: {', '.join(f'{k} ({v})' for k, v in VIEW_NAMES.items())}
"""


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="diffuse sheet",
        description="Compiled character reference sheet (preset + layout guide) "
                    "generated with the local H3-Image. The command writes the "
                    "structured prompt for you and runs the whole render.",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    inp = p.add_argument_group("character input")
    inp.add_argument("--desc", metavar="TEXT",
                     help="short character description (T2I from scratch). Required without --ref.")
    inp.add_argument("--ref", metavar="FILE", nargs="+", default=[],
                     help="references: the first is the character base (Picture 1); "
                          "the rest become general supplements of the same character")
    inp.add_argument("--part", metavar="ROLE[,ROLE]:FILE", action="append", default=[],
                     help="supplementary reference with an explicit role; repeatable; "
                          "one file can carry multiple roles (dedup by path, max 3 files)")

    sh = p.add_argument_group("sheet composition")
    sh.add_argument("--views", metavar="PRESET|CSV", default="standard5",
                    help="preset (standard5, basic4, head6, all13, turnaround, single, outfit) "
                         "or CSV of view ids (default: standard5)")
    sh.add_argument("--lower", choices=SETTING_CHOICES["clothing_type"], default="reference",
                    help="lower garment when not taken from the reference (default: reference)")
    sh.add_argument("--no-clothing", action="store_true",
                    help="No clothing · natural body surface: unclothed body keeps reference "
                         "fur/colors/markings (outfit views are dropped)")
    sh.add_argument("--hands", choices=SETTING_CHOICES["hand_state"], default="reference",
                    help="reference | bare | gloved (default: reference)")
    sh.add_argument("--feet", choices=SETTING_CHOICES["footwear_state"], default="reference",
                    help="reference | shod | unshod (default: reference)")
    sh.add_argument("--layout", choices=("on", "off"), default="off",
                    help="layout guide: white canvas with mannequin panels sent as a "
                         "Picture that supplies arrangement and viewing directions "
                         "(default: off; requires --ref)")
    sh.add_argument("--style", default="none", metavar="NAME",
                    help="rendering style: none (match the reference), anime, photo, "
                         "realistic_painting, semi_realistic_anime, oil_painting, "
                         "watercolor, gouache, colored_pencil, 3d")
    sh.add_argument("--profile", choices=PROFILE_CHOICES, default="current",
                    help="compiler resolution profile (default: current, the upstream GPU-"
                         "tested canvas). 'high' scales up (~6 MP); >= 8 MP hits the house "
                         "vertical-banding zone — prefer generating ~4-6 MP and upscaling "
                         "with Real-ESRGAN")
    sh.add_argument("--sheet-size", metavar="WxH",
                    help="manual canvas (overrides compiler sizing)")
    sh.add_argument("--view-prompt", metavar="VIEW:TEXT", action="append", default=[],
                    help="override the compiled text of one view; repeatable (up to 2000 "
                         "UTF-16 units each)")

    eng = p.add_argument_group("engine (h3img early path)")
    eng.add_argument("-m", "--model", default="h3-image")
    eng.add_argument("--seed", type=int, default=None)
    eng.add_argument("--steps", type=int, default=None,
                     help="default: 6 (official H3-Image schedule; --turbo remaps to 2)")
    eng.add_argument("--turbo", action="store_true", help="pDMD 2-step @1.0")
    eng.add_argument("--no-lora-select", action="store_true",
                     help="raw base: no selector, no pDMD")
    eng.add_argument("--lora-selector", default=None, metavar="MODE",
                     help="LoRA selector, opt-in (only wired mode: decision)")
    eng.add_argument("--nsfw", action="store_true")
    eng.add_argument("--enhance", action="store_true",
                     help="NOT recommended with sheet: rewrites the compiled prompt")
    eng.add_argument("--dry-run", action="store_true",
                     help="compile, save the guide/prompt sidecars and print everything, "
                          "but skip the render")
    eng.add_argument("--output", "-o", default=None, metavar="FILE")
    return p


def _parse_size(text: str) -> tuple[int, int]:
    t = text.lower().replace("×", "x")
    try:
        w, h = (int(v) for v in t.split("x", 1))
        if w <= 0 or h <= 0:
            raise ValueError
        return w, h
    except ValueError:
        raise SystemExit(f"  ✗ invalid --sheet-size: {text!r} (expected WxH, e.g. 3168x1280)")


def _resolve_input(path_txt: str) -> str:
    """Relative input paths resolve against the shell cwd (DIFFUSE_ORIG_CWD), not the
    repo dir the run.sh cd'd into before exec."""
    path = Path(path_txt).expanduser()
    if not path.is_absolute():
        base = os.environ.get("DIFFUSE_ORIG_CWD")
        path = (Path(base) / path) if base else path
    return str(path.resolve())


def _parse_parts(specs: list[str]) -> tuple[dict[str, list[str]], list[str]]:
    """--part ROLE[,ROLE]:FILE -> ({path: [roles]}, [paths in first-seen order])."""
    part_files: list[str] = []
    file_roles: dict[str, list[str]] = {}
    for spec in specs:
        if ":" not in spec:
            raise SystemExit(f"  ✗ --part expects ROLE[,ROLE]:FILE — got: {spec!r}")
        roles_txt, file_txt = spec.split(":", 1)
        roles = [PART_ROLE_ALIASES.get(r.strip(), r.strip()) for r in roles_txt.split(",") if r.strip()]
        if not roles:
            raise SystemExit(f"  ✗ --part without a role: {spec!r}")
        bad = [r for r in roles if r not in PART_ROLE_CHOICES]
        if bad:
            raise SystemExit(f"  ✗ unknown role in --part: {','.join(bad)} "
                             f"(valid: {', '.join(PART_ROLE_CHOICES)})")
        if "same_character" in roles and len(roles) > 1:
            raise SystemExit(f"  ✗ same_character cannot be combined with part roles ({spec!r})")
        path = str(Path(file_txt).expanduser().resolve())
        if not Path(path).is_file():
            raise SystemExit(f"  ✗ --part file not found: {file_txt}")
        for r in roles:
            if r in file_roles.get(path, []):
                raise SystemExit(f"  ✗ role {r} repeated for {file_txt}")
        if path not in file_roles:
            file_roles[path] = []
            part_files.append(path)
        file_roles[path].extend(roles)
    return file_roles, part_files


def _resolve_views(preset_or_csv: str) -> list[str]:
    p = preset_or_csv.strip()
    if p.lower() in PRESET_ALIASES:
        from diffuse.sheet.view_catalog_13 import PRESETS
        return list(PRESETS[PRESET_ALIASES[p.lower()]])
    views = [v.strip() for v in p.split(",") if v.strip()]
    bad = [v for v in views if v not in VIEW_IDS]
    if bad or not views:
        raise SystemExit(f"  ✗ unknown view(s): {','.join(bad)} — use a preset "
                         f"(standard5, head6, all13, ...) or a CSV of valid view ids")
    return views


def _build_state(args, views: list[str]) -> dict:
    state = {
        "schema_version": 3, "views": views,
        "size": {"mode": "auto", "body_height": 1120, "manual_width": 2208, "manual_height": 1280},
        "part_prompts": {},
        "hand_state": args.hands,
        "clothing_mode": "garment",
        "clothing_type": "none" if args.no_clothing else args.lower,
        "footwear_state": args.feet,
        "resolution_profile": args.profile,
    }
    if args.sheet_size:
        w, h = _parse_size(args.sheet_size)
        state["size"] = {"mode": "manual", "body_height": 1120,
                         "manual_width": w, "manual_height": h}
    if args.view_prompt:
        view_prompts = {}
        for spec in args.view_prompt:
            if ":" not in spec:
                raise SystemExit(f"  ✗ --view-prompt expects VIEW:TEXT — got: {spec!r}")
            view, text = spec.split(":", 1)
            if view.strip() not in VIEW_IDS:
                raise SystemExit(f"  ✗ unknown view in --view-prompt: {view!r} "
                                 f"(valid: {', '.join(VIEW_IDS)})")
            view_prompts[view.strip()] = text
        state["view_prompts"] = view_prompts
    return state


def _retag_t2i(prompt: str, desc: str) -> str:
    """T2I without a character reference: the description becomes the subject;
    supplementary pictures shift down one position to match the refs list."""
    subject = desc.strip().rstrip(".")

    first = subject[:1]
    if first.isupper() and (len(subject) < 2 or subject[1].islower()):
        subject = first.lower() + subject[1:]
    prompt = prompt.replace("from <Picture 1>", "\x00FROMCHAR\x00")
    prompt = prompt.replace("<Picture 1>", "\x00CHAR\x00")
    for k in (4, 3, 2):
        prompt = prompt.replace(f"<Picture {k}>", f"<Picture {k - 1}>")
    prompt = prompt.replace("\x00FROMCHAR\x00", f"of {subject}")
    return prompt.replace("\x00CHAR\x00", f"the described character ({subject})")


def run_sheet(argv: list[str]) -> None:
    args = build_parser().parse_args(argv)

    if args.model != "h3-image":
        raise SystemExit("  ✗ sheet currently runs only on -m h3-image (the Qwen-Image "
                         "variant of the designer is a different conditioning; later)")

    from diffuse.paths import setup_environment
    setup_environment()

    ref_paths = [_resolve_input(r) for r in (args.ref or [])]
    for r in ref_paths:
        if not Path(r).is_file():
            raise SystemExit(f"  ✗ reference not found: {r}")

    file_roles, part_files = _parse_parts(args.part)
    part_files = [_resolve_input(p) for p in part_files]
    file_roles = {_resolve_input(p): roles for p, roles in file_roles.items()}
    t2i = not ref_paths
    if t2i and not (args.desc or "").strip():
        raise SystemExit('  ✗ describe the character: --desc "description" (T2I) or --ref image.png')

    base_ref: str | None = ref_paths[0] if ref_paths else None
    extra_refs = ref_paths[1:] if base_ref else []

    # Roles declared on the base file attach to Picture 1 (no extra slot).
    base_roles: list[str] = []
    if base_ref and base_ref in file_roles:
        base_roles = file_roles.pop(base_ref)
        part_files = [p for p in part_files if p != base_ref]
        print(f"  Note: --part role(s) {'+'.join(base_roles)} declared on the base "
              f"reference attach to <Picture 1> as a global instruction.")

    # Merge supplements: same file via --ref extra and --part -> explicit role wins.
    supplements: dict[str, str] = {}
    order: list[str] = []
    for path in part_files + extra_refs:
        if path in supplements:
            if path in part_files:
                if supplements[path] == "same_character":
                    supplements[path] = "+".join(file_roles.get(path, [])) or "same_character"
            continue
        supplements[path] = "+".join(file_roles.get(path, [])) if path in file_roles else "same_character"
        if supplements[path] == "":
            supplements[path] = "same_character"
        order.append(path)
    extras = [{"path": p, "role": supplements[p]} for p in order]
    if len(extras) > 3:
        raise SystemExit("  ✗ max 3 supplementary files (--part + extra --ref; the same "
                         "file can carry multiple roles and the base takes none)")

    views = _resolve_views(args.views)
    if args.no_clothing and any(v in CLOTHING_VIEW_IDS for v in views):
        views = [v for v in views if v not in CLOTHING_VIEW_IDS]
        print("  No clothing: outfit views removed.")
    if args.no_clothing and not any(v.startswith(("face_", "body_")) for v in views):
        raise SystemExit("  ✗ --no-clothing needs at least one face or body view")

    if args.layout == "on" and t2i:
        raise SystemExit("  ✗ --layout on requires --ref (the guide positions the reference "
                         "character; in T2I keep layout off)")

    state = _build_state(args, views)
    layout_on = args.layout == "on"
    compile_kwargs = dict(use_layout_image=layout_on, style=args.style)
    if extras:
        state["reference_roles"] = {f"reference_image_{i + 1}": extras[i]["role"]
                                    if i < len(extras) else "same_character"
                                    for i in range(3)}
        compile_kwargs["reference_presence"] = {
            f"reference_image_{i + 1}": i < len(extras) for i in range(3)}

    try:
        result = compile_minimax_image_state(json.dumps(state), **compile_kwargs)
    except StateValidationError as exc:
        raise SystemExit(f"  ✗ compiler: {exc}")

    prompt = result["prompt"]
    if t2i:
        prompt = _retag_t2i(prompt, args.desc)

    # Global role instructions attached to the base reference (Picture 1).
    from diffuse.sheet.appearance_policy_13 import ROLE_NAMES
    for role in base_roles:
        prompt += f"\nUse <Picture 1> for {ROLE_NAMES[role]}."

    width, height = result["width"], result["height"]
    mp = result["pixel_count"] / 1_000_000
    if mp >= BANDING_LIMIT_MP:
        print(f"  ⚠️  {mp:.1f} MP >= {BANDING_LIMIT_MP:.0f} MP: real risk of regular "
              f"vertical banding on this pipeline (decode-level; extra steps do not fix "
              f"it). Prefer --profile current (or ~4 MP --sheet-size) + Real-ESRGAN.")

    seed = args.seed if args.seed is not None else secrets.randbits(31)
    from diffuse.output import resolve_output_path
    output_path = resolve_output_path(args.model, seed, args.output,
                                      cwd=Path(os.environ.get("DIFFUSE_ORIG_CWD") or str(Path.cwd())))

    # Conditioning order must match the compiled <Picture N> numbering exactly:
    # edit mode: base (1), guide (2), extras (3..5). T2I: extras only (renumbered).
    final_refs: list[str] = []
    if base_ref:
        final_refs.append(base_ref)
    guide_path = None
    if layout_on:
        guide_path = output_path.with_name(output_path.stem + "-guide.png")
        guide = render_reference13_guide(result["guide_geometry"], result["semantic_layout"])
        guide_path.parent.mkdir(parents=True, exist_ok=True)
        guide.save(guide_path)
        final_refs.append(str(guide_path))
    final_refs += [e["path"] for e in extras]

    prompt_path = output_path.with_name(output_path.stem + "-prompt.txt")
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(prompt)

    print(f"  📋 Compiled sheet: {result['megapixels']:.2f} MP ({width}x{height}) "
          f"views={args.views} layout={'on' if layout_on else 'off'} style={args.style}")
    pic_map = []
    for i, r in enumerate(final_refs, start=1):
        if guide_path and str(r) == str(guide_path):
            pic_map.append(f"<Picture {i}> layout guide")
        elif i == 1 and not t2i:
            pic_map.append("<Picture 1> base")
        else:
            pic_map.append(f"<Picture {i}> {Path(r).name}")
    print("     " + " | ".join(pic_map)) if pic_map else None
    print(f"     Prompt saved to: {prompt_path}")
    if layout_on:
        print(f"     Guide saved to: {guide_path}")
    print("\n  ── compiled prompt ──")
    print(prompt)
    print("  ──────────────────────\n")

    if args.dry_run:
        print("  Dry run: render skipped (--dry-run).")
        return

    from diffuse.cli import _run_h3img_wan2gp_image
    from diffuse.models import MODELS
    ns = argparse.Namespace(
        steps=args.steps, output=args.output, turbo=args.turbo,
        no_lora_select=args.no_lora_select, lora_selector=args.lora_selector,
        nsfw=args.nsfw, enhance=args.enhance, enhance_with=None,
    )
    _run_h3img_wan2gp_image(
        ns, args.model, MODELS[args.model], prompt,
        original_prompt=prompt, seed=seed, width=width, height=height,
        ref_image_paths=final_refs or None,
        ref_mode="I" if final_refs else None,
    )
