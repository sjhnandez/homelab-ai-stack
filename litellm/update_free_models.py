#!/usr/bin/env python3
"""
Auto-sync LiteLLM free-model deployments from OpenRouter's live catalog.

Why: hand-maintaining ":free" model slugs in config.yaml goes stale the moment
OpenRouter rotates free models (e.g. a model is pulled from the free tier and
starts returning 404). This script treats OpenRouter's own model catalog as the
source of truth and rewrites ONLY the "auto free models" block of config.yaml.

It splits the current free models into the same two deployment groups the
config already uses:
    * text-chat    -> models whose inputs are text only  (supports_vision: false)
    * vision-chat  -> models that accept image inputs    (supports_vision: true)

Every candidate is additionally verified with a minimal live request before it
is written to the config. This catches two classes of rot the catalog does not
advertise:
    * models pulled from the free tier that still 404 at inference time
    * models gated to specific apps (HTTP 403 "only available on agentic
      harnesses") which silently poison a load-balanced group
Probe policy: 403/404 (and other deterministic 4xx) -> excluded;
429/5xx/network errors are treated as transient -> kept.

Tuning knobs (edit the ALL-CAPS constants below):
    PINNED_MODELS      ids that are always kept (OpenRouter's dynamic free
                       router is pinned because it never goes stale).
    EXCLUDE_PATTERNS   lowercase substrings matched against a model id; models
                       matching any are skipped (domain/content-safety/code
                       tuned models that don't belong in a general chat alias).
    MIN_CONTEXT        drop free models with a smaller context window.
    MAX_PER_GROUP      safety cap on deployments per group.

Usage:
    python3 update_free_models.py --dry-run   # show what would change
    python3 update_free_models.py             # write config if changed
    python3 update_free_models.py --restart   # write config + recreate container
    python3 update_free_models.py --no-probe  # skip live verification probes

Schedule it (e.g. hourly) so the list never rots:
    0 * * * * cd /home/server/stacks/apps/litellm && python3 update_free_models.py --restart >> update_free_models.log 2>&1
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

# --------------------------------------------------------------------------- #
# Tuning knobs                                                                 #
# --------------------------------------------------------------------------- #

CATALOG_URL = "https://openrouter.ai/api/v1/models"

# Model ids always included regardless of filters. "openrouter/free" is
# OpenRouter's own dynamic router: it never 404s and guarantees the text-chat
# group is never empty even if every concrete free model is rotated out.
# Pinned models are always placed in the text-chat group (text-only fallback).
PINNED_MODELS = ["openrouter/free"]

# Lowercased substrings; matching ids are skipped. Edit freely to taste.
# Defaults filter out guard/content-safety, domain-tuned (finance/health),
# code-specialist, and tiny/notebook models from the general chat aliases.
EXCLUDE_PATTERNS = [
    "content-safety",   # guardrail model, not for chat
    "north-mini",       # code-specialist (Cohere)
    "laguna",           # code-specialist (Poolside)
    "-sante",           # health-tuned (InclusionAI Ling)
    "-fin",             # finance-tuned (InclusionAI Ling)
    "dots-3",           # note-preview model
    "lfm-2.5-2.6b",     # 2.6B tiny model
]

# Drop free models with a context window below this many tokens.
MIN_CONTEXT = 32768

# Safety caps so a sudden flood of free models doesn't bloat the config.
MAX_TEXT_MODELS = 12
MAX_VISION_MODELS = 8

# Live verification probes (see module docstring). Set PROBE_ENABLED = False or
# pass --no-probe to skip them (metadata-only filtering).
PROBE_ENABLED = True
PROBE_TIMEOUT = 60        # seconds per probe
PROBE_WORKERS = 6         # parallel probes

# --------------------------------------------------------------------------- #
# Marker handling                                                              #
# --------------------------------------------------------------------------- #

BEGIN_MARKER = "# >>> BEGIN AUTO FREE MODELS (managed by update_free_models.py)"
END_MARKER = "# <<< END AUTO FREE MODELS"

CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"
COMPOSE_PATH = Path(__file__).resolve().parent.parent / "docker-compose.yaml"
SERVICE_NAME = "litellm"


def fetch_catalog() -> list[dict]:
    req = urllib.request.Request(
        CATALOG_URL, headers={"User-Agent": "litellm-free-sync/1.0"}
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.load(resp)
    return data.get("data", [])


def is_free(model: dict) -> bool:
    pricing = model.get("pricing") or {}
    return pricing.get("prompt") == "0" and pricing.get("completion") == "0"


def supports_image(model: dict) -> bool:
    arch = model.get("architecture") or {}
    return "image" in (arch.get("input_modalities") or [])


def is_chat_model(model: dict) -> bool:
    """Exclude audio/video *generation* models (e.g. music, video synth).

    Chat models answer with text; generators emit audio/video as output.
    """
    arch = model.get("architecture") or {}
    out = set(arch.get("output_modalities") or [])
    return bool(out) and not (out & {"audio", "video"})


def fmt_ctx(n) -> str:
    try:
        n = int(n)
    except (TypeError, ValueError):
        return "?"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.0f}M"
    if n >= 1_000:
        return f"{n / 1_000:.0f}K"
    return str(n)


def _get_openrouter_key() -> str | None:
    """Prefer env var; fall back to the stack's .env (kept out of logs)."""
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        return key.strip()
    env_path = CONFIG_PATH.parent.parent / ".env"
    try:
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("OPENROUTER_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    except OSError:
        pass
    return None


def probe_model(model_id: str, key: str) -> tuple[bool, str]:
    """Minimal live request. Returns (usable, reason).

    Deterministic 4xx (403 gated / 404 gone / 400 incompatible) -> excluded.
    429 (busy) / 5xx / network errors are transient -> kept.
    """
    body = json.dumps(
        {
            "model": model_id,
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "ping"}],
        }
    ).encode()
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=PROBE_TIMEOUT) as resp:
            return resp.status == 200, f"HTTP {resp.status}"
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode("utf-8", "replace")[:160]
        except Exception:
            pass
        if e.code == 429:
            return True, "rate-limited now (transient) - keeping"
        if 400 <= e.code < 500:
            return False, f"HTTP {e.code} {detail}"
        return True, f"HTTP {e.code} (server-side, transient) - keeping"
    except Exception as e:  # timeout / network blip
        return True, f"probe error ({e}) - keeping"


def probe_candidates(candidates: list[dict]) -> list[dict]:
    """Run live probes in parallel; drop models that deterministically fail."""
    key = _get_openrouter_key()
    if not key:
        print(
            "WARNING: no OPENROUTER_API_KEY found - skipping live probes",
            file=sys.stderr,
        )
        return candidates

    kept: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=PROBE_WORKERS) as ex:
        futures = {ex.submit(probe_model, m["id"], key): m for m in candidates}
        for fut in concurrent.futures.as_completed(futures):
            m = futures[fut]
            try:
                ok, reason = fut.result()
            except Exception as e:  # never let a probe crash the sync
                ok, reason = True, f"probe exception ({e}) - keeping"
            if ok:
                kept.append(m)
            else:
                print(f"  probe excluded {m['id']}: {reason}")
    return kept


def select_models(models: list[dict], probe: bool = PROBE_ENABLED) -> list[dict]:
    """Apply free + exclusion + context + liveness filters. Returns catalog entries."""
    pinned = [m for m in models if m.get("id") in PINNED_MODELS]
    picked: list[dict] = []
    for m in models:
        mid = m.get("id", "")
        if mid in PINNED_MODELS:
            continue  # handled above
        if not is_free(m):
            continue
        if not is_chat_model(m):
            continue
        low = mid.lower()
        if any(p in low for p in EXCLUDE_PATTERNS):
            continue
        try:
            if int(m.get("context_length") or 0) < MIN_CONTEXT:
                continue
        except (TypeError, ValueError):
            pass
        picked.append(m)

    if probe:
        print(f"Probing {len(picked)} candidate models...")
        picked = probe_candidates(picked)

    # Newest first (newest releases are the current flagship gen).
    picked.sort(key=lambda m: m.get("created") or 0, reverse=True)
    text = [m for m in picked if not supports_image(m)]
    vision = [m for m in picked if supports_image(m)]
    return pinned + text[:MAX_TEXT_MODELS] + vision[:MAX_VISION_MODELS]


def render_entries(models: list[dict]) -> str:
    """Render the YAML deployment block (indented inside model_list)."""
    pinned = [m for m in models if m.get("id") in PINNED_MODELS]
    others = [m for m in models if m.get("id") not in PINNED_MODELS]

    def group_of(m: dict) -> str:
        # Pinned models (e.g. openrouter/free dynamic router) act as a
        # guaranteed text-only fallback, matching the original intent.
        if m.get("id") in PINNED_MODELS:
            return "text-chat"
        return "vision-chat" if supports_image(m) else "text-chat"

    # Pinned dynamic router goes first, under text-chat.
    ordered = list(pinned) + [m for m in others if group_of(m) == "text-chat"] + \
        [m for m in others if group_of(m) == "vision-chat"]

    lines: list[str] = []
    for m in ordered:
        mid = m["id"]
        name = (m.get("name") or mid).strip()
        ctx = fmt_ctx(m.get("context_length"))
        lines.append(f"  # {mid} - {name} ({ctx} ctx)")
        lines.append("  - model_name: " + group_of(m))
        lines.append("    litellm_params:")
        lines.append(f"      model: openrouter/{mid}")
        lines.append("      api_key: os.environ/OPENROUTER_API_KEY")
        is_vision = m.get("id") not in PINNED_MODELS and supports_image(m)
        lines.append("    model_info:")
        lines.append(f"      supports_vision: {str(is_vision).lower()}")
    return "\n".join(lines)


def splice_config(new_block: str) -> tuple[bool, list[str], list[str]]:
    """Replace content between markers in config.yaml.

    Returns (changed, added_ids, removed_ids).
    """
    old_text = CONFIG_PATH.read_text()
    lines = old_text.splitlines(keepends=True)

    begin_idx = next(
        (i for i, l in enumerate(lines) if BEGIN_MARKER in l), None
    )
    end_idx = next((i for i, l in enumerate(lines) if END_MARKER in l), None)
    if begin_idx is None or end_idx is None or end_idx <= begin_idx:
        raise SystemExit(
            f"Markers not found in {CONFIG_PATH}. Add:\n"
            f"{BEGIN_MARKER}\n{END_MARKER}\ninside model_list."
        )

    # Capture current ids inside the block for the change summary.
    def ids_between(a: int, b: int) -> list[str]:
        out = []
        for l in lines[a + 1 : b]:
            s = l.strip()
            if s.startswith("# ") and "openrouter/" in s:
                out.append(s.split("-", 1)[0].strip().lstrip("# ").strip())
        return out

    old_ids = ids_between(begin_idx, end_idx)
    new_ids = [
        l.split(" - ")[0].strip().lstrip("# ").strip()
        for l in new_block.splitlines()
        if l.strip().startswith("# ") and "openrouter/" in l
    ]

    new_text = (
        "".join(lines[: begin_idx + 1])
        + "\n"
        + new_block
        + "\n"
        + "".join(lines[end_idx:])
    )
    if new_text == old_text:
        return False, [], []

    tmp = CONFIG_PATH.with_suffix(".yaml.tmp")
    tmp.write_text(new_text)
    os.replace(tmp, CONFIG_PATH)

    removed = [i for i in old_ids if i not in new_ids]
    added = [i for i in new_ids if i not in old_ids]
    return True, added, removed


def restart_container() -> None:
    cmd = [
        "docker", "compose", "-f", str(COMPOSE_PATH),
        "up", "-d", "--force-recreate", SERVICE_NAME,
    ]
    print(f"Recreating {SERVICE_NAME} container...")
    subprocess.run(cmd, check=True, cwd=str(COMPOSE_PATH.parent))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true",
                    help="show what would change without writing")
    ap.add_argument("--restart", action="store_true",
                    help="recreate the litellm container after a change")
    ap.add_argument("--no-probe", action="store_true",
                    help="skip live verification probes (metadata-only)")
    args = ap.parse_args()

    try:
        catalog = fetch_catalog()
    except Exception as e:  # network errors should not take down the proxy
        print(f"ERROR: could not fetch catalog: {e}", file=sys.stderr)
        return 1

    selected = select_models(catalog, probe=(PROBE_ENABLED and not args.no_probe))
    block = render_entries(selected)

    if args.dry_run:
        print("Catalog fetch OK. Selected free models:")
        for m in selected:
            mid = m["id"]
            tag = "text" if (mid in PINNED_MODELS or not supports_image(m)) else "vision"
            print(f"  [{tag:6s}] {mid}")
        # Simulate splice without writing to show the diff
        old_text = CONFIG_PATH.read_text()
        lines = old_text.splitlines(keepends=True)
        b = next(i for i, l in enumerate(lines) if BEGIN_MARKER in l)
        e = next(i for i, l in enumerate(lines) if END_MARKER in l)
        sim = "".join(lines[: b + 1]) + "\n" + block + "\n" + "".join(lines[e:])
        if sim == old_text:
            print("\nConfig already up to date.")
        else:
            print("\nConfig would be UPDATED (see diff):")
            import difflib
            sys.stdout.writelines(difflib.unified_diff(
                old_text.splitlines(True), sim.splitlines(True),
                fromfile="config.yaml", tofile="config.yaml (new)", n=1))
        return 0

    try:
        changed, added, removed = splice_config(block)
    except SystemExit as e:
        print(e, file=sys.stderr)
        return 1

    if not changed:
        print("No changes - config already matches the live catalog.")
        return 0

    if added:
        print("Added:")
        for i in added:
            print(f"  + {i}")
    if removed:
        print("Removed (no longer free / filtered):")
        for i in removed:
            print(f"  - {i}")
    print(f"Updated {CONFIG_PATH}")

    if args.restart:
        try:
            restart_container()
        except subprocess.CalledProcessError as e:
            print(f"ERROR: container restart failed: {e}", file=sys.stderr)
            return 1
    else:
        print("Config written. Restart litellm to apply: "
              "--restart flag or 'docker compose up -d --force-recreate litellm'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
