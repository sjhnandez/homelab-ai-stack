"""Unit tests for litellm/update_free_models.py.

Canned fixtures mimic the shape of OpenRouter's /api/v1/models payload, so the
whole suite runs offline:

* ``select_models`` is always called with ``probe=False``,
* ``main()`` runs with ``--no-probe`` and a monkeypatched ``fetch_catalog``,
* probe tests replace ``urlopen`` / ``probe_model`` at the boundary.

The script itself is not modified by these tests: its module-level state that
needs to vary per test (``CONFIG_PATH``) is monkeypatched instead.
"""

import sys
import urllib.error

import pytest

import update_free_models as ufm


# --------------------------------------------------------------------------- #
# Fixtures / helpers                                                          #
# --------------------------------------------------------------------------- #

def make_model(
    model_id,
    *,
    prompt="0",
    completion="0",
    input_modalities=("text",),
    output_modalities=("text",),
    context_length=65536,
    created=1_700_000_000,
    name=None,
):
    """Build a catalog entry shaped like OpenRouter's model JSON."""
    return {
        "id": model_id,
        "name": name or model_id,
        "created": created,
        "context_length": context_length,
        "pricing": {"prompt": prompt, "completion": completion},
        "architecture": {
            "input_modalities": list(input_modalities),
            "output_modalities": list(output_modalities),
        },
    }


CONFIG_FIXTURE = """\
model_list:
  - model_name: text-chat
    litellm_params:
      model: ollama/gemma3:9b
  # >>> BEGIN AUTO FREE MODELS (managed by update_free_models.py)
  # openrouter/oldmodel - Old Model (65K ctx)
  - model_name: text-chat
    litellm_params:
      model: openrouter/openrouter/oldmodel
      api_key: os.environ/OPENROUTER_API_KEY
    model_info:
      supports_vision: false
  # <<< END AUTO FREE MODELS

litellm_settings:
  drop_params: true
"""


def write_config(tmp_path, text=CONFIG_FIXTURE):
    path = tmp_path / "config.yaml"
    path.write_text(text)
    return path


# --------------------------------------------------------------------------- #
# Pure filter / classification helpers                                        #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "pricing, expected",
    [
        ({"prompt": "0", "completion": "0"}, True),
        ({"prompt": "0", "completion": "0.000001"}, False),
        ({"prompt": "0.0000005", "completion": "0"}, False),
        ({}, False),
        (None, False),
    ],
)
def test_is_free_only_when_both_prices_are_zero(pricing, expected):
    model = make_model("acme/model")
    model["pricing"] = pricing
    assert ufm.is_free(model) is expected


def test_supports_image_reads_input_modalities():
    assert ufm.supports_image(make_model("a", input_modalities=("text", "image"))) is True
    assert ufm.supports_image(make_model("a", input_modalities=("text",))) is False
    assert ufm.supports_image({"id": "a"}) is False  # missing architecture


def test_is_chat_model_drops_audio_video_generators():
    assert ufm.is_chat_model(make_model("a", output_modalities=("text",))) is True
    # Image *output* is still a chat-style model; audio/video are generators.
    assert ufm.is_chat_model(make_model("a", output_modalities=("text", "image"))) is True
    assert ufm.is_chat_model(make_model("a", output_modalities=("audio",))) is False
    assert ufm.is_chat_model(make_model("a", output_modalities=("text", "video"))) is False
    assert ufm.is_chat_model({"id": "a"}) is False  # no output modalities


# --------------------------------------------------------------------------- #
# select_models                                                               #
# --------------------------------------------------------------------------- #

def test_select_models_keeps_only_free_chat_models():
    catalog = [
        make_model("acme/free-text"),
        make_model("acme/paid-text", prompt="0.5", completion="1.5"),
        make_model("acme/free-music", output_modalities=("audio",)),
        make_model("acme/free-unknown-output", output_modalities=()),
    ]
    selected = ufm.select_models(catalog, probe=False)
    assert [m["id"] for m in selected] == ["acme/free-text"]


def test_select_models_splits_text_and_vision_groups():
    catalog = [
        make_model("acme/text-model", created=2),
        make_model("acme/vision-model", input_modalities=("text", "image"), created=1),
    ]
    selected = ufm.select_models(catalog, probe=False)
    rendered = ufm.render_entries(selected)

    assert "  # acme/text-model - acme/text-model (66K ctx)\n" \
        "  - model_name: text-chat\n" in rendered
    assert "  # acme/vision-model - acme/vision-model (66K ctx)\n" \
        "  - model_name: vision-chat\n" in rendered
    # Text group comes before vision group.
    assert rendered.index("text-chat") < rendered.index("vision-chat")


def test_select_models_applies_exclude_patterns_case_insensitively():
    catalog = [
        make_model("inclusionai/ling-fin-1t"),
        make_model("COHERE/north-mini-code"),
        make_model("poolside/laguna-m.1"),
        make_model("acme/keep-me"),
    ]
    selected = ufm.select_models(catalog, probe=False)
    assert [m["id"] for m in selected] == ["acme/keep-me"]


def test_select_models_enforces_min_context():
    catalog = [
        make_model("acme/too-small", context_length=ufm.MIN_CONTEXT - 1),
        make_model("acme/at-threshold", context_length=ufm.MIN_CONTEXT),
        make_model("acme/no-context", context_length=None),
        # Non-numeric values are tolerated by the script (int() raises -> kept).
        make_model("acme/weird-context", context_length="not-a-number"),
    ]
    selected = ufm.select_models(catalog, probe=False)
    assert [m["id"] for m in selected] == ["acme/at-threshold", "acme/weird-context"]


def test_select_models_always_includes_pinned_models():
    # Pinned entries bypass every filter: paid pricing, tiny context, and they
    # are never double-added by the candidate scan.
    pinned = make_model(
        "openrouter/free",
        prompt="0.5",
        completion="0.5",
        context_length=1024,
        name="OpenRouter Free",
    )
    catalog = [pinned, make_model("acme/free-text")]
    selected = ufm.select_models(catalog, probe=False)

    ids = [m["id"] for m in selected]
    assert ids[0] == "openrouter/free"
    assert ids.count("openrouter/free") == 1
    assert "acme/free-text" in ids


def test_select_models_caps_text_and_vision_groups_keeping_newest():
    n_text = ufm.MAX_TEXT_MODELS + 5
    n_vision = ufm.MAX_VISION_MODELS + 5
    catalog = [
        make_model(f"acme/text-{i:02d}", created=100 + i)
        for i in range(1, n_text + 1)
    ] + [
        make_model(
            f"acme/vision-{i:02d}",
            input_modalities=("text", "image"),
            created=200 + i,
        )
        for i in range(1, n_vision + 1)
    ]

    selected = ufm.select_models(catalog, probe=False)

    text_ids = [m["id"] for m in selected if not ufm.supports_image(m)]
    vision_ids = [m["id"] for m in selected if ufm.supports_image(m)]
    # Newest (highest `created`) wins the race for the capped slots.
    assert text_ids == [
        f"acme/text-{i:02d}"
        for i in range(n_text, n_text - ufm.MAX_TEXT_MODELS, -1)
    ]
    assert vision_ids == [
        f"acme/vision-{i:02d}"
        for i in range(n_vision, n_vision - ufm.MAX_VISION_MODELS, -1)
    ]


# --------------------------------------------------------------------------- #
# render_entries / fmt_ctx                                                    #
# --------------------------------------------------------------------------- #

def test_render_entries_formats_deployments():
    models = [
        make_model("openrouter/free", name="OpenRouter Free", context_length=1_000_000),
        make_model("acme/text-model", name="Text Model", context_length=131072),
        make_model(
            "acme/vision-vl",
            name="Vision VL",
            input_modalities=("text", "image"),
            context_length=8192,
        ),
    ]
    expected = "\n".join(
        [
            "  # openrouter/free - OpenRouter Free (1M ctx)",
            "  - model_name: text-chat",
            "    litellm_params:",
            "      model: openrouter/openrouter/free",
            "      api_key: os.environ/OPENROUTER_API_KEY",
            "    model_info:",
            "      supports_vision: false",
            "  # acme/text-model - Text Model (131K ctx)",
            "  - model_name: text-chat",
            "    litellm_params:",
            "      model: openrouter/acme/text-model",
            "      api_key: os.environ/OPENROUTER_API_KEY",
            "    model_info:",
            "      supports_vision: false",
            "  # acme/vision-vl - Vision VL (8K ctx)",
            "  - model_name: vision-chat",
            "    litellm_params:",
            "      model: openrouter/acme/vision-vl",
            "      api_key: os.environ/OPENROUTER_API_KEY",
            "    model_info:",
            "      supports_vision: true",
        ]
    )
    assert ufm.render_entries(models) == expected


@pytest.mark.parametrize(
    "value, expected",
    [
        (128000, "128K"),
        (1_000_000, "1M"),
        (1_048_576, "1M"),
        (512, "512"),
        ("65536", "66K"),
        (None, "?"),
        ("n/a", "?"),
    ],
)
def test_fmt_ctx(value, expected):
    assert ufm.fmt_ctx(value) == expected


# --------------------------------------------------------------------------- #
# splice_config (marker-block rewrite)                                        #
# --------------------------------------------------------------------------- #

def test_splice_config_replaces_only_between_markers(tmp_path, monkeypatch):
    path = write_config(tmp_path)
    monkeypatch.setattr(ufm, "CONFIG_PATH", path)

    new_block = ufm.render_entries(
        [make_model("openrouter/newmodel", name="New Model")]
    )
    changed, added, removed = ufm.splice_config(new_block)

    assert changed is True
    assert added == ["openrouter/newmodel"]
    assert removed == ["openrouter/oldmodel"]

    text = path.read_text()
    # Outside the markers: byte-identical to the fixture.
    begin_line_end = CONFIG_FIXTURE.index("\n", CONFIG_FIXTURE.index(ufm.BEGIN_MARKER)) + 1
    end_line_start = CONFIG_FIXTURE.rindex("\n", 0, CONFIG_FIXTURE.index(ufm.END_MARKER)) + 1
    assert text.startswith(CONFIG_FIXTURE[:begin_line_end])
    assert text.endswith(CONFIG_FIXTURE[end_line_start:])
    # Inside: old block gone, new block present.
    assert "openrouter/oldmodel" not in text
    assert "  # openrouter/newmodel - New Model (66K ctx)" in text


def test_splice_config_is_idempotent(tmp_path, monkeypatch):
    path = write_config(tmp_path)
    monkeypatch.setattr(ufm, "CONFIG_PATH", path)
    block = ufm.render_entries(
        [make_model("openrouter/newmodel", name="New Model")]
    )

    assert ufm.splice_config(block) == (True, ["openrouter/newmodel"], ["openrouter/oldmodel"])
    after_first = path.read_text()

    # Second run with the same block: no change reported, file untouched.
    assert ufm.splice_config(block) == (False, [], [])
    assert path.read_text() == after_first


def test_splice_config_without_markers_raises_system_exit(tmp_path, monkeypatch):
    path = write_config(tmp_path, "model_list: []\n")
    monkeypatch.setattr(ufm, "CONFIG_PATH", path)

    with pytest.raises(SystemExit):
        ufm.splice_config("  # nothing\n")
    assert path.read_text() == "model_list: []\n"


# --------------------------------------------------------------------------- #
# main()                                                                      #
# --------------------------------------------------------------------------- #

def _run_main(monkeypatch, path, catalog, *flags):
    monkeypatch.setattr(ufm, "CONFIG_PATH", path)
    monkeypatch.setattr(ufm, "fetch_catalog", lambda: catalog)
    monkeypatch.setattr(sys, "argv", ["update_free_models.py", *flags])


def test_main_dry_run_does_not_touch_config(tmp_path, monkeypatch, capsys):
    path = write_config(tmp_path)
    _run_main(monkeypatch, path, [make_model("openrouter/newmodel", name="New Model")],
              "--dry-run", "--no-probe")

    assert ufm.main() == 0
    assert path.read_text() == CONFIG_FIXTURE
    out = capsys.readouterr().out
    assert "would be UPDATED" in out
    assert "openrouter/newmodel" in out


def test_main_writes_config_and_reports_changes(tmp_path, monkeypatch, capsys):
    path = write_config(tmp_path)
    _run_main(monkeypatch, path, [make_model("openrouter/newmodel", name="New Model")],
              "--no-probe")

    assert ufm.main() == 0

    out = capsys.readouterr().out
    assert "  + openrouter/newmodel" in out
    assert "  - openrouter/oldmodel" in out

    text = path.read_text()
    assert "  # openrouter/newmodel - New Model (66K ctx)" in text
    assert "openrouter/oldmodel" not in text
    # Content outside the markers survives untouched.
    assert text.startswith("model_list:\n  - model_name: text-chat\n")
    assert text.endswith("\nlitellm_settings:\n  drop_params: true\n")


def test_main_returns_1_when_catalog_fetch_fails(monkeypatch, capsys):
    def boom():
        raise RuntimeError("network down")

    monkeypatch.setattr(ufm, "fetch_catalog", boom)
    monkeypatch.setattr(sys, "argv", ["update_free_models.py"])

    assert ufm.main() == 1
    assert "could not fetch catalog" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# Probe policy (no real network: urlopen / probe_model are mocked)            #
# --------------------------------------------------------------------------- #

class _FakeResp:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.mark.parametrize(
    "exc, expected_ok, needle",
    [
        (None, True, "HTTP 200"),
        (urllib.error.HTTPError("https://openrouter.ai", 404, "gone", None, None),
         False, "404"),
        (urllib.error.HTTPError("https://openrouter.ai", 429, "busy", None, None),
         True, "transient"),
        (TimeoutError("timed out"), True, "keeping"),
    ],
)
def test_probe_model_outcome_policy(monkeypatch, exc, expected_ok, needle):
    if exc is None:
        monkeypatch.setattr(ufm.urllib.request, "urlopen", lambda *a, **k: _FakeResp())
    else:
        def boom(*a, **k):
            raise exc

        monkeypatch.setattr(ufm.urllib.request, "urlopen", boom)

    ok, reason = ufm.probe_model("openrouter/free", "test-key")
    assert ok is expected_ok
    assert needle in reason


def test_probe_candidates_filters_deterministic_failures(monkeypatch):
    good = make_model("openrouter/good")
    dead = make_model("openrouter/dead")

    monkeypatch.setattr(ufm, "_get_openrouter_key", lambda: "test-key")
    monkeypatch.setattr(
        ufm,
        "probe_model",
        lambda mid, key: (mid != "openrouter/dead", "HTTP 404" if mid == "openrouter/dead" else "HTTP 200"),
    )

    kept = ufm.probe_candidates([good, dead])
    assert [m["id"] for m in kept] == ["openrouter/good"]


def test_probe_candidates_skips_probing_without_api_key(monkeypatch, capsys):
    monkeypatch.setattr(ufm, "_get_openrouter_key", lambda: None)
    candidates = [make_model("openrouter/good")]

    assert ufm.probe_candidates(candidates) == candidates
    assert "no OPENROUTER_API_KEY" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# API key resolution order                                                    #
# --------------------------------------------------------------------------- #

def test_get_openrouter_key_prefers_env_over_env_file(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "  env-key  ")
    (tmp_path / ".env").write_text("OPENROUTER_API_KEY=file-key\n")
    monkeypatch.setattr(ufm, "CONFIG_PATH", tmp_path / "litellm" / "config.yaml")

    assert ufm._get_openrouter_key() == "env-key"


def test_get_openrouter_key_falls_back_to_env_file_and_strips_quotes(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    (tmp_path / ".env").write_text('OTHER=x\nOPENROUTER_API_KEY="quoted-key"\n')
    monkeypatch.setattr(ufm, "CONFIG_PATH", tmp_path / "litellm" / "config.yaml")

    assert ufm._get_openrouter_key() == "quoted-key"

    # No env var and no .env file anywhere -> None.
    monkeypatch.setattr(ufm, "CONFIG_PATH", tmp_path / "missing" / "litellm" / "config.yaml")
    assert ufm._get_openrouter_key() is None
