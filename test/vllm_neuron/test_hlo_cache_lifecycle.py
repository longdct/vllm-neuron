# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from vllm_neuron.vllm.worker.hlo_cache import quarantine_incomplete_hlo_captures


PREFIX = "dev0_7"


def _capture(
    cache: Path,
    key: str,
    *,
    prefix: str = PREFIX,
    rank: int = 0,
    complete: bool = False,
) -> Path:
    hash_dir = cache / key
    capture_dir = hash_dir / f"{prefix}.rank{rank}"
    capture_dir.mkdir(parents=True)
    (capture_dir / "graph.hlo").write_text(f"hlo:{key}")
    (capture_dir / "model.fx.txt").write_text("diagnostic FX graph")
    (capture_dir / "compiler.log").write_text("diagnostic compiler log")
    if complete:
        (hash_dir / ".compilation_complete").write_text("completed")
    return capture_dir


def _discover_for_parallel_compile(cache: Path) -> list[str]:
    """Mirror lite's immediate-rank-directory HLO discovery for this server."""
    return sorted(
        hash_dir.name
        for hash_dir in cache.iterdir()
        if hash_dir.is_dir()
        and not (hash_dir / ".compilation_complete").exists()
        and any(
            child.is_dir()
            and child.name.startswith(f"{PREFIX}.rank")
            and (child / "graph.hlo").exists()
            for child in hash_dir.iterdir()
        )
    )


def test_quarantine_is_scoped_preserving_diagnostics_and_recreation(tmp_path):
    stale = _capture(tmp_path, "stale")
    completed = _capture(tmp_path, "completed", complete=True)
    other_prefix = _capture(tmp_path, "other-prefix", prefix="dev8_15")
    other_rank = _capture(tmp_path, "other-rank", rank=1)

    moved = quarantine_incomplete_hlo_captures(tmp_path, PREFIX, 0)

    assert [key for key, _ in moved] == ["stale"]
    destination = moved[0][1]
    assert not stale.exists()
    assert (destination / "graph.hlo").read_text() == "hlo:stale"
    assert (destination / "model.fx.txt").exists()
    assert (destination / "compiler.log").exists()
    assert completed.exists()
    assert other_prefix.exists()
    assert other_rank.exists()

    assert quarantine_incomplete_hlo_captures(tmp_path, PREFIX, 0) == []

    recreated = _capture(tmp_path, "stale")
    assert recreated.exists()
    assert destination.exists()


def test_stale_hlo_is_not_submitted_with_current_extraction(tmp_path):
    _capture(tmp_path, "old-decode")
    quarantine_incomplete_hlo_captures(tmp_path, PREFIX, 0)

    for key in ("prefill-8", "prefill-64", "decode-1x64"):
        _capture(tmp_path, key)

    submitted_keys = _discover_for_parallel_compile(tmp_path)

    assert submitted_keys == ["decode-1x64", "prefill-64", "prefill-8"]
    quarantined_hlo = next(
        (tmp_path / "old-decode" / ".incomplete_hlo_quarantine").glob(
            "*/graph.hlo"
        )
    )
    assert quarantined_hlo.read_text() == "hlo:old-decode"
