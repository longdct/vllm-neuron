# SPDX-License-Identifier: Apache-2.0

import importlib.util
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[4]
    / "tools"
    / "deepseek_v4"
    / "analyze_compile_artifacts.py"
)
SPEC = importlib.util.spec_from_file_location("deepseek_compile_artifacts", MODULE_PATH)
artifacts = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(artifacts)


def test_gnu_time_measurements_are_normalized(tmp_path):
    log = tmp_path / "compile.log"
    log.write_text(
        "Maximum resident set size (kbytes): 22528000\n"
        "Elapsed (wall clock) time (h:mm:ss or m:ss): 9:41.25\n"
    )
    result = artifacts.analyze_log(log)
    assert result["time_v_peak_rss_kbytes"] == 22528000
    assert result["time_v_elapsed_seconds"] == 581.25
