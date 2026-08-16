# SPDX-License-Identifier: Apache-2.0

import os
import sys

from vllm_neuron.compile.backend import _compiler_command_prefix


def test_binary_compiler_is_executed_directly(tmp_path):
    compiler = tmp_path / "neuronx-cc"
    compiler.write_bytes(b"\x7fELF")
    assert _compiler_command_prefix(os.fspath(compiler)) == [os.fspath(compiler)]


def test_valid_console_script_is_executed_directly(tmp_path):
    compiler = tmp_path / "neuronx-cc"
    compiler.write_text(f"#!{sys.executable}\n")
    assert _compiler_command_prefix(os.fspath(compiler)) == [os.fspath(compiler)]


def test_relocated_venv_launcher_uses_current_python(tmp_path):
    compiler = tmp_path / "neuronx-cc"
    compiler.write_text("#!/missing/old-venv/bin/python\n")
    assert _compiler_command_prefix(os.fspath(compiler)) == [
        sys.executable,
        os.fspath(compiler),
    ]
