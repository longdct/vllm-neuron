# SPDX-License-Identifier: Apache-2.0
"""Which disaggregated-inference connectors this build actually supports.

Kept free of vllm/torch imports so the decision is unit-testable on a bare
interpreter, matching ``scheduler_selection.py``.

Background (execution plan P0.4). vLLM 0.26 split the monolithic NIXL connector
into pull and push variants -- ``NixlPullConnector`` / ``NixlPushConnector``,
each with its own scheduler and worker class -- and removed
``NixlConnectorWorker``, which ``NeuronNixlConnector`` subclasses. That port is
a substantial rewrite (~15 private parent attributes) and DeepSeek-V4 bring-up
does not need it, so it is deferred.

Deferred must not mean *silent*. Without this check the failure is an
``ImportError`` from deep inside connector construction, or -- worse, on a path
that catches it -- a server that starts and quietly runs without KV transfer,
which looks like a performance problem rather than a missing feature. So the
rejection is explicit, names the cause, and happens at configuration time.

This is a *development*-gate decision. Publishing ``0.26.0.1.0.0`` as a normal
replacement for 0.21 requires the connector restored (plan P0.5, release gate);
shipping it with DI rejected is only acceptable for an explicitly
experimental/pre-release artifact.
"""

#: Connectors this build cannot construct, mapped to why.
#:
#: Scoped deliberately to the one connector that is actually broken. The Neuron
#: decode-bench connector subclasses upstream's ``decode_bench_connector``, which
#: 0.26 left intact, so it still works -- rejecting all of DI would remove a
#: working feature on the strength of an unrelated breakage.
UNSUPPORTED_KV_CONNECTORS = {
    "NeuronNixlConnector": (
        "NeuronNixlConnector has not been ported to vLLM 0.26. Upstream replaced "
        "the monolithic NixlConnector/NixlConnectorWorker pair with separate pull "
        "and push connectors (NixlPullConnector, NixlPushConnector), and the "
        "NixlConnectorWorker base class this connector subclasses no longer exists."
    ),
}


class UnsupportedDIConfigError(ValueError):
    """Raised at configuration time for a KV connector this build cannot run."""


def _message(kv_connector: str, reason: str) -> str:
    return (
        f"kv_connector={kv_connector!r} is not supported by this build of "
        f"vllm-neuron. {reason}\n\n"
        f"Disaggregated inference is unavailable in this release. Remove "
        f"--kv-transfer-config to run without it, or use a vllm-neuron build "
        f"pinned to vLLM 0.21."
    )


def unsupported_nixl_connector_error() -> UnsupportedDIConfigError:
    """The error raised when ``neuron_nixl_connector`` fails to import.

    Lives here so the import-time failure and the config-time rejection carry
    the same text. They fire on different paths -- vLLM imports the connector
    module during ``VllmConfig`` construction when ``kv_connector_module_path``
    is given, which happens *before* the platform's ``check_and_update_config``
    runs -- and an operator should not get two different explanations of one
    missing feature depending on how they spelled the flag.
    """
    return UnsupportedDIConfigError(
        _message("NeuronNixlConnector", UNSUPPORTED_KV_CONNECTORS["NeuronNixlConnector"])
    )


def check_kv_connector_supported(kv_connector: str | None) -> None:
    """Raise if *kv_connector* is not supported by this build.

    ``None`` means disaggregated inference is off, which is always fine.
    Unknown names pass through: a caller may have registered their own
    connector, and this is a deny-list of known-broken ones, not an allow-list
    of blessed ones.
    """
    if kv_connector is None:
        return

    reason = UNSUPPORTED_KV_CONNECTORS.get(kv_connector)
    if reason is None:
        return

    raise UnsupportedDIConfigError(_message(kv_connector, reason))
