# SPDX-License-Identifier: Apache-2.0
"""Disaggregated-inference connector support decision (plan P0.4).

The requirement is not merely "DI does not work" -- it is that an unsupported
connector is rejected *at configuration time*, loudly, instead of failing deep
inside connector construction or, worse, degrading into a server that runs
without KV transfer while looking healthy.
"""

import pytest


class TestUnsupportedConnectorsAreRejected:
    def test_nixl_connector_is_rejected(self, di_support):
        with pytest.raises(di_support.UnsupportedDIConfigError):
            di_support.check_kv_connector_supported("NeuronNixlConnector")

    def test_rejection_explains_the_cause_and_a_way_forward(self, di_support):
        """An operator hitting this must learn why and what to do.

        Asserted because the entire value of an early rejection is the message;
        a bare exception type at config time is barely better than the
        ImportError it replaces.
        """
        with pytest.raises(di_support.UnsupportedDIConfigError) as excinfo:
            di_support.check_kv_connector_supported("NeuronNixlConnector")

        message = str(excinfo.value)
        assert "NixlConnectorWorker" in message  # the specific upstream removal
        assert "0.26" in message  # the version that removed it
        assert "--kv-transfer-config" in message  # how to run without it

    def test_error_is_a_valueerror(self, di_support):
        """Config validation elsewhere on this path raises ValueError.

        Subclassing it keeps the rejection catchable by anything already
        handling configuration errors, rather than escaping as a novel type.
        """
        assert issubclass(di_support.UnsupportedDIConfigError, ValueError)


class TestSupportedConfigurationsPass:
    def test_no_connector_is_fine(self, di_support):
        """DI off is the default and must never be rejected."""
        di_support.check_kv_connector_supported(None)

    def test_decode_bench_connector_is_not_rejected(self, di_support):
        """Scope check: the bench connector still works at 0.26.

        It subclasses upstream's ``decode_bench_connector``, which 0.26 left
        intact. Rejecting all of DI would remove a working feature on the
        strength of an unrelated breakage, so this pins the narrow scope.
        """
        di_support.check_kv_connector_supported("NeuronDecodeBenchConnector")

    def test_unknown_connectors_pass_through(self, di_support):
        """A deny-list, not an allow-list -- callers may register their own."""
        di_support.check_kv_connector_supported("SomeThirdPartyConnector")
