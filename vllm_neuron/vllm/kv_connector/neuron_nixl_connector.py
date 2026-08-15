# SPDX-License-Identifier: Apache-2.0
"""
NeuronNixlConnector — standalone NIXL connector for Neuron with DCP support.

Replaces the monkey-patch approach. Users specify this connector directly:

    --kv-transfer-config '{
        "kv_connector": "NeuronNixlConnector",
        "kv_connector_module_path": "vllm_neuron.vllm.kv_connector.neuron_nixl_connector",
        ...
    }'

Supports all DCP DI topologies via unified head_ratio / seq_ratio math:
  - DCP prefill → TP decode
  - DCP prefill → DCP decode (same or different DCP degrees)
  - TP prefill → DCP decode
  - Standard TP → TP (passthrough)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    NixlAgentMetadata,
    NixlHandshakePayload,
)
try:
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.connector import (
        NixlConnector,
        NixlConnectorWorker,
    )
except ImportError as exc:  # pragma: no cover - exercised by test_di_support
    # vLLM 0.26 removed NixlConnectorWorker, splitting the monolithic connector
    # into NixlPullConnector / NixlPushConnector. Re-raise with the cause and a
    # way forward: vLLM imports this module during VllmConfig construction (via
    # kv_connector_module_path), so a bare ImportError here reaches the operator
    # as an unexplained traceback from inside config validation, before the
    # platform's own check_and_update_config guard can run.
    from vllm_neuron.vllm.di_support import unsupported_nixl_connector_error

    raise unsupported_nixl_connector_error() from exc
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.tp_mapping import ReadSpec
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.distributed.kv_transfer.kv_connector.v1.base import (
        KVConnectorRole,
    )
    from vllm.v1.kv_cache_interface import KVCacheConfig

logger = init_logger(__name__)


@dataclass
class NeuronNixlAgentMetadata(NixlAgentMetadata):
    """Extended metadata with Neuron DCP info.

    The prefill advertises its DCP degree so the decode can compute
    correct remote rank mappings without user config.
    """

    dcp_size: int = 0
    physical_blocks_per_logical_kv_block: int = 1
    attn_backend_name: str = "neuron"


class NeuronNixlConnectorWorker(NixlConnectorWorker):
    """NIXL connector worker with DCP-aware KV transfer.

    Uses unified head_ratio / seq_ratio logic to handle all topologies:
    - head_ratio > 1: split (remote block has more heads, read a subset)
    - head_ratio < 1: merge (remote block has fewer heads, read multiple)
    - seq_ratio > 1: read from multiple remote DCP ranks
    - seq_ratio <= 1: read from subset of one remote DCP rank's blocks
    """

    def __init__(self, *args, **kwargs):
        # The upstream decoder builds msgspec.msgpack.Decoder(NixlAgentMetadata)
        # inside _nixl_handshake, resolving NixlAgentMetadata from the *worker*
        # module's namespace (it was bound there at import via `from ...metadata
        # import NixlAgentMetadata`). Rebinding the attribute on the metadata
        # module alone does NOT change that already-imported name, so the decode
        # falls back to the base class and `dcp_size` is missing. Patch the name
        # in the worker module (where the decoder actually resolves it) so the
        # prefill's dcp_size field is decoded. Also patch the metadata module for
        # any consumer that resolves it there.
        import vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata as _nixl_mod
        import vllm.distributed.kv_transfer.kv_connector.v1.nixl.worker as _nixl_worker

        if _nixl_mod.NixlAgentMetadata is not NeuronNixlAgentMetadata:
            _nixl_mod.NixlAgentMetadata = NeuronNixlAgentMetadata
        if _nixl_worker.NixlAgentMetadata is not NeuronNixlAgentMetadata:
            _nixl_worker.NixlAgentMetadata = NeuronNixlAgentMetadata

        super().__init__(*args, **kwargs)
        self._local_dcp_size: int = 1
        self._local_dcp_rank: int = 0
        self._remote_dcp: dict[str, int] = {}
        self._head_ratio: dict[str, float] = {}
        self._remote_tp_size: dict[str, int] = {}
        self._inverse_handle_cache: dict[str, list] = {}
        # Per-engine remote geometry tracked by the DCP split/merge paths.
        # vLLM <=0.19 exposed these on the upstream worker; 0.20.2 dropped them
        # in favour of TransferTopology._engines (EngineTransferInfo). The
        # match/passthrough path registers there via super().add_remote_agent;
        # split/merge bypass that, so we track their geometry here.
        self._tp_size: dict[str, int] = {}
        self._block_size: dict[str, int] = {}

    # ── Override: register_kv_caches ─────────────────────────────────────

    def register_kv_caches(self, kv_caches):
        import msgspec

        super().register_kv_caches(kv_caches)
        neuron_cfg = self.vllm_config.additional_config.get("neuron_config", {})

        if neuron_cfg.get("apply_prefill_dcp", False):
            # Prefill: re-encode handshake metadata with DCP degree.
            dcp = self.vllm_config.parallel_config.decode_context_parallel_size
            agent_meta = NeuronNixlAgentMetadata(
                engine_id=self.engine_id,
                agent_metadata=self.nixl_wrapper.get_agent_metadata(),
                device_id=self.device_id,
                kv_caches_base_addr=self.kv_caches_base_addr[self.engine_id][
                    self.tp_rank
                ],
                num_blocks=self.num_blocks,
                block_lens=self.block_len_per_layer,
                kv_cache_layout=self.kv_cache_layout
                if not self.use_host_buffer
                else self.host_buffer_kv_cache_layout,
                block_size=self.block_size,
                ssm_sizes=self._mamba_ssm_size,
                attn_backend_name="neuron",
                physical_blocks_per_logical_kv_block=1,
                dcp_size=dcp,
            )
            encoder = msgspec.msgpack.Encoder()
            self.xfer_handshake_metadata = NixlHandshakePayload(
                compatibility_hash=self.compat_hash,
                agent_metadata_bytes=encoder.encode(agent_meta),
            )
            logger.info("DCP NIXL: prefill metadata tagged with dcp_size=%d", dcp)
        else:
            # Decode: store local DCP rank for block filtering.
            from vllm.distributed.parallel_state import get_dcp_group

            g = get_dcp_group()
            self._local_dcp_size = g.world_size
            self._local_dcp_rank = g.rank_in_group

        # Patch tp_ratio to always return 1, preventing upstream from
        # applying its own head splitting logic. We handle all DCP
        # topologies in our _read_blocks_for_req / add_remote_agent.
        if self.transfer_topo is not None:
            self.transfer_topo.tp_ratio = lambda remote_tp_size: 1

        if self._local_dcp_size > 1:
            logger.info(
                "DCP NIXL: local decode dcp_size=%d, dcp_rank=%d, tp_rank=%d",
                self._local_dcp_size,
                self._local_dcp_rank,
                self.tp_rank,
            )

    # ── Override: _validate_remote_agent_handshake ────────────────────────

    def _validate_remote_agent_handshake(self, nixl_agent_meta, remote_tp_size):
        """Skip upstream strict validation for all DCP topologies as the upstream doesn't consider DCP configs."""
        assert self.transfer_topo is not None
        # both should have num_kv_cache_groups entres
        assert len(nixl_agent_meta.kv_caches_base_addr) == len(
            self.block_len_per_layer
        ), "KV group count mismatch between P and D"
        remote_block_len = nixl_agent_meta.block_lens[0]
        local_block_len = self.block_len_per_layer[0]
        assert all(bl == remote_block_len for bl in nixl_agent_meta.block_lens), (
            f"Remote block_lens not uniform: {nixl_agent_meta.block_lens}"
        )
        assert all(bl == local_block_len for bl in self.block_len_per_layer), (
            f"Local block_lens not uniform: {self.block_len_per_layer}"
        )
        ratio = remote_block_len / local_block_len
        inv_ratio = local_block_len / remote_block_len
        assert ratio == int(ratio) or inv_ratio == int(inv_ratio), (
            f"Block lens must be integer-divisible: "
            f"remote_block_len={remote_block_len}, local_block_len={local_block_len}"
        )

    # ── Override: add_remote_agent ────────────────────────────────────────

    def add_remote_agent(self, nixl_agent_meta, remote_tp_rank=0, remote_tp_size=1):
        """Register remote agent with head_ratio-aware descriptors."""
        eid = nixl_agent_meta.engine_id
        remote_dcp = getattr(nixl_agent_meta, "dcp_size", 0) or 1
        remote_block_len = nixl_agent_meta.block_lens[0]
        local_block_len = self.block_len_per_layer[0]

        head_ratio = remote_block_len / local_block_len

        if eid not in self._remote_dcp:
            self._remote_dcp[eid] = remote_dcp
            self._head_ratio[eid] = head_ratio
            self._remote_tp_size[eid] = remote_tp_size
            logger.info(
                "DCP NIXL: engine %s head_ratio=%.2f, remote_dcp=%d, "
                "remote_block_len=%d, local_block_len=%d, "
                "P_TP=%d, D_TP=%d, D_DCP=%d",
                eid,
                head_ratio,
                remote_dcp,
                remote_block_len,
                local_block_len,
                remote_tp_size,
                self.world_size,
                self._local_dcp_size,
            )

        # For split/merge/match we handle descriptor registration ourselves.
        # Match case (head_ratio=1) also goes through our path to avoid
        # upstream's _group_spec_types / tp_mapping logic which breaks for DCP.
        if eid not in self.dst_num_blocks:
            self.dst_num_blocks[eid] = (
                nixl_agent_meta.num_blocks
            )  # total capacity (all slots in the KV cache), not specific to a certain request

        self.kv_caches_base_addr[eid][remote_tp_rank] = (
            nixl_agent_meta.kv_caches_base_addr
        )

        remote_agent_name = self.nixl_wrapper.add_remote_agent(
            nixl_agent_meta.agent_metadata
        )
        self._remote_agents[eid][remote_tp_rank] = remote_agent_name

        # Register with TransferTopology so _read_blocks can resolve get_engine_info()
        assert self.transfer_topo is not None
        if eid not in self.transfer_topo._engines:
            from vllm.distributed.kv_transfer.kv_connector.utils import (
                EngineTransferInfo,
            )

            self.transfer_topo.register_remote_engine(
                eid,
                EngineTransferInfo(
                    remote_tp_size=remote_tp_size,
                    remote_block_len=nixl_agent_meta.block_lens[0],
                    remote_block_size=nixl_agent_meta.block_size,
                    remote_physical_blocks_per_logical=1,
                ),
            )

        if head_ratio > 1:
            # Split: remote block has more heads than local.
            num_kv_heads = self.vllm_config.model_config.get_total_num_kv_heads()
            D_ranks_per_dcp_group = self.world_size // self._local_dcp_size
            D_num_kv_replica = max(1, D_ranks_per_dcp_group // num_kv_heads)
            D_kv_head_rank = self.tp_rank // (self._local_dcp_size * D_num_kv_replica)
            head_idx = D_kv_head_rank % int(head_ratio)
            head_offset = head_idx * local_block_len
            self._register_remote_descs(
                eid,
                nixl_agent_meta,
                remote_agent_name,
                remote_tp_rank,
                desc_len=local_block_len,
                offset=head_offset,
            )
        else:
            # Merge: register full remote block.
            self._register_remote_descs(
                eid,
                nixl_agent_meta,
                remote_agent_name,
                remote_tp_rank,
                desc_len=remote_block_len,
                offset=0,
            )

        return remote_agent_name

    def _register_remote_descs(
        self,
        eid,
        nixl_agent_meta,
        remote_agent_name,
        remote_tp_rank,
        desc_len,
        offset,
    ):
        """Register NIXL descriptors for a remote agent's blocks."""
        kv_addrs = nixl_agent_meta.kv_caches_base_addr
        blocks_data = []
        for kv_group_idx, base_addr in enumerate(kv_addrs):
            page_size = nixl_agent_meta.block_lens[kv_group_idx]
            for block_id in range(nixl_agent_meta.num_blocks):
                addr = base_addr + block_id * page_size + offset
                blocks_data.append((addr, desc_len, nixl_agent_meta.device_id))
        descs = self.nixl_wrapper.get_xfer_descs(blocks_data, self.nixl_memory_type)
        handle = self.nixl_wrapper.prep_xfer_dlist(remote_agent_name, descs)
        self.dst_xfer_side_handles[eid][remote_tp_rank] = handle

    # ── Override: _nixl_handshake ─────────────────────────────────────────

    def _nixl_handshake(self, host, port, remote_tp_size, expected_engine_id):
        """Connect to all remote ranks unconditionally.

        We cannot know the remote's DCP degree before the handshake, so
        always register all remote ranks. The read logic selects the
        correct subset at transfer time.
        """
        self.transfer_topo.handshake_target_ranks = lambda rtp: list(range(rtp))
        try:
            result = super()._nixl_handshake(
                host, port, remote_tp_size, expected_engine_id
            )
        finally:
            if "handshake_target_ranks" in self.transfer_topo.__dict__:
                del self.transfer_topo.__dict__["handshake_target_ranks"]
        return result

    # ── Override: _read_blocks_for_req ────────────────────────────────────

    def _read_blocks_for_req(self, req_id, meta):
        """Unified DCP-aware block reading using head topology and seq_ratio."""
        assert meta.remote is not None and self.transfer_topo is not None
        eid = meta.remote.engine_id

        P_DCP = self._remote_dcp.get(eid, 1)
        D_DCP = self._local_dcp_size
        head_ratio = self._head_ratio.get(eid, 1.0)

        P_TP = self._remote_tp_size[eid]
        seq_ratio = P_DCP / D_DCP

        D_rank = self.tp_rank
        D_DCP_rank = self._local_dcp_rank

        num_kv_heads = self.vllm_config.model_config.get_total_num_kv_heads()
        D_ranks_per_dcp_group = self.world_size // D_DCP
        P_ranks_per_dcp_group = P_TP // P_DCP
        D_num_kv_replica = max(1, D_ranks_per_dcp_group // num_kv_heads)
        P_num_kv_replica = max(1, P_ranks_per_dcp_group // num_kv_heads)
        D_kv_head_rank = D_rank // (D_DCP * D_num_kv_replica)

        # --- Determine which prefill KV head ranks to read from ---
        if head_ratio >= 1:
            P_kv_head_ranks_to_read = [D_kv_head_rank // int(head_ratio)]
            split_local_handles = None
            remote_block_size = self.transfer_topo.get_engine_info(
                eid
            ).remote_block_size
            local_xfer_side_handle = self.src_xfer_handles_by_block_size[
                remote_block_size
            ]
        else:
            inverse_head_ratio = int(1 / head_ratio)
            P_kv_head_ranks_to_read = [
                D_kv_head_rank * inverse_head_ratio + j
                for j in range(inverse_head_ratio)
            ]
            split_local_handles = self._get_or_create_inverse_handles(
                eid, inverse_head_ratio
            )

        # --- Determine which prefill DCP ranks and block mappings ---
        # meta.remote.block_ids is a list of lists — one inner list per KV cache group.
        num_groups = len(meta.remote.block_ids) if meta.remote.block_ids else 0

        if seq_ratio > 1:
            int_seq_ratio = int(seq_ratio)
            P_DCP_ranks_to_read = [D_DCP_rank + i * D_DCP for i in range(int_seq_ratio)]
            remote_blk_id_indices_per_dcp_rank = []
            local_blk_id_indices_per_dcp_rank = []
            for k in range(int_seq_ratio):
                remote_indices_per_group = []
                local_indices_per_group = []
                for g in range(num_groups):
                    num_remote_blks = len(meta.remote.block_ids[g])
                    num_local_blks = len(meta.local_physical_block_ids[g])
                    local_indices = [
                        k + i * int_seq_ratio
                        for i in range(num_remote_blks)
                        if k + i * int_seq_ratio < num_local_blks
                    ]
                    remote_indices_per_group.append(list(range(len(local_indices))))
                    local_indices_per_group.append(local_indices)
                remote_blk_id_indices_per_dcp_rank.append(remote_indices_per_group)
                local_blk_id_indices_per_dcp_rank.append(local_indices_per_group)
        else:
            P_DCP_ranks_to_read = [D_DCP_rank % P_DCP]
            step = D_DCP // P_DCP
            offset = D_DCP_rank // P_DCP
            remote_indices_per_group = []
            local_indices_per_group = []
            for g in range(num_groups):
                num_local_blks = len(
                    meta.local_physical_block_ids[g]
                )  # valid blocks for the request
                num_remote_blks = len(meta.remote.block_ids[g])
                remote_indices = [
                    offset + i * step
                    for i in range(num_local_blks)
                    if offset + i * step < num_remote_blks
                ]
                remote_indices_per_group.append(remote_indices)
                local_indices_per_group.append(list(range(len(remote_indices))))
            remote_blk_id_indices_per_dcp_rank = [remote_indices_per_group]
            local_blk_id_indices_per_dcp_rank = [local_indices_per_group]

        def get_p_rank_id(p_dcp_rank, p_kv_head_rank):
            return (
                p_dcp_rank * P_ranks_per_dcp_group + p_kv_head_rank * P_num_kv_replica
            )

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Unified read: req=%s, D_rank=%d, D_DCP_rank=%d/%d, "
                "D_kv_head_rank=%d, seq_ratio=%.2f, "
                "P_kv_head_ranks=%s, P_DCP_ranks=%s",
                req_id,
                D_rank,
                D_DCP_rank,
                D_DCP,
                D_kv_head_rank,
                seq_ratio,
                P_kv_head_ranks_to_read,
                P_DCP_ranks_to_read,
            )

        # --- Issue reads ---
        for i, p_dcp_rank in enumerate(P_DCP_ranks_to_read):
            remote_indices_per_group = remote_blk_id_indices_per_dcp_rank[i]
            local_indices_per_group = local_blk_id_indices_per_dcp_rank[i]

            if not any(local_indices_per_group):
                continue

            remote_block_ids = [
                [meta.remote.block_ids[g][idx] for idx in remote_indices_per_group[g]]
                for g in range(num_groups)
            ]
            local_block_ids = [
                [
                    meta.local_physical_block_ids[g][idx]
                    for idx in local_indices_per_group[g]
                ]
                for g in range(num_groups)
            ]

            for j, p_kv_head_rank in enumerate(P_kv_head_ranks_to_read):
                p_rank_id = get_p_rank_id(p_dcp_rank, p_kv_head_rank)
                remote_xfer_side_handle = self.dst_xfer_side_handles[eid][p_rank_id]

                if split_local_handles is not None:
                    cur_local_handle = split_local_handles[j]
                else:
                    cur_local_handle = local_xfer_side_handle

                self._read_blocks(
                    read_spec=ReadSpec(
                        remote_rank=p_rank_id,
                        local_block_ids=local_block_ids,
                        remote_block_ids=remote_block_ids,
                    ),
                    dst_engine_id=eid,
                    request_id=req_id,
                    remote_request_id=meta.remote.request_id,
                    local_xfer_side_handle=cur_local_handle,
                    remote_xfer_side_handle=remote_xfer_side_handle,
                )

        # Ensure get_finished() can mark this request as done even if all
        # remote ranks had empty blocks (defaultdict creates the empty list).
        if req_id not in self._recving_transfers:
            self._recving_transfers.setdefault(req_id, [])

    def _get_or_create_inverse_handles(self, eid, inverse_head_ratio):
        """Create local split handles for merge (inverse_head_ratio > 1)."""
        if eid in self._inverse_handle_cache:
            return self._inverse_handle_cache[eid]

        remote_block_len = self.block_len_per_layer[0] // inverse_head_ratio
        local_base_addrs = self.kv_caches_base_addr[self.engine_id][self.tp_rank]
        split_handles = []
        for hp in range(inverse_head_ratio):
            blocks_data = []
            for kv_group_idx, base_addr in enumerate(local_base_addrs):
                local_page_size = self.block_len_per_layer[kv_group_idx]
                for block_id in range(self.num_blocks):
                    addr = (
                        base_addr + block_id * local_page_size + hp * remote_block_len
                    )
                    blocks_data.append((addr, remote_block_len, self.device_id))
            descs = self.nixl_wrapper.get_xfer_descs(blocks_data, self.nixl_memory_type)
            handle = self.nixl_wrapper.prep_xfer_dlist("NIXL_INIT_AGENT", descs)
            split_handles.append(handle)
        self._inverse_handle_cache[eid] = split_handles
        return split_handles


class NeuronNixlConnector(NixlConnector):
    """Neuron NIXL connector with DCP support."""

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: "KVConnectorRole",
        kv_cache_config: "KVCacheConfig",
    ):
        from vllm.distributed.kv_transfer.kv_connector.v1.base import (
            KVConnectorBase_V1,
        )
        from vllm.distributed.kv_transfer.kv_connector.v1.nixl.connector import (
            KVConnectorRole,
            NixlConnectorScheduler,
        )

        KVConnectorBase_V1.__init__(self, vllm_config, role, kv_cache_config)
        assert vllm_config.kv_transfer_config is not None
        assert vllm_config.kv_transfer_config.engine_id is not None
        self.kv_cache_config = kv_cache_config
        self.engine_id = vllm_config.kv_transfer_config.engine_id
        self.kv_transfer_config = vllm_config.kv_transfer_config

        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler = NixlConnectorScheduler(
                vllm_config, self.engine_id, kv_cache_config
            )
            self.connector_worker = None
        elif role == KVConnectorRole.WORKER:
            self.connector_scheduler = None
            self.connector_worker = NeuronNixlConnectorWorker(
                vllm_config, self.engine_id, kv_cache_config
            )
