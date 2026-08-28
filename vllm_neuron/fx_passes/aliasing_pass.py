# SPDX-License-Identifier: Apache-2.0
import logging
import operator
from collections import OrderedDict

import torch
from torch.fx import Node

from .base import FXPass

logger = logging.getLogger(__name__)


# =========================================================================
# Aliasing operation classification sets
# =========================================================================

ALIASING_METHODS = {
    "t",
    "transpose",
    "view",
    "slice",
    "reshape",
    "permute",
    "expand",
    "expand_as",
    "squeeze",
    "unsqueeze",
    "mH",
    "mT",
    "as_strided",
    "chunk",
    "unfold",
    "narrow",
    "split",
    "unbind",
    "select",
    "unflatten",
    "flatten",
}

ALIASING_ATEN_OPS = {
    "view",
    "reshape",
    "transpose",
    "permute",
    "t",
    "squeeze",
    "unsqueeze",
    "expand",
    "slice",
    "select",
    "narrow",
    "chunk",
    "split",
    "unbind",
    "as_strided",
    "unfold",
    "flatten",
    "unflatten",
    "_unsafe_view",
    "view_as",
    "reshape_as",
    "expand_as",
}


class AliasingOutputRewritePass(FXPass):
    """Rewrite graph outputs so that in-place mutations and NKI kernel aliases
    are surfaced as explicit output tensors with correct ``io_map`` entries.

    The resulting ``io_map = {output_idx: input_idx}`` is forwarded to
    ``convert_fx_to_hlo`` → ``_add_aliasing_info``, which writes HLO
    ``input_output_alias`` entries that allow the Neuron runtime to reuse
    input buffers for outputs instead of allocating new ones.
    """

    def __init__(self):
        pass

    @property
    def name(self) -> str:
        return "aliasing_output_rewrite"

    def run(
        self, gm: torch.fx.GraphModule, **kwargs
    ) -> tuple[torch.fx.GraphModule, dict]:
        """Analyze the graph for aliasing and transform it by adding output
        nodes for input placeholders modified by in-place operations.

        Handles four sources of aliasing:

        1. In-place operations (e.g. ``x.add_(y)``)
        2. ``wrap_nki`` HOP nodes with ``operand_output_aliases``
        3. Torch custom ops with mutable arguments (``is_write=True``)
        4. Passthrough or view ops that yield aliases of an input tensor

        Args:
            gm: The PyTorch FX GraphModule to transform.
            **kwargs: Additional arguments passed from the pass manager.

        Returns:
            Tuple of (transformed GraphModule, metadata dict with ``io_map``
            and ``original_output_count``).

        Raises:
            RuntimeError: If the aliasing analysis or rewrite fails.
        """
        try:
            # alias_chain maps each node to the node it aliases.  Following
            # the chain eventually reaches a graph-level input placeholder
            # (or terminates if the node doesn't derive from an input).
            alias_chain: dict[Node, Node] = {}
            input_placeholders: OrderedDict[str, Node] = OrderedDict()
            # mutated_inputs tracks which input placeholders are modified
            # in-place, keyed by their positional index.
            mutated_inputs: dict[int, Node] = {}

            # Collect input placeholders in order.  The positional index
            # matters because io_map uses it to pair outputs with inputs.
            for node in gm.graph.nodes:
                if node.op == "placeholder":
                    input_name = self._extract_placeholder_name(node)
                    if input_name:
                        input_placeholders[input_name] = node
                        logger.debug(
                            f"Input[{len(input_placeholders) - 1}]: {input_name} "
                            f"(node: {node.name})"
                        )

            # Build three parallel lookup structures:
            #   input_nodes      — O(1) membership test
            #   input_nodes_list — positional index lookup
            #   input_names_list — human-readable names for debug logging
            input_nodes = set(input_placeholders.values())
            input_nodes_list = list(input_placeholders.values())
            input_names_list = list(input_placeholders.keys())

            logger.debug(f"Total inputs: {len(input_placeholders)}")

            # Single-pass analysis: classify every node as a mutation,
            # alias, custom op mutation, or NKI kernel mutation.  Each
            # category updates alias_chain and/or mutated_inputs so the
            # output-rewrite step below knows what to surface.
            for node in gm.graph.nodes:
                if self._is_torch_mutating_op(node):
                    mutated_tensor = self._get_torch_op_mutated_tensor(node)
                    if mutated_tensor is None:
                        continue
                    root_input = self._find_root_input(
                        mutated_tensor, alias_chain, input_nodes
                    )
                    if root_input is not None and root_input in input_nodes:
                        input_index = input_nodes_list.index(root_input)
                        if input_index not in mutated_inputs:
                            mutated_inputs[input_index] = input_nodes_list[input_index]
                            logger.debug(
                                f"Mutation detected: input[{input_index}] "
                                f"({input_names_list[input_index]}) via {node.target}"
                            )
                        alias_chain[node] = root_input
                        # Stash root_input in node.meta so the subsequent
                        # InPlaceToOutOfPlace pass can name the replacement
                        # node correctly.
                        node.meta["root_input"] = root_input
                elif self._is_aliasing_op(node):
                    source_node = self._get_aliasing_source(node)
                    if source_node is not None:
                        alias_chain[node] = source_node
                        logger.debug(f"Alias: {node.name} -> {source_node.name}")
                elif self._is_custom_op_with_mutations(node):
                    raise RuntimeError(
                        f"Custom op with in-place mutations is not supported: "
                        f"{node.target}. Use wrap_nki() for NKI kernels that "
                        f"mutate their arguments."
                    )
                elif self._is_nki_call_with_mutations(node):
                    logger.debug(f"NKI kernel with mutations detected: {node.name}")
                    nki_kernel_mutated = self._get_nki_kernel_mutated_inputs(
                        node,
                        alias_chain,
                        input_nodes,
                        input_nodes_list,
                        input_names_list,
                    )
                    logger.debug(f"NKI kernel mutated inputs: {nki_kernel_mutated}")
                    for input_index in nki_kernel_mutated:
                        mutated_inputs[input_index] = input_nodes_list[input_index]
                    self._add_nki_output_aliases(gm, node, alias_chain, input_nodes)

            # Find output node
            output_node: Node = next(
                (n for n in reversed(gm.graph.nodes) if n.op == "output"), None
            )
            if output_node is None:
                raise RuntimeError('The graph has no "output" node and is malformed.')

            # FX doesn't see write-aliasing: an NKI HOP takes a placeholder
            # as its write-aliased operand and returns a fresh SSA value,
            # but other consumers of that placeholder still read the bare
            # placeholder with no dataflow edge to the write. Walk the
            # graph once, advancing a per-input "latest write" cursor, and
            # rewire every consumer (later writers and readers alike) to
            # consume the latest write. Serializes parallel writers into a
            # chain and forces reads to see fresh data.
            nki_output_for_input = self._serialize_nki_write_aliases(
                gm, alias_chain, input_nodes, input_nodes_list
            )

            io_map, original_output_count = self._update_outputs_and_build_aliasing(
                mutated_inputs,
                output_node,
                alias_chain,
                input_nodes,
                input_nodes_list,
                nki_output_for_input,
            )

            # Second pass over outputs: detect pass-through and view-alias
            # relationships between explicit outputs and inputs.  These
            # don't involve mutations but still allow buffer reuse via
            # HLO input_output_alias.
            for output_index, output_item in enumerate(output_node.args[0]):
                if not isinstance(output_item, Node):
                    continue

                # Skip slice views — they change shape and cannot alias
                # the full input buffer.
                if self._is_slice_operation(output_item):
                    continue

                if output_item in input_nodes:
                    input_index = input_nodes_list.index(output_item)
                    io_map[output_index] = input_index
                    logger.debug(
                        f"output[{output_index}] is pass-through of "
                        f"input[{input_index}] ({input_names_list[input_index]})"
                    )
                    continue

                root_input = self._find_root_input(
                    output_item, alias_chain, input_nodes
                )
                if root_input is not None and root_input in input_nodes:
                    # HLO aliasing requires the output shape to match the
                    # input shape exactly.  View/reshape ops produce a
                    # different shape and the Neuron compiler will reject
                    # the alias, so skip them.
                    if not self._shapes_match(output_item, root_input):
                        logger.debug(
                            f"output[{output_index}] derives from "
                            f"input[{input_nodes_list.index(root_input)}] "
                            f"but shapes differ — skipping alias"
                        )
                        continue
                    input_index = input_nodes_list.index(root_input)
                    io_map[output_index] = input_index
                    logger.debug(
                        f"output[{output_index}] aliases "
                        f"input[{input_index}] ({input_names_list[input_index]})"
                    )

            gm.recompile()

            return gm, {
                "io_map": io_map,
                "original_output_count": original_output_count,
            }
        except Exception as e:
            raise RuntimeError(f"Aliasing output rewrite failed: {e}") from e

    # =========================================================================
    # NKI write-alias serialization
    # =========================================================================

    def _serialize_nki_write_aliases(
        self,
        gm: torch.fx.GraphModule,
        alias_chain: dict[Node, Node],
        input_nodes: set[Node],
        input_nodes_list: list[Node],
    ) -> dict[int, Node]:
        """Make every consumer of a mutated placeholder see the latest write.

        Addresses NKI HOPs specifically: they declare write-aliasing
        through ``operand_output_aliases`` metadata, but the FX graph
        still shows them reading the placeholder and producing a fresh
        SSA value, with no edge to other consumers of that placeholder.
        In-place torch ops (``add_``, ``copy_``, ``setitem``) don't need
        this — Dynamo's variable tracker rebinds the Python name to the
        mutator's output, so subsequent reads in the FX graph already
        consume the post-mutation value.

        Walks the graph once with a per-input "latest write" cursor. For
        each non-placeholder node, rewires operands pointing at a mutated
        placeholder to consume the cursor, then advances the cursor if
        the node is itself a write. Handles two cases:

        - Writer-vs-writer: the second of two HOPs that both write-alias
          the same placeholder gets its operand rewired to the first
          HOP's output, producing a chain ``k0 → k1 → ... → kN``.
        - Writer-vs-reader: a downstream read of the placeholder gets
          rewired to the most recent write, so the compiler can't
          reorder the read before the write.

        Returns ``{input_idx: latest_write_node}`` for the output rewrite
        step to append as the post-mutation value of each mutated input.
        """
        # Collect writes per input from alias_chain. A "write" is any
        # mutation node (in-place op or NKI HOP output) that resolves to
        # a placeholder. Skip views and slices (they change shape and
        # can't stand in for the post-write value) and skip nodes that
        # are themselves further mutated downstream (the follow-up
        # mutator carries the post-mutation value).
        input_to_writes: dict[int, list[Node]] = {}
        nki_output_for_input: dict[int, Node] = {}
        for alias_node, alias_target in alias_chain.items():
            if alias_target in input_nodes:
                if self._is_slice_operation(alias_node) or self._is_aliasing_op(
                    alias_node
                ):
                    continue
                if any(self._is_torch_mutating_op(u) for u in alias_node.users):
                    continue
                idx = input_nodes_list.index(alias_target)
                nki_output_for_input[idx] = alias_node
                input_to_writes.setdefault(idx, []).append(alias_node)

        # Only inputs where every previous write has the same shape as the
        # placeholder are eligible for replacing subsequent reads/writes to
        # the placeholder
        rewirable_inputs = {
            idx: input_nodes_list[idx]
            for idx, writes in input_to_writes.items()
            if all(self._shapes_match(w, input_nodes_list[idx]) for w in writes)
        }
        if not rewirable_inputs:
            return nki_output_for_input

        # Inverse of input_to_writes
        write_to_input: dict[Node, int] = {
            w: idx
            for idx, ws in input_to_writes.items()
            if idx in rewirable_inputs
            for w in ws
        }

        # Walk the graph in topological order. For each node: rewire any
        # operand that points at a stale placeholder to the latest write,
        # then — if this node is itself a write — advance the cursor so
        # downstream consumers see this node instead.
        current: dict[int, Node] = dict(rewirable_inputs)
        for n in gm.graph.nodes:
            if n.op == "placeholder":
                continue

            # Rewire stale placeholder operands.
            for idx, placeholder in rewirable_inputs.items():
                latest = current[idx]
                if latest is placeholder:
                    continue
                if placeholder in n.all_input_nodes:
                    n.replace_input_with(placeholder, latest)

            # If n is a write, it becomes the new latest for its input.
            owner_idx = write_to_input.get(n)
            if owner_idx is not None:
                current[owner_idx] = n

        return nki_output_for_input

    # =========================================================================
    # Output update and aliasing map construction
    # =========================================================================

    def _update_outputs_and_build_aliasing(
        self,
        mutated_inputs: dict[int, torch.fx.Node],
        output_node: torch.fx.Node,
        alias_chain: dict[Node, Node],
        input_nodes: set[Node],
        input_nodes_list: list[Node],
        nki_output_for_input: dict[int, Node],
    ) -> tuple[dict[int, int], int]:
        """Extend the graph output with newly-added nodes and build the io_map.

        Args:
            mutated_inputs: Mapping of input index to the mutated placeholder node.
            output_node: The graph's output node.
            alias_chain: Mapping from nodes to their alias sources.
            input_nodes: Set of all input placeholder nodes.
            input_nodes_list: Ordered list of input placeholder nodes.
            nki_output_for_input: Mapping from input index to the latest post-mutation value

        Returns:
            Tuple of (``{output_idx: input_idx}`` aliasing map,
            original explicit output count).
        """
        current_output = output_node.args[0]
        if isinstance(current_output, (tuple, list)):
            existing_output_nodes = list(current_output)
        else:
            existing_output_nodes = [current_output]

        explicit_output_count = len(existing_output_nodes)
        logger.debug(f"Explicit outputs: {explicit_output_count}")

        original_len = len(current_output) if isinstance(current_output, tuple) else 1

        if len(existing_output_nodes) > original_len:
            output_node.args = (tuple(existing_output_nodes),)

        aliasing_map = self._add_inplace_aliasing(
            mutated_inputs,
            existing_output_nodes,
            output_node,
            {},
            alias_chain,
            input_nodes,
            input_nodes_list,
            nki_output_for_input,
        )

        return aliasing_map, explicit_output_count

    def _add_inplace_aliasing(
        self,
        mutated_inputs: dict[int, torch.fx.Node],
        existing_output_nodes: list[torch.fx.Node],
        output_node: torch.fx.Node,
        output_to_input_alias_map: dict[int, int],
        alias_chain: dict[Node, Node],
        input_nodes: set[Node],
        input_nodes_list: list[Node],
        nki_output_for_input: dict[int, Node],
    ) -> dict[int, int]:
        """Append in-place-modified tensors to the graph output and record
        their aliasing entries.

        Skips mutated inputs already covered by an existing output (either
        directly present or reachable via the alias chain).

        Args:
            mutated_inputs: Mapping of input index to the mutated placeholder node.
            existing_output_nodes: Current list of output nodes.
            output_node: The graph's output node.
            output_to_input_alias_map: Aliasing map to update in place.
            alias_chain: Mapping from nodes to their alias sources.
            input_nodes: Set of all input placeholder nodes.
            input_nodes_list: Ordered list of input placeholder nodes.
            nki_output_for_input: Mapping from input index to the latest post-mutation value

        Returns:
            Updated ``output_to_input_alias_map``.
        """
        # Determine which mutated inputs are already represented in the
        # existing output tuple (directly or via an alias chain).  These
        # don't need a new output entry.
        covered_input_indices: set[int] = set()
        for out_node in existing_output_nodes:
            if not isinstance(out_node, Node):
                continue
            # Slice views change shape and cannot cover a mutated input.
            if self._is_slice_operation(out_node):
                continue
            if out_node in input_nodes:
                covered_input_indices.add(input_nodes_list.index(out_node))
            else:
                root = self._find_root_input(out_node, alias_chain, input_nodes)
                if root is not None and root in input_nodes:
                    if self._shapes_match(out_node, root):
                        covered_input_indices.add(input_nodes_list.index(root))

        # For each uncovered mutated input, append either the NKI getitem
        # node (preferred, since it carries the post-mutation value) or
        # the raw input placeholder to the output tuple.
        states_to_add = []
        original_output_len = len(existing_output_nodes)
        for input_index, input_node in sorted(mutated_inputs.items()):
            if input_node in existing_output_nodes:
                continue
            if input_index in covered_input_indices:
                continue

            node_to_add = nki_output_for_input.get(input_index, input_node)
            # If the node shape doesn't match the input placeholder shape
            # (e.g. 3D squeezed view vs 4D cache), insert a reshape so
            # HLO aliasing sees matching shapes.
            if node_to_add is not input_node and not self._shapes_match(
                node_to_add, input_node
            ):
                input_shape = input_node.meta.get("example_value")
                if input_shape is not None and hasattr(input_shape, "shape"):
                    target_shape = list(input_shape.shape)
                    with output_node.graph.inserting_before(output_node):
                        reshape_node = output_node.graph.call_method(
                            "reshape", args=(node_to_add, target_shape)
                        )
                        reshape_node.meta["example_value"] = torch.empty(
                            *target_shape,
                            dtype=input_shape.dtype,
                        )
                    node_to_add = reshape_node
            output_index = original_output_len + len(states_to_add)
            states_to_add.append(node_to_add)
            output_to_input_alias_map[output_index] = input_index

        if states_to_add:
            current_output = output_node.args[0]
            if isinstance(current_output, tuple):
                output_node.args = (current_output + tuple(states_to_add),)
            else:
                output_node.args = ((current_output,) + tuple(states_to_add),)

        return output_to_input_alias_map

    # =========================================================================
    # Operation detection
    # =========================================================================

    def _contains_slice(self, arg) -> bool:
        """Check if an argument contains a slice object.

        Args:
            arg: A single argument or a tuple/list of arguments.

        Returns:
            True if *arg* is or contains a slice object.
        """
        return isinstance(arg, slice) or (
            isinstance(arg, (tuple, list)) and any(isinstance(a, slice) for a in arg)
        )

    def _is_slice_operation(self, node: Node) -> bool:
        """Check if a node is a slice/getitem operation that creates a view.

        Args:
            node: The FX graph node to inspect.

        Returns:
            True if the node is a ``getitem`` call with slice arguments.
        """
        if node.op == "call_function" and node.target == operator.getitem:
            return any(self._contains_slice(arg) for arg in node.args)
        return False

    def _is_setitem_operation(self, node: Node) -> bool:
        """Check if a node is a setitem operation (in-place mutation).

        Args:
            node: The FX graph node to inspect.

        Returns:
            True if the node is an ``operator.setitem`` call.
        """
        # TODO: XLA can DCE setitem and this causes issues with aliasing as it maps to
        # a DCE'd input and will fail compile as the map will no longer be accurate.
        return node.op == "call_function" and node.target == operator.setitem
        return False

    def _is_inplace_method(self, node: Node) -> bool:
        """Check if a node is an in-place method call.

        Args:
            node: The FX graph node to inspect.

        Returns:
            True if the node is a ``call_method`` whose target ends with
            ``'_'`` or is in ``INPLACE_METHODS``.
        """
        if node.op != "call_method":
            return False
        target = node.target
        if not isinstance(target, str):
            return False
        return target.endswith("_") and not target.startswith("__")

    def _is_aten_aliasing_op(self, node: Node) -> bool:
        """Check if a node is an ATen aliasing operation.

        When tracing with ``make_fx`` or ``torch.export``, high-level ops
        like ``view``, ``transpose``, etc. are lowered to their ATen
        equivalents.  This method detects those lowered forms.

        Args:
            node: The FX graph node to inspect.

        Returns:
            True if the node is a ``call_function`` targeting an ATen op
            in ``ALIASING_ATEN_OPS``.
        """
        if node.op != "call_function":
            return False

        target = node.target
        # ATen ops from torch.export have __module__ containing "aten".
        # Their __name__ may be overloaded (e.g. "slice.Tensor"), so we
        # strip the overload suffix before checking the set.
        if hasattr(target, "__module__") and "aten" in str(target.__module__):
            op_name = getattr(target, "__name__", str(target))
            base_name = op_name.split(".")[0] if "." in op_name else op_name
            return base_name in ALIASING_ATEN_OPS

        # Fallback for non-ATen targets that may still match by name.
        target_name = getattr(target, "__name__", str(target))
        return target_name in ALIASING_ATEN_OPS

    def _is_method_aliasing_op(self, node: Node) -> bool:
        """Check if a node is a method-style aliasing operation.

        Args:
            node: The FX graph node to inspect.

        Returns:
            True if the node is a ``call_method`` whose target is in
            ``ALIASING_METHODS``.
        """
        if node.op != "call_method":
            return False
        return node.target in ALIASING_METHODS

    def _is_nki_call_with_mutations(self, node: Node) -> bool:
        """Check if a node is an NKI kernel call with output aliases.

        Args:
            node: The FX graph node to inspect.

        Returns:
            True if the node targets an ``NKIKernelWrapper`` with
            non-empty ``operand_output_aliases``.
        """
        from vllm_neuron.nki.nki_hop import NKIKernelWrapper

        if node.op != "call_function" or not isinstance(node.target, NKIKernelWrapper):
            return False
        return bool(node.kwargs.get("operand_output_aliases", {}))

    def _is_aliasing_op(self, node: Node) -> bool:
        """Check if a node creates a view/alias of another tensor.

        Args:
            node: The FX graph node to inspect.

        Returns:
            True if the node is a slice, method-aliasing, or ATen-aliasing op.
        """
        return (
            self._is_slice_operation(node)
            or self._is_method_aliasing_op(node)
            or self._is_aten_aliasing_op(node)
        )

    def _is_torch_mutating_op(self, node: Node) -> bool:
        """Check if a node mutates a tensor in-place.

        Covers three forms:
        - ``operator.setitem``
        - ``call_method`` with trailing ``_`` (e.g. ``add_``, ``copy_``)
        - ``call_function`` targeting an ATen in-place op (e.g.
          ``aten.copy_.default`` from ``torch.export``)

        Args:
            node: The FX graph node to inspect.

        Returns:
            True if the node is a setitem, in-place method, or ATen
            in-place function call.
        """
        if self._is_setitem_operation(node) or self._is_inplace_method(node):
            return True
        # ATen in-place ops traced via torch.export appear as call_function
        # nodes (e.g. aten.copy_.default).  Detect them by name.
        if node.op == "call_function":
            name = getattr(node.target, "__name__", "")
            base = name.split(".")[0] if "." in name else name
            if base.endswith("_") and not base.startswith("__"):
                return True
        return False

    def _is_custom_op_with_mutations(self, node: Node) -> bool:
        """Check if a node is a custom op with mutable arguments.

        Custom ops declare mutable arguments via schema annotations like
        ``Tensor(a!)`` which sets ``alias_info.is_write=True``.

        This method is only reached for nodes that are not already
        handled by ``_is_torch_mutating_op`` (which covers ATen in-place
        ops in both ``call_method`` and ``call_function`` forms).

        Args:
            node: The FX graph node to inspect.

        Returns:
            True if the node has a schema with at least one writable argument.
        """
        if node.op != "call_function":
            return False
        target = node.target
        schema = getattr(target, "_schema", None)
        if schema is None and hasattr(target, "default"):
            schema = getattr(target.default, "_schema", None)
        if schema is None:
            return False
        return any(
            arg.alias_info and arg.alias_info.is_write for arg in schema.arguments
        )

    # =========================================================================
    # Mutation extraction
    # =========================================================================

    def _get_nki_kernel_mutated_inputs(
        self,
        node: Node,
        alias_chain: dict[Node, Node],
        input_nodes: set[Node],
        input_nodes_list: list[Node],
        input_names_list: list[str],
    ) -> list[int]:
        """Return input placeholder indices mutated by an NKI kernel.

        Reads ``node.target.operand_output_aliases`` — a dict of the form
        ``{input_tensor_idx: output_tensor_idx}`` set by
        ``NKIKernelWrapper.__call__``.

        Args:
            node: The NKI kernel node.
            alias_chain: Mapping from nodes to their alias sources.
            input_nodes: Set of all input placeholder nodes.
            input_nodes_list: Ordered list of input placeholder nodes.
            input_names_list: Ordered list of input placeholder names.

        Returns:
            List of input placeholder indices that are mutated.
        """
        assert self._is_nki_call_with_mutations(node)
        operand_output_aliases: dict[int, int] = node.kwargs.get(
            "operand_output_aliases", {}
        )
        kernel_args = node.kwargs.get("args", ())
        mutated: list[int] = []

        for input_tensor_idx, output_tensor_idx in operand_output_aliases.items():
            if input_tensor_idx >= len(kernel_args):
                continue

            kernel_arg_node = kernel_args[input_tensor_idx]
            if not isinstance(kernel_arg_node, Node):
                continue

            root_input = self._find_root_input(
                kernel_arg_node, alias_chain, input_nodes
            )
            if root_input is not None and root_input in input_nodes:
                placeholder_idx = input_nodes_list.index(root_input)
                if placeholder_idx not in mutated:
                    mutated.append(placeholder_idx)
                    logger.debug(
                        f"NKI kernel mutation detected: input[{placeholder_idx}] "
                        f"({input_names_list[placeholder_idx]}) via "
                        f"operand_output_aliases[{input_tensor_idx}] = {output_tensor_idx}"
                    )

        return mutated

    def _add_nki_output_aliases(
        self,
        gm: torch.fx.GraphModule,
        node: Node,
        alias_chain: dict[Node, Node],
        input_nodes: set[Node],
    ):
        """Register NKI kernel getitem outputs in ``alias_chain``.

        For each ``operand_output_aliases`` entry ``{input_idx: output_idx}``,
        find (or create) the corresponding ``getitem(kernel, output_idx)``
        node and map it back to the graph-level placeholder that the kernel
        mutates.

        Args:
            gm: The FX GraphModule to insert nodes into.
            node: The NKI kernel node.
            alias_chain: Mapping from nodes to their alias sources
                (mutated in place).
            input_nodes: Set of all input placeholder nodes.
        """
        operand_output_aliases: dict[int, int] = node.kwargs.get(
            "operand_output_aliases", {}
        )
        kernel_args = node.kwargs.get("args", ())

        for input_tensor_idx, output_tensor_idx in operand_output_aliases.items():
            if input_tensor_idx >= len(kernel_args):
                continue
            kernel_arg_node = kernel_args[input_tensor_idx]
            if not isinstance(kernel_arg_node, Node):
                continue
            root_input = self._find_root_input(
                kernel_arg_node, alias_chain, input_nodes
            )
            if root_input is None or root_input not in input_nodes:
                continue

            # For tuple-returning kernels, the aliased output is a
            # specific element — find or create the getitem node for it.
            # For single-tensor-returning kernels, the kernel node
            # itself is the output reference.
            if self._kernel_returns_tuple(node):
                output_node_ref = self._find_or_create_getitem(
                    gm, node, output_tensor_idx
                )
            else:
                output_node_ref = node

            alias_chain[output_node_ref] = root_input
            node.meta["root_input"] = output_node_ref

            logger.debug(
                f"NKI output alias: {output_node_ref.name} -> {root_input.name}"
            )

    def _find_or_create_getitem(
        self, gm: torch.fx.GraphModule, kernel_node: torch.fx.Node, output_idx: int
    ) -> torch.fx.Node:
        """Return an existing ``getitem`` node or create one after *kernel_node*.

        Args:
            gm: The FX GraphModule to insert nodes into.
            kernel_node: The kernel node whose output is indexed.
            output_idx: The tuple index to extract.

        Returns:
            The ``getitem`` FX Node.
        """
        for node in gm.graph.nodes:
            if (
                node.op == "call_function"
                and node.target == operator.getitem
                and len(node.args) >= 2
                and node.args[0] == kernel_node
                and node.args[1] == output_idx
            ):
                return node
        with gm.graph.inserting_after(kernel_node):
            getitem = gm.graph.call_function(
                operator.getitem, args=(kernel_node, output_idx)
            )
        # Propagate example_value from the kernel's output tuple so downstream
        # shape checks (e.g. _shapes_match in _serialize_nki_write_aliases)
        # can reason about this synthesized node. Without it, the getitem has
        # no shape metadata and write-alias rewiring is silently skipped —
        # leaving recurrent reads (e.g. Eagle3 fused propose) pointing at the
        # stale placeholder instead of the latest in-place write.
        kernel_val = (
            kernel_node.meta.get("example_value")
            if hasattr(kernel_node, "meta")
            else None
        )
        if isinstance(kernel_val, (tuple, list)) and output_idx < len(kernel_val):
            getitem.meta["example_value"] = kernel_val[output_idx]
        return getitem

    def _kernel_returns_tuple(self, kernel_node: torch.fx.Node) -> bool:
        """Return True if the kernel node returns a tuple of tensors.

        Checks ``example_value`` metadata first; falls back to looking
        for a downstream ``getitem`` node.

        Args:
            kernel_node: The kernel node to inspect.

        Returns:
            True if the kernel output is a tuple or list.
        """
        if hasattr(kernel_node, "meta") and "example_value" in kernel_node.meta:
            return isinstance(kernel_node.meta["example_value"], (tuple, list))

        for node in kernel_node.graph.nodes:
            if (
                node.op == "call_function"
                and node.target == operator.getitem
                and len(node.args) >= 2
                and node.args[0] is kernel_node
            ):
                return True
        return False

    # =========================================================================
    # Graph traversal helpers
    # =========================================================================

    def _get_torch_op_mutated_tensor(self, node: Node) -> Node | None:
        """Return the tensor being mutated by a setitem or in-place method call.

        Args:
            node: The mutating FX graph node.

        Returns:
            The first argument node (the mutated tensor), or None if the
            node has no arguments or the first argument is not a Node.
        """
        if not node.args:
            return None
        if self._is_setitem_operation(node) or self._is_inplace_method(node):
            return node.args[0] if isinstance(node.args[0], Node) else None
        return None

    def _extract_placeholder_name(self, node: Node) -> str | None:
        """Convert Dynamo's mangled placeholder target names to human-readable names.

        Handles these Dynamo naming patterns:

        - Regular inputs:  ``L_<name>_`` → ``<name>``
        - Parameters:      ``L_self_modules_<path>_parameters_<name>_`` →
          ``<path>.<name>``
        - Buffers:         ``L_self_buffers_<name>_`` → ``<name>``

        Args:
            node: A placeholder node.

        Returns:
            The cleaned-up name, or the raw node name as a fallback.
        """
        target = node.target

        if not isinstance(target, str):
            return node.name

        if "parameters" in target:
            name = target.replace("L_self_modules_", "").replace("_parameters_", ".")
            return name.rstrip("_")

        if "buffers" in target:
            return target.replace("L_self_buffers_", "").rstrip("_")

        if target.startswith("L_") and "self" not in target:
            return target[2:].rstrip("_")

        return target.rstrip("_") if target else node.name

    def _get_aliasing_source(self, node: Node) -> Node | None:
        """Return the source tensor for an aliasing operation.

        Args:
            node: An aliasing FX graph node.

        Returns:
            The first argument node (the aliased tensor), or None if the
            node has no arguments or the first argument is not a Node.
        """
        if not node.args:
            return None
        first_arg = node.args[0]
        return first_arg if isinstance(first_arg, Node) else None

    def _shapes_match(self, a: Node, b: Node) -> bool:
        """Return True if two nodes have the same tensor shape.

        Uses ``example_value`` metadata attached by Dynamo.  If metadata
        is missing for either node, conservatively returns False so that
        the alias is not attempted.

        Args:
            a: First FX graph node.
            b: Second FX graph node.

        Returns:
            True if both shapes are equal, False if shape info is unavailable.
        """
        a_val = a.meta.get("example_value") if hasattr(a, "meta") else None
        b_val = b.meta.get("example_value") if hasattr(b, "meta") else None
        if a_val is None or b_val is None:
            return False
        if not hasattr(a_val, "shape") or not hasattr(b_val, "shape"):
            return False
        return a_val.shape == b_val.shape

    def _find_root_input(
        self,
        node: Node,
        alias_chain: dict[Node, Node],
        input_nodes: set[Node],
    ) -> Node | None:
        """Trace a node back through the alias chain to its root input placeholder.

        Args:
            node: The starting node to trace.
            alias_chain: Mapping from nodes to their alias sources.
            input_nodes: Set of all input placeholder nodes.

        Returns:
            The root input placeholder node, or None if the node doesn't
            derive from any input.
        """
        if node in input_nodes:
            return node

        current = node
        # Guard against cycles (shouldn't happen in a well-formed graph,
        # but protects against infinite loops during development).
        visited: set[Node] = set()

        while current is not None and current not in visited:
            visited.add(current)

            if current in input_nodes:
                return current

            # Two resolution strategies:
            # 1. Explicit alias_chain entry (set earlier in this pass
            #    for mutations, NKI outputs, and custom ops).
            # 2. Implicit aliasing via view/reshape ops — walk through
            #    the first argument which is the source tensor.
            if current in alias_chain:
                current = alias_chain[current]
                continue

            if self._is_aliasing_op(current):
                current = self._get_aliasing_source(current)
            else:
                return None

        return None
