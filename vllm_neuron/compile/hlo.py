# SPDX-License-Identifier: Apache-2.0
import os
import logging
import re
from typing import Dict

import torch
import torch_xla
from torch_xla.core import xla_builder
import torch_neuronx
from torch_neuronx.pyhlo import hlo_pb2, xla_data_pb2
from torch_neuronx.xla_impl import structure
from torch_neuronx.xla_impl.xla_hlo_tools.xla_primitive_enum_utils import (
    XlaPrimitiveProperties,
)

logger = logging.getLogger(__name__)


def load_hlo_module(hlo_path: str) -> hlo_pb2.HloModuleProto:
    """Load the final processed HLO module from cache.

    Args:
        hlo_path: Path to the HLO file

    Returns:
        hlo_pb2.HloModuleProto: The loaded HLO module

    Raises:
        FileNotFoundError: If the HLO file is not found
    """
    if not os.path.exists(hlo_path):
        raise FileNotFoundError(f"Cached HLO file not found: {hlo_path}")

    with open(hlo_path, "rb") as f:
        hlo_data = f.read()

    hlo_module = hlo_pb2.HloModuleProto()
    hlo_module.ParseFromString(hlo_data)
    return hlo_module


def extract_aliasing_from_hlo(hlo_module: hlo_pb2.HloModuleProto) -> Dict[int, int]:
    """Extract input-output aliasing map from HLO module.

    Args:
        hlo_module: The HLO module to extract aliasing from

    Returns:
        dict: Aliasing map from output index to input index
    """
    io_map = {}
    if (
        hasattr(hlo_module, "input_output_alias")
        and hlo_module.input_output_alias.entries
    ):
        for entry in hlo_module.input_output_alias.entries:
            if entry.output_shape_index:
                output_idx = entry.output_shape_index[0]
                input_idx = entry.parameter_number
                io_map[output_idx] = input_idx
    return io_map


# top frame of an fx node.stack_trace: '  File "model.py", line 123, in forward'
_FRAME_RE = re.compile(r'File "([^"]+)", line (\d+)')


def _remap_hlo_source_to_model(hlo_module, gm):
    """Rewrite HLO op source metadata from the Dynamo generated forward
    (``<eval_with_key>.N``) back to the real model ``file:line``.

    torch_xla assigns each op's ``source_file``/``source_line`` from the live
    Python frame during lowering, which is the generated forward. We use that
    ``source_line`` to look up the originating FX node (via FX's generated
    line->node map) and overwrite the metadata with the node's recorded
    ``stack_trace`` (the real model source). Best effort: ops with no
    resolvable node (e.g. nodes synthesized by FX passes) are left unchanged.
    """
    try:
        line_to_node = getattr(gm.graph.python_code("self"), "_lineno_map", None)
        nodes = list(gm.graph.nodes)
        if not line_to_node:
            return
    except Exception:
        return

    def real_loc(gen_line):
        # _lineno_map keys are lines into the generated forward; tolerate a
        # small preamble offset between FX's numbering and HLO source_line.
        idx = line_to_node.get(gen_line, line_to_node.get(gen_line - 1))
        if idx is None or idx >= len(nodes):
            return None
        st = nodes[idx].stack_trace
        if not st:
            return None
        frames = _FRAME_RE.findall(st)
        return (frames[-1][0], int(frames[-1][1])) if frames else None

    for comp in hlo_module.computations:
        for instr in comp.instructions:
            md = instr.metadata
            if md.source_file and "<eval_with_key>" in md.source_file:
                loc = real_loc(md.source_line)
                if loc:
                    md.source_file, md.source_line = loc

    # Also rewrite the structured stack_frame_index. neuronx-cc propagates this
    # (via OpMetadata.stack_frame_id) into the BIR/Penguin SourceFiles that back
    # nki_source_location / bir_debug_info_source_location and the framework
    # StackFrameFileName table; the flat source_file rewrite above does not cover it.
    sfi = hlo_module.stack_frame_index
    if sfi.file_names:
        name_to_id = {
            n: i + 1 for i, n in enumerate(sfi.file_names)
        }  # XLA ids are 1-based

        def file_id(name):
            fid = name_to_id.get(name)
            if fid is None:
                sfi.file_names.append(name)
                fid = len(sfi.file_names)
                name_to_id[name] = fid
            return fid

        for fl in sfi.file_locations:
            fname = (
                sfi.file_names[fl.file_name_id - 1]
                if 1 <= fl.file_name_id <= len(sfi.file_names)
                else ""
            )
            if "<eval_with_key>" in fname:
                loc = real_loc(fl.line)  # fl.line is the generated-forward line
                if loc:
                    fl.file_name_id = file_id(loc[0])
                    fl.line = loc[1]


def convert_fx_to_hlo(gm, example_inputs, log_path, aliasing_map=None):
    """Convert FX graph to HLO with full processing pipeline.

    Args:
        gm: The PyTorch FX GraphModule to convert
        example_inputs: Example inputs for tracing
        log_path: Path for logging HLO passes
        aliasing_map: Input-output aliasing map

    Returns:
        hlo_pb2.HloModuleProto: The processed HLO module
    """
    os.environ["PJRT_DEVICE"] = "CPU"

    # clean xla graph before creating placeholders
    torch_xla.sync(wait=True, reset_scope=True)
    xla_device = torch_xla.core.xla_model.xla_device()
    torch_xla._XLAC._clear_pending_irs(f"{xla_device.type}:{xla_device.index}")

    # create_placeholder_tensor loses unsigned dtypes (uint16 → int16).
    # Use BitcastConvert + .to() to restore the correct HLO type and PyTorch
    # dtype.
    from torch_neuronx.xla_impl.hlo_conversion import PlaceholderMixin

    xla_placeholders = []
    for tensor in example_inputs:
        placeholder = xla_builder.create_placeholder_tensor(tensor.shape, tensor.dtype)
        if placeholder.dtype != tensor.dtype:
            bitcast_op = PlaceholderMixin._BITCAST_OPS.get(tensor.dtype)
            if bitcast_op is not None:
                placeholder = bitcast_op(placeholder).to(tensor.dtype)
                torch_xla._XLAC._clear_pending_irs(
                    f"{xla_device.type}:{xla_device.index}"
                )
        xla_placeholders.append(placeholder)

    with torch_neuronx.contexts.lowering():
        outputs = gm(*xla_placeholders)

    layout, uniques, constants = structure.extract(outputs)
    if not uniques:
        raise ValueError("Cannot compile a module that has no output")
    tensors, identifiers = zip(*uniques.items(), strict=True)

    # Lower the HLO graph
    context = torch_xla._XLAC.lowering.LoweringContext()
    context.build(tensors)

    hlo = context.hlo()
    hlo_module = hlo_pb2.HloModuleProto()
    hlo_module.ParseFromString(hlo)

    # torch_xla stamps op source from the live frame (the Dynamo generated
    # forward, <eval_with_key>.N). Remap back to the real model source using the
    # FX node stack traces so profilers show useful locations.
    _remap_hlo_source_to_model(hlo_module, gm)

    _log_hlo(hlo_module, log_path + "step1_torch_xla_trace.hlo")

    # Consider example:
    # Say there are 5 parameters in the HLO and the tensors are
    # parameter.keys = dict_keys([4, 3, 2, 1, 0])
    # input_param_ids = [4, 3]
    # constant_param_ids = [2, 1, 0]
    #
    # This means there are 3 inputs (2, 1, 0) which are constants and (4, 3)
    # are the example_inputs.
    #   example_input[0] is the 4th input to the graph
    #   example_input[1] is the 3rd input to the graph
    parameters = context.device_parameter_id_tensor_mapping()
    input_param_ids = [
        context.tensor_parameter_id(tensor) for tensor in xla_placeholders
    ]
    constant_param_ids = list(set(parameters.keys()) - set(input_param_ids))

    # Track which input indices are unused (DCE'd by XLA, indicated by -1)
    # These inputs won't have corresponding HLO parameters
    unused_input_indices = [i for i, pid in enumerate(input_param_ids) if pid == -1]
    if unused_input_indices:
        logger.debug(
            f"[convert_to_hlo] Unused input indices (DCE'd by XLA): {unused_input_indices}"
        )

    # Non input parameters are constants inlined to the HLO.
    # This produces a new HLO with just 2 inputs which are (4, 3)
    constants: dict[int, torch.Tensor] = {
        id: parameters[id] for id in constant_param_ids
    }
    hlo_module = _inline_constants_to_hlo(hlo_module, constants)

    # with open("/home/ubuntu/vllm_neuron/src/VllmNeuronBackend/hlo_filename.hlo", "wb") as f:
    #     f.write(hlo_module.SerializeToString())
    _log_hlo(hlo_module, log_path + "step2_inline_constants_pass.hlo")

    # XLA can record input parameters in any order. Make the HLO Entry input parameter
    # ordering match that of what torch.compile gives us.
    #
    # Now HLO is Entry(example_input[1], example_input[0])
    # We reorder to  Entry(example_input[0], example_input[1])
    inputs = [pid for pid in sorted(parameters.keys()) if pid not in constants]
    order = {}
    has_rng_seed_param = False

    # When XLA DCEs unused inputs, we need to normalize positions to avoid out-of-index param
    positions = []
    for i, pid in enumerate(inputs):
        try:
            original_pos = input_param_ids.index(pid)
            positions.append((original_pos, i))
        except ValueError:
            # RNG seed is an input to HLO but not input to FX, move it to last position.
            positions.append((len(input_param_ids), i))
            has_rng_seed_param = True

    # Sort by original position and build order dict with continuous dst values
    positions.sort()
    for new_pos, (_, old_pos) in enumerate(positions):
        order[old_pos] = new_pos

    hlo_module = _match_input_order(hlo_module, order)

    _log_hlo(hlo_module, log_path + "step3_match_input_order_pass.hlo")

    # Adjust aliasing indices for DCE'd inputs. XLA removes unused inputs,
    # shifting parameter numbers down. Subtract preceding DCE'd count.
    unused_set = set(unused_input_indices)
    io_map = {}
    if aliasing_map:
        for output_idx, input_idx in aliasing_map.items():
            if input_idx >= len(xla_placeholders):
                raise ValueError(
                    f"Input index {input_idx} is out of range for {len(xla_placeholders)} inputs"
                )
            if input_idx in unused_set:
                logger.warning(
                    "Aliasing map references unused (DCE'd) input %d for output %d — skipping",
                    input_idx,
                    output_idx,
                )
                continue
            num_unused_before = sum(1 for u in unused_input_indices if u < input_idx)
            hlo_param_num = input_idx - num_unused_before
            io_map[output_idx] = hlo_param_num

    hlo_module = _add_aliasing_info(hlo_module, io_map)
    _log_hlo(hlo_module, log_path + "step4_add_aliasing_pass.hlo")

    if has_rng_seed_param:
        hlo_module = replace_rng_bit_generator(hlo_module)
        _log_hlo(hlo_module, log_path + "step5_replace_rng_pass.hlo")

    return hlo_module, unused_input_indices, has_rng_seed_param


def log_hlo_debug(hlo_module, compiler_workdir: str, options: dict):
    """Log HLO debug information if enabled.

    Args:
        hlo_module: The HLO module to log
        compiler_workdir: Working directory for debug files
        options: Compilation options containing debug_hlo flag
    """
    debug_hlo = options.get("debug_hlo", False)
    if debug_hlo:
        # FIXME: Write a clean way to log HLOs across multiple passes
        # Just log the last final state of the HLO before compilation.
        # This code is not clean.
        hlo_debug_path = os.path.join(compiler_workdir, "graph_debug.hlo")
        _log_hlo(hlo_module, hlo_debug_path)
        logger.info(f"Traced HLO -  {hlo_debug_path}.txt")


def _add_aliasing_info(hlo_module, updated_input_output_aliases):
    """Add aliasing information to HLO module.

    Args:
        hlo_module: The HLO module to modify
        updated_input_output_aliases: Dict mapping output indices to input indices

    Returns:
        hlo_pb2.HloModuleProto: The modified HLO module
    """
    if len(updated_input_output_aliases) == 0:
        return hlo_module

    io_alias_proto = hlo_pb2.HloInputOutputAliasProto()

    for output_idx, input_idx in updated_input_output_aliases.items():
        alias_entry_proto = io_alias_proto.AliasEntryProto()

        alias_entry_proto.output_shape_index.append(output_idx)
        alias_entry_proto.parameter_number = input_idx
        alias_entry_proto.kind = hlo_pb2.Kind.MUST_ALIAS

        io_alias_proto.entries.append(alias_entry_proto)

    hlo_module.input_output_alias.CopyFrom(io_alias_proto)
    return hlo_module


def _match_input_order(hlo_module, order):
    """Reorder the HLO Entry inputs to match the order of the inputs.

    Args:
        hlo_module: The HLO module to modify
        order: Dict mapping new positions to old positions

    Returns:
        hlo_pb2.HloModuleProto: The modified HLO module
    """
    id_to_computation = {cpt.id: cpt for cpt in hlo_module.computations}
    entry_computation = id_to_computation[hlo_module.entry_computation_id]
    entry_instructions = entry_computation.instructions

    param_insts_mapping = {}
    for instruction in entry_instructions:
        if instruction.opcode == OpCode.parameter:
            param_insts_mapping[instruction.parameter_number] = instruction

    original_shape = hlo_module.host_program_shape

    program_shape = xla_data_pb2.ProgramShapeProto()
    program_shape.CopyFrom(original_shape)

    for src, dst in order.items():
        program_shape.parameters[dst].CopyFrom(original_shape.parameters[src])
        program_shape.parameter_names[dst] = param_insts_mapping[src].name
        param_insts_mapping[src].parameter_number = dst

    # Update both hlo module and root computation program shapes
    hlo_module.host_program_shape.CopyFrom(program_shape)
    entry_computation.program_shape.CopyFrom(program_shape)

    return hlo_module


def _inline_constants_to_hlo(hlo_module, constants):
    """Change the HLO to inline the constants.

    Args:
        hlo_module: The HLO module to modify
        constants: Dict mapping parameter indices to constant tensors

    Returns:
        hlo_pb2.HloModuleProto: The modified HLO module
    """
    id_to_computation = {cpt.id: cpt for cpt in hlo_module.computations}
    entry_computation = id_to_computation[hlo_module.entry_computation_id]
    entry_instructions = entry_computation.instructions

    def _param_insts():
        # Create a mapping from parameter number to the HLO instruction
        param_insts_mapping = {}
        id_to_insts_mapping = {}
        rng_seed_params = set()

        def _trace_to_parameter(instr, visited):
            visited.add(instr.id)
            if instr.opcode == OpCode.parameter:
                return instr

            for op_id in instr.operand_ids:
                if op_id in visited:
                    continue
                instr = id_to_insts_mapping.get(op_id)
                instr = instr and _trace_to_parameter(instr, visited)
                if instr is not None:
                    return instr

        for instruction in entry_instructions:
            id_to_insts_mapping[instruction.id] = instruction
            if instruction.opcode == OpCode.parameter:
                param_insts_mapping[instruction.parameter_number] = instruction
            elif instruction.opcode == OpCode.rng_bit_generator:
                rng_seed_param_inst = _trace_to_parameter(instruction, set())
                rng_seed_params.add(rng_seed_param_inst.parameter_number)

        return param_insts_mapping, rng_seed_params

    param_insts, rng_seed_params = _param_insts()
    for idx in rng_seed_params:
        del constants[idx]

    for idx, const_tensor in constants.items():
        param_inst = param_insts[idx]

        # clear/overwrite existing fields to constant specific op semantics
        param_inst.ClearField("parameter_number")
        param_inst.opcode = "constant"
        param_inst.name = f"xlaconst{idx}"

        # create literal proto message that contains the actual tensor data
        literal = xla_data_pb2.LiteralProto()
        literal.shape.CopyFrom(param_inst.shape)
        element_type = literal.shape.element_type

        const_tensor = const_tensor.cpu()
        # modify tensor to match layout defined in shape
        # torch indices are major to minor, but hlo is minor to major
        layout = list(reversed(literal.shape.layout.minor_to_major))
        if layout:
            const_tensor = const_tensor.permute(*layout)

        if XlaPrimitiveProperties.is_encoded_as_bytes(element_type):
            # num bytes is either 1 or 2
            torch_view_repr = (
                torch.int8
                if XlaPrimitiveProperties.get_num_bytes(element_type) == 1
                else torch.int16
            )
            const_tensor = const_tensor.view(torch_view_repr)
        flattened_np = const_tensor.numpy().flatten()
        if XlaPrimitiveProperties.is_encoded_as_bytes(element_type):
            # encoded as single bytestring
            byte_size = XlaPrimitiveProperties.get_num_bytes(element_type)
            flattened_np = flattened_np.view(f"V{byte_size * flattened_np.shape[0]}")
        for scalar in flattened_np:
            # sets the appropriate field in literal proto
            XlaPrimitiveProperties.apply_literal_modifier(
                element_type, (literal, scalar.item())
            )

        param_inst.literal.CopyFrom(literal)

    # modify shape proto to remove the converted parameters
    # as well as modify the constant_parameter_tensors and
    # linearized_param_num_to_module_path mappings respectively
    entry_computation_program_shape = entry_computation.program_shape
    new_program_shape = xla_data_pb2.ProgramShapeProto()
    new_program_shape.result.CopyFrom(entry_computation_program_shape.result)

    offset = 0
    for i in range(len(entry_computation_program_shape.parameter_names)):
        if i not in constants:
            new_program_shape.parameters.append(
                entry_computation_program_shape.parameters[i]
            )

            new_name = f"p{i - offset}"
            new_program_shape.parameter_names.append(new_name)
            param_insts[i].name = new_name
            param_insts[i].parameter_number = i - offset
        else:  # this is a converted parameter, we shouldn't add it
            offset += 1

    entry_computation.program_shape.CopyFrom(new_program_shape)
    hlo_module.host_program_shape.CopyFrom(new_program_shape)

    return hlo_module


def _log_hlo(hlo_module, path) -> None:
    """Log HLO module to file.

    Args:
        hlo_module: The HLO module to log
        path: Path to save the HLO file
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(hlo_module.SerializeToString())


class OpCode:
    # https://github.com/tensorflow/tensorflow/blob/v2.8.0/tensorflow/compiler/xla/service/hlo_opcode.h
    constant = "constant"
    parameter = "parameter"
    custom_call = "custom-call"
    rng_bit_generator = "rng-bit-generator"
    get_tuple_element = "get-tuple-element"
    convert = "convert"
    multiply = "multiply"
    rng = "rng"


def replace_rng_bit_generator(
    hlo_module: hlo_pb2.HloModuleProto,
) -> hlo_pb2.HloModuleProto:
    """Replace rng-bit-generator with simple rng op for Neuron compatibility.

    torch_xla generates rng-bit-generator for torch.rand(), which uses u64 types
    that have issues on Neuron hardware. This function replaces the entire RNG
    chain with a simple stateless rng op that produces uniform [0,1) values directly.

    Args:
        hlo_module: The HLO module to modify

    Returns:
        Modified HLO module with rng-bit-generator replaced by rng

    Raises:
        RuntimeError: If rng-bit-generator is found but the expected conversion
            chain (convert, multiply) cannot be located, indicating XLA's
            decomposition pattern may have changed.

    Example:
        >>> hlo_module = convert_fx_to_hlo(gm, inputs, log_path)
        >>> hlo_module = replace_rng_bit_generator(hlo_module)
    """
    # Find entry computation by ID
    entry = None
    for comp in hlo_module.computations:
        if comp.id == hlo_module.entry_computation_id:
            entry = comp
            break
    if entry is None:
        return hlo_module

    instructions = entry.instructions

    # Step 1: Find rng-bit-generator
    rng_bit_gen = None
    for inst in instructions:
        if inst.opcode == OpCode.rng_bit_generator:
            rng_bit_gen = inst
            break

    if rng_bit_gen is None:
        return hlo_module  # No rng-bit-generator, nothing to do

    # Step 2: Trace forward to find convert (u32 -> f32) and final multiply
    def find_users(inst_id):
        return [inst for inst in instructions if inst_id in inst.operand_ids]

    convert_inst = None
    chain_ids = {rng_bit_gen.id}
    queue = [rng_bit_gen.id]

    # Find convert instruction
    while queue and convert_inst is None:
        current_id = queue.pop(0)
        for user in find_users(current_id):
            if user.id in chain_ids:
                continue
            chain_ids.add(user.id)
            if user.opcode == OpCode.convert:
                convert_inst = user
                break
            queue.append(user.id)

    if convert_inst is None:
        raise RuntimeError(
            "Found rng-bit-generator but could not find convert instruction in chain. "
            "XLA's torch.rand decomposition may have changed."
        )

    # Step 3: From convert, trace through multiplies to find final one
    final_multiply = None
    queue = [convert_inst.id]

    while queue:
        current_id = queue.pop(0)
        for user in find_users(current_id):
            if user.id in chain_ids:
                continue
            if user.opcode == OpCode.multiply:
                chain_ids.add(user.id)
                final_multiply = user
                queue.append(user.id)

    if final_multiply is None:
        raise RuntimeError(
            "Found rng-bit-generator and convert but could not find multiply in chain. "
            "XLA's torch.rand decomposition may have changed."
        )

    # Step 4: Create constants for rng
    max_id = max(inst.id for inst in instructions)

    # Create proper scalar f32 shape with layout
    def make_scalar_f32_shape():
        shape = xla_data_pb2.ShapeProto()
        shape.element_type = 11  # F32
        # Scalar: no dimensions, empty minor_to_major layout
        shape.layout.minor_to_major.extend([])
        return shape

    scalar_shape = make_scalar_f32_shape()

    const0 = hlo_pb2.HloInstructionProto()
    const0.id = max_id + 1
    const0.name = "rng_const_min"
    const0.opcode = OpCode.constant
    const0.shape.CopyFrom(scalar_shape)
    const0.literal.shape.CopyFrom(scalar_shape)
    const0.literal.f32s.append(0.0)

    const1 = hlo_pb2.HloInstructionProto()
    const1.id = max_id + 2
    const1.name = "rng_const_max"
    const1.opcode = OpCode.constant
    const1.shape.CopyFrom(scalar_shape)
    const1.literal.shape.CopyFrom(scalar_shape)
    const1.literal.f32s.append(1.0)

    # Step 5: Repurpose final_multiply as rng (keeps ID, consumers stay valid)
    chain_ids.remove(final_multiply.id)
    final_multiply.opcode = OpCode.rng
    final_multiply.distribution = 1  # RNG_UNIFORM
    del final_multiply.operand_ids[:]
    final_multiply.operand_ids.extend([const0.id, const1.id])

    # Step 6: Delete chain, add constants at beginning (before rng that uses them)
    new_instructions = [const0, const1]
    new_instructions.extend([inst for inst in instructions if inst.id not in chain_ids])

    del entry.instructions[:]
    entry.instructions.extend(new_instructions)

    return hlo_module
