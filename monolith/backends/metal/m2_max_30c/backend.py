import copy
import struct

from ..base import MetalBackend


class Backend(MetalBackend):
    """Native shader path for the 30-core M2 Max."""
    id = "m2_max_30c"
    # ICB speculative replay intermittently hangs on the qualified Apple8 device,
    # including with every command barriered. Direct serial encoding passes the
    # same workloads with shader validation; keep ICB replay an explicit opt-in.
    reencode_default = True

    def gemm_tile_shape(self, info, tm):
        # Smaller operand/destination tiles avoid the slow large-register
        # configurations measured on Apple8. Scaled formats need TK/8 >= 16.
        return (16, 128) if info.scale_bytes else (16, 64)

    def finalize(self, program, graph, emitted, config, **options):
        replaced = set()
        for op in program.ops:
            kernel = program.kernels[op.kernel]
            m = kernel.macros
            if (kernel.function != "gdn_mixer" or m.get("LOCAL_PREPARE") != "1"
                    or m.get("SINGLE_PASS") != "1" or m.get("SPB") != "1u"
                    or m.get("DK") != "128" or m.get("DV") != "128"
                    or m.get("SL") not in ("2u", "4u") or op.threadgroup[0] <= 256):
                continue
            # Apple8's compiled limits are below the shared 1024-thread groups:
            # 896 for fused/permuted T=1 (384 with validation), 512 for T=8
            # with validation. Keep preparation local in 256-thread groups.
            name, offset = next((name, offset) for slot, name, offset in op.bindings if slot == 9)
            data = bytearray(program.buffers[name].init)
            heads = struct.unpack_from("<I", data, offset)[0]
            groups = 8
            fused = m.get("FUSED_NORM") == "1"
            slices = 16 if fused else int(m["SL"].rstrip("u"))
            n_sg = heads * (128 // slices)
            if fused:
                # Norm needs every state column of a head in the same group.
                struct.pack_into("<I", data, offset + 13 * 4, n_sg)
                program.buffers[name].init = bytes(data)
            key = op.kernel + ".m2_groups8"
            if key not in program.kernels:
                tuned = copy.deepcopy(kernel)
                tuned.macros.update(SL=f"{slices}u", LOCAL_GROUPS="8u")
                if "STATIC_GDN_P_N_SG" in tuned.macros:
                    tuned.macros["STATIC_GDN_P_N_SG"] = f"{n_sg}u"
                program.kernels[key] = tuned
            replaced.add(op.kernel)
            op.kernel = key
            op.grid = (n_sg // groups, 1, 1)
            op.threadgroup = (32 * groups, 1, 1)
        used = {op.kernel for op in program.ops}
        for key in replaced - used:
            del program.kernels[key]
        return program
