
import struct, os, math

OUT = 'fixtures/tiny-ds41/tiny-ds41-00001-of-00001.gguf'
os.makedirs(os.path.dirname(OUT), exist_ok=True)

ALIGN = 32

def s(v):
    b = v.encode('utf-8')
    return struct.pack('<Q', len(b)) + b

TYPE_CODES = {'u8': 0, 'i8': 1, 'u16': 2, 'i16': 3, 'u32': 4, 'i32': 5, 'f32': 6, 'bool': 7, 'str': 8, 'u64': 10, 'i64': 11, 'f64': 12}

def scalar(t, v):
    if t == 'str':
        return s(v)
    if t == 'u32':
        return struct.pack('<I', v)
    if t == 'i32':
        return struct.pack('<i', v)
    if t == 'u64':
        return struct.pack('<Q', v)
    if t == 'f32':
        return struct.pack('<f', v)
    if t == 'bool':
        return b'\x01' if v else b'\x00'
    raise ValueError(t)

def val(t, v):
    if isinstance(t, tuple) and t[0] == 'arr':
        et, items = t[1], v
        out = struct.pack('<I', 9) + struct.pack('<I', TYPE_CODES[et]) + struct.pack('<Q', len(items))
        for item in items:
            out += scalar(et, item)
        return out
    return struct.pack('<I', TYPE_CODES[t]) + scalar(t, v)

# byte-level alphabet, same rule as GPT-2 / RustMoE
bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
cs = bs[:]
n = 0
for b in range(256):
    if b not in bs:
        bs.append(b)
        cs.append(256 + n)
        n += 1
chars = [chr(c) for c in cs]
assert len(set(chars)) == 256

meta = [
    ('general.architecture', 'str', 'deepseek41'),
    ('general.name', 'str', 'RustMoE Tiny DS41 Fixture'),
    ('general.type', 'str', 'model'),
    ('general.alignment', 'u32', ALIGN),
    ('general.file_type', 'u32', 0),
    ('deepseek41.block_count', 'u32', 1),
    ('deepseek41.context_length', 'u32', 256),
    ('deepseek41.embedding_length', 'u32', 64),
    ('deepseek41.embedding_length_out', 'u32', 64),
    ('deepseek41.attention.head_count', 'u32', 2),
    ('deepseek41.attention.head_count_kv', 'u32', 1),
    ('deepseek41.attention.key_length', 'u32', 8),
    ('deepseek41.attention.value_length', 'u32', 8),
    ('deepseek41.attention.q_lora_rank', 'u32', 16),
    ('deepseek41.attention.sliding_window', 'u32', 4),
    ('deepseek41.attention.output_group_count', 'u32', 1),
    ('deepseek41.attention.output_lora_rank', 'u32', 16),
    ('deepseek41.attention.layer_norm_rms_epsilon', 'f32', 1e-5),
    ('deepseek41.rope.dimension_count', 'u32', 4),
    ('deepseek41.rope.freq_base', 'f32', 10000.0),
    ('deepseek41.expert_count', 'u32', 4),
    ('deepseek41.expert_used_count', 'u32', 2),
    ('deepseek41.expert_shared_count', 'u32', 1),
    ('deepseek41.expert_feed_forward_length', 'u32', 32),
    ('deepseek41.expert_gating_func', 'u32', 4),
    ('deepseek41.expert_weights_scale', 'f32', 1.5),
    ('deepseek41.expert_weights_norm', 'bool', True),
    # A non-zero ratio on a layer that is not a KV source is legal. The real
    # model has many such entries; compressor tensors belong only to sources.
    ('deepseek41.attention.compress_ratios', ('arr','u32'), [2]),
    ('deepseek41.attention.kv_source_layers', ('arr','u32'), []),
    ('deepseek41.attention.indexer.head_count', 'u32', 2),
    ('deepseek41.attention.indexer.key_length', 'u32', 4),
    ('deepseek41.attention.indexer.top_k', 'u32', 4),
    ('deepseek41.attention.indexer.source_layers', ('arr','u32'), []),
    ('deepseek41.hyper_connection.count', 'u32', 1),
    ('deepseek41.hyper_connection.sinkhorn_iterations', 'u32', 2),
    ('deepseek41.hyper_connection.epsilon', 'f32', 1e-6),
    ('deepseek41.swiglu_clamp_exp', ('arr','f32'), [10.0]),
    ('deepseek41.swiglu_clamp_shexp', ('arr','f32'), [10.0]),
    ('tokenizer.ggml.model', 'str', 'gpt2'),
    ('tokenizer.ggml.pre', 'str', 'gpt2'),
    ('tokenizer.ggml.tokens', ('arr','str'), chars),
    ('tokenizer.ggml.token_type', ('arr','u32'), [1] * 256),
    ('tokenizer.ggml.merges', ('arr','str'), []),
    ('tokenizer.ggml.add_bos_token', 'bool', False),
    ('tokenizer.ggml.add_eos_token', 'bool', False),
    ('split.no', 'u32', 0),
    ('split.count', 'u32', 1),
]

tensors = [
    ('token_embd.weight', [64, 256]),
    ('output.weight', [64, 256]),
    ('output_norm.weight', [64]),
    ('blk.0.attn_norm.weight', [64]),
    ('blk.0.attn_sinks.weight', [2]),
    ('blk.0.attn_q_a.weight', [64, 16]),
    ('blk.0.attn_q_a_norm.weight', [16]),
    ('blk.0.attn_q_b.weight', [16, 16]),
    ('blk.0.attn_kv.weight', [64, 8]),
    ('blk.0.attn_kv_a_norm.weight', [8]),
    ('blk.0.attn_output_a.weight', [16, 16]),
    ('blk.0.attn_output_b.weight', [16, 64]),
    ('blk.0.hc_attn_fn.weight', [64, 3]),
    ('blk.0.hc_attn_base.weight', [3]),
    ('blk.0.hc_attn_scale.weight', [3]),
    ('blk.0.hc_ffn_fn.weight', [64, 3]),
    ('blk.0.hc_ffn_base.weight', [3]),
    ('blk.0.hc_ffn_scale.weight', [3]),
    ('blk.0.ffn_norm.weight', [64]),
    ('blk.0.ffn_gate_inp.weight', [64, 4]),
    ('blk.0.exp_probs_b.bias', [4]),
    ('blk.0.exp_probs_b_vl.bias', [4]),
    ('blk.0.ffn_gate_exps.weight', [64, 32, 4]),
    ('blk.0.ffn_down_exps.weight', [32, 64, 4]),
    ('blk.0.ffn_up_exps.weight', [64, 32, 4]),
    ('blk.0.ffn_gate_shexp.weight', [64, 32]),
    ('blk.0.ffn_down_shexp.weight', [32, 64]),
    ('blk.0.ffn_up_shexp.weight', [64, 32]),
]

meta.append(('split.tensors.count', 'u32', len(tensors)))

# header
h = b'GGUF' + struct.pack('<I', 3) + struct.pack('<Q', len(tensors)) + struct.pack('<Q', len(meta))
for k, t, v in meta:
    h += s(k) + val(t, v)
# tensor infos
infos = b''
offset = 0
data = b''
for name, dims in tensors:
    nbytes = 4
    for d in dims:
        nbytes *= d
    pad = (-offset) % ALIGN
    offset += pad
    data += b'\0' * pad
    infos += s(name)
    infos += struct.pack('<I', len(dims))
    for d in dims:
        infos += struct.pack('<Q', d)
    infos += struct.pack('<I', 0)  # F32
    infos += struct.pack('<Q', offset)
    data += b'\0' * nbytes
    offset += nbytes

h += infos
header_end = len(h)
data_start = (header_end + ALIGN - 1) // ALIGN * ALIGN
h += b'\0' * (data_start - header_end)
out = h + data
open(OUT, 'wb').write(out)
print(OUT, len(out), 'bytes; header', header_end, 'data_start', data_start, 'tensors', len(tensors))
