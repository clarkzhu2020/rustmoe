# Supported formats

## GGUF container

RustMoE reads GGUF version 2 and version 3.

- Magic: GGUF
- Little-endian scalars
- Metadata scalar types 0 through 12
- Arrays of scalar values
- Tensor directory with name, dimensions, ggml_type, and offset
- Default alignment 32 bytes; general.alignment is honored
- Split metadata: split.no, split.count, split.tensors.count

The parser reads only the header and tensor directory. Header size is bounded
by ParseOptions (256 MiB by default), as are string, array, tensor, and dimension
counts.

## Split naming

The entry file must contain a marker of the form -NNNNN-of-MMMMM.gguf. Given
shard 1, RustMoE constructs all expected paths and fails if one is absent.
Indices are zero-based inside the file and one-based in the filename.

A directory containing multiple unrelated GGUF sets is rejected; pass the entry
shard explicitly. A single non-split file is accepted only if split.count is
absent or 1.

## Tensor validation

For every tensor:

- shape has 1 to 8 dimensions, all non-zero;
- dimensions do not overflow the byte count;
- row dimension is a multiple of the quantization block size;
- offset is aligned to general.alignment;
- offset plus stored byte size is inside that shard;
- tensor names are unique across all shards;
- split.tensors.count matches the merged catalog.

## DeepSeek-V4.1 tensor roles

The catalog classifier recognizes:

- token_embd, output, output_norm;
- per_layer_token_embd (Engram/PLE table);
- attention Q/KV/output projections;
- attn_compressor_* (compressed K-only sources);
- indexer.* (indexer key and query projections);
- hc_attn_* and hc_ffn_* (superposed hyper-connections);
- ffn_gate_inp and exp_probs_b (router);
- ffn_*_exps (routed experts);
- ffn_*_shexp (shared expert);
- engram_gate and engram_kv.

Layer expectations are derived from compress_ratios, kv_source_layers,
index_source_layers, and ple.layers. Optional V4 tensors such as
attn_compressor_ape and hc_head_* are accepted but not required for V4.1.

## GGML types

The type table below is used for validation and byte accounting. Reference
dequantization is implemented for the types marked yes.

| id | name | block elements | block bytes | reference dequant |
|---:|---|---:|---:|---|
| 0 | f32 | 1 | 4 | yes |
| 1 | f16 | 1 | 2 | yes |
| 2 | q4_0 | 32 | 18 | yes |
| 3 | q4_1 | 32 | 20 | no |
| 6 | q5_0 | 32 | 22 | no |
| 7 | q5_1 | 32 | 24 | no |
| 8 | q8_0 | 32 | 34 | yes |
| 9 | q8_1 | 32 | 36 | no |
| 10 | q2_K | 256 | 84 | no |
| 11 | q3_K | 256 | 110 | no |
| 12 | q4_K | 256 | 144 | yes |
| 13 | q5_K | 256 | 176 | no |
| 14 | q6_K | 256 | 210 | no |
| 15 | q8_K | 256 | 292 | no |
| 16 | iq2_xxs | 256 | 66 | no |
| 17 | iq2_xs | 256 | 74 | no |
| 18 | iq3_xxs | 256 | 98 | no |
| 19 | iq1_s | 256 | 50 | no |
| 20 | iq4_nl | 32 | 18 | no |
| 21 | iq3_s | 256 | 112 | no |
| 22 | iq2_s | 256 | 82 | no |
| 23 | iq4_xs | 256 | 136 | no |
| 24 | i8 | 1 | 1 | no |
| 25 | i16 | 1 | 2 | no |
| 26 | i32 | 1 | 4 | no |
| 27 | i64 | 1 | 8 | no |
| 28 | f64 | 1 | 8 | no |
| 29 | iq1_m | 256 | 56 | no |
| 30 | bf16 | 1 | 2 | yes |
| 34 | tq1_0 | 256 | 54 | no |
| 35 | tq2_0 | 256 | 66 | no |
| 39 | mxfp4 | 32 | 17 | yes |
| 40 | nvfp4 | 64 | 36 | no |
| 41 | q1_0 | 128 | 18 | no |
| 42 | q2_0 | 64 | 18 | no |
| 43 | mxfp8 (extension) | 32 | 33 | yes |
| 64 | iq1_xs (fork extension) | 256 | 46 | no |
| 65 | iq1_xxs (fork extension) | 256 | 42 | no |
| 66 | iq1_xxxs (fork extension) | 256 | 38 | no |

Unknown type ids are errors. Known types without a reference dequantizer are
errors in strict validation and warnings with --allow-undecodable. They are
never silently converted.

Types 64-66 are Unsloth fork-local IQ1 variants used by UD-IQ1_S. They are
accepted for byte accounting and passed through to the llama.cpp reference
backend; RustMoE has no native dequantizer for them.

## The real UD-Q4_K_XL directory

This quant is a mixture:

- routed experts: MXFP4, 120 tensors, 288,777,830,400 bytes;
- PLE table: MXFP8 type 43, one tensor of 256 x 768,022,850, 202,758,032,400
  bytes;
- dense Q8_0: 8,691,744,768 bytes;
- F32: 474,120,640 bytes;
- total tensor bytes: 500,701,728,208;
- total file bytes including headers: 500,707,576,573.

The file name is a recipe name. It is not a promise that every tensor uses
Q4_K. Always dispatch on the numeric type in the tensor directory.

## DSpark

The main Q4_K_XL model files do not contain the DSpark drafter. DSpark is an
optional separate GGUF with general.architecture = dflash, 78 tensors, and its
own quantized experts. RustMoE v0.1 recognizes the DSpark metadata in the main
model but does not load or execute the drafter.

## Tokenizer

Supported GGUF tokenizer model: gpt2 byte-level BPE.

- Reads tokens, token types, merges, BOS/EOS/PAD ids, and add-BOS/add-EOS flags.
- Implements the exact GPT-2 byte alphabet and merge-rank loop.
- Control/user-defined tokens decode literally.
- Reports pre_tokenizer_exact=false for unknown custom pre-tokenizers such as
  joyai-llm. The generic Unicode-aware splitter is used in that case.
- The official DeepSeek-V4.1 Jinja chat template is stored in metadata but is
  not evaluated by v0.1. The HTTP layer uses a labelled simple fallback.

## Models other than deepseek41

The parser can inspect any GGUF v2/v3 file. Typed model validation currently
requires general.architecture = deepseek41. A dflash drafter can be inspected as
raw GGUF but is not accepted by the runtime.
