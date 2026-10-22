# Model snapshot and verified facts

## Snapshot

- Model: unsloth/DeepSeek-V4.1-Flash-1017-GGUF
- Repository commit: 60ad3651641b5c97679c807e064ab1d7a3ca1455
- Last modified: 2026-10-20T10:18:03Z
- Quant directories inspected: UD-Q4_K_XL and UD-IQ1_S
- Reference implementation used for real execution:
  unslothai/llama.cpp, branch dsv41-all, commit
d1781c3ea6ab7c75c635a8fea60d2b3f4937c645 (2026-10-20)
- The model card points to unslothai/llama.cpp PR 298 for the current loader.
- Earlier structural checks also used JigSawPT/llama.cpp dsv41-porte commit
  8ce0968539b1a6795f3413c2783dfa5646b4688a.
- Inspection date: 2026-10-21.

RustMoE pins no model download. These values are recorded so tests and benchmark
results can be reproduced.

## Q4_K_XL shards

All sizes are bytes. SHA-256 values are the Hugging Face LFS object ids.

| Shard | Size | SHA-256 |
|---:|---:|---|
| 00001 | 5,780,589 | cc15d470afcb030558bbc4d8e2ccd840182e2e95de1a95198138efe63a5fd99f |
| 00002 | 48,129,640,000 | 17df217d8fc26a39f21ea40805dc6ab788c7ce8623a59a95c431051df7eafc94 |
| 00003 | 48,129,640,000 | 0c8065f620a049cb437024c54ff12c314c94603a1cc683f5191be4689d6d6d3b |
| 00004 | 48,129,640,000 | 3bd8b2ebf8f51d8708a73ae0895f8d4acc7e4c983ba4e89d038492b203cc89ac |
| 00005 | 48,129,640,000 | 929054a6bbf3701765779b58c37988b6c6284a3b8413901858b59c0e95f37cc3 |
| 00006 | 48,129,640,000 | 49d3de04bd6a09de786c9ef7db900eae02ee9fd391573f91a923013d7d5b24d3 |
| 00007 | 48,129,640,000 | f18feae5878dd680f1ea87ecd1d84f4fb098480298b25a3162dcbdab8dc2ca5b |
| 00008 | 202,758,032,592 | 12df293355efbd104b088a77e13ee22410fb1ec3f7ab3863947f09896e64bec7 |
| 00009 | 9,165,923,392 | c21ad6105a780bc2f44eb7dd41f9b1a1efb4e8ec87618f60b733bc2652bd2ffe |

Total file bytes: 500,707,576,573.

## IQ1_S shards

UD-IQ1_S is the quant used for the 12 GB real-decode validation. It fits a
500 GB disk and is much smaller than Q4_K_XL.

| Shard | Size | SHA-256 |
|---:|---:|---|
| 00001 | 5,780,923 | b843148a82de2b24df0c5a656669edbb9212eb69a41953249fe6c50d04186177 |
| 00002 | 49,120,285,248 | 4cb1b13e50db73650d6d9fcaf22cd938b4b2e993740c2abe6815f82107048461 |
| 00003 | 48,899,707,008 | def479df0dca997b9cad4716f6375296c1f805887b5cde669105db99f0d9dd75 |
| 00004 | 24,358,544,704 | 863e1ba47e2479aedc9ce2cc3fe2f57e091548c943bd56d7dd149a771637a17d |
| 00005 | 110,595,290,592 | 771805301e695d72d268151dfc43043581ba904298574aa879d4349cbe67b020 |

Total file bytes: 232,979,608,475 (233.0 GB decimal, 217.0 GiB). The model has the same 40-layer
DeepSeek-V4.1 architecture. IQ1_S uses fork-local GGML types 64-66, added to
RustMoE's type table for byte accounting.


## Metadata observed

| Key | Value |
|---|---|
| general.architecture | deepseek41 |
| general.name | DeepSeek-V4.1-Flash-1017 |
| general.size_label | 748B-A16B |
| deepseek41.block_count | 40 |
| deepseek41.context_length | 1048576 |
| deepseek41.embedding_length | 5120 |
| deepseek41.attention.head_count | 64 |
| deepseek41.attention.head_count_kv | 1 |
| deepseek41.attention.key_length | 512 |
| deepseek41.attention.value_length | 512 |
| deepseek41.attention.q_lora_rank | 1280 |
| deepseek41.attention.sliding_window | 128 |
| deepseek41.attention.output_group_count | 8 |
| deepseek41.attention.output_lora_rank | 1024 |
| deepseek41.rope.dimension_count | 64 |
| deepseek41.expert_count | 384 |
| deepseek41.expert_used_count | 6 |
| deepseek41.expert_shared_count | 1 |
| deepseek41.expert_feed_forward_length | 2304 |
| deepseek41.expert_gating_func | 4 (sqrtsoftplus) |
| deepseek41.expert_weights_scale | 1.5 |
| deepseek41.attention.kv_source_layers | [2, 8, 14, 20] |
| deepseek41.attention.indexer.source_layers | [2, 8, 14, 20, 24, 28, 32, 36] |
| deepseek41.attention.compress_ratios | length 43, first 40 used |
| deepseek41.ple.layers | [1, 14] |
| deepseek41.ple.head_vocab_sizes | 48 values, sum 768,022,850 |
| deepseek41.embedding_length_per_layer_input | 256 |
| deepseek41.dspark.block_size | 5 |
| deepseek41.dspark.target_layers | [37, 38, 39] |
| tokenizer.ggml.model | gpt2 |
| tokenizer.ggml.pre | joyai-llm |
| tokenizer.ggml.tokens | 129,280 |
| tokenizer.ggml.merges | 127,741 |
| tokenizer.ggml.eos_token_id | 1 |
| general.file_type | 38 |

compress_ratios has 43 entries because the checkpoint configuration includes
the optional MTP/DSpark stage. The main GGUF has 40 blocks; no blk.40..42 tensors
are present. DSpark is a separate dflash GGUF.

## Tensor catalog

- 1043 tensors across 9 shards.
- 40 layers, 23-26 tensors per layer depending on compressors and indexers.
- 120 routed-expert tensors = 3 per layer x 40.
- One PLE table tensor, type 43, shape [256, 768022850].
- Token embeddings and output are Q8_0 [5120, 129280].
- All norms, routers, biases, and hyper-connection tensors are F32.
- Routed experts are MXFP4 [5120, 2304, 384], [2304, 5120, 384], and
  [5120, 2304, 384].

Byte totals by type:

| Type | Tensors | Bytes |
|---|---:|---:|
| MXFP4 (39) | 120 | 288,777,830,400 |
| MXFP8 extension (43) | 1 | 202,758,032,400 |
| Q8_0 (8) | 351 | 8,691,744,768 |
| F32 (0) | 571 | 474,120,640 |
| Total | 1043 | 500,701,728,208 |

## Facts that differ from shorthand descriptions

1. The architecture key is deepseek41. A loader that expects deepseek_v41 will
   reject the real file.
2. The Q4_K_XL name does not mean Q4_K tensors. Experts are MXFP4 and the PLE
   table is MXFP8 type 43.
3. The 202.76 GB Engram/PLE table lives in shard 8. It is not a normal dense
   embedding and cannot be treated as resident by default.
4. DSpark is a separate dflash drafter, not 3 extra blocks inside these main
   files.
5. The model is image-text-to-text in its upstream configuration. This runtime
   is text-only.

## Reproduce the inspection

Shard 1 is small enough to download alone:

    curl -L -o shard1.gguf \
      https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF/resolve/main/UD-Q4_K_XL/DeepSeek-V4.1-Flash-1017-UD-Q4_K_XL-00001-of-00009.gguf

    rustmoe inspect shard1.gguf --single-file
    rustmoe tokenize shard1.gguf --single-file --text "Hello, world!"

The shard header can be fetched with an HTTP range request for the other shards.
A full validation requires all nine files and 500.7 GB of free disk.

## Reference behavior checked

The reference branch was read for:

- tensor naming and dimensions in src/models/deepseek4.cpp;
- metadata keys in src/llama-arch.cpp;
- compressed cache structure in src/llama-kv-cache-dsv4.cpp;
- GGML type ids in ggml/include/ggml.h;
- MXFP4 block layout in ggml/src/ggml-common.h.

Type 43 is not in that branch's ggml_type enum. The Unsloth quantizer uses it for
MXFP8 and the Engram README names that format explicitly. RustMoE treats it as a
documented extension, with a 32-element, 33-byte block and E8M0 scale. It must be
validated against the Unsloth loader before real execution.
