# Model setup

RustMoE can list HuggingFace-backed model profiles, query recent GGUF
repositories, and add a model with a resumable download wizard.

    rustmoe --setup models list
    rustmoe --setup models list --search deepseek
    rustmoe --setup models add
    rustmoe --setup models remove <profile>

The same commands work without the leading double dash:

    rustmoe setup models list

## List

The offline list shows validated profiles, approximate sizes, context, and
whether a profile is installed or current:

    PROFILE                                QUANT    SIZE   CONTEXT  STATUS
    deepseek-v4.1-flash-iq1-s              IQ1_S    233 GB   128K   current
    deepseek-v4.1-flash-q4-k-xl            Q4_K_XL  501 GB   128K   not installed
    deepseek-v4.1-flash-q8-k-xl            Q8_K_XL  509 GB   128K   not installed

The --search option queries the HuggingFace models API for recent GGUF
repositories:

    rustmoe --setup models list --search deepseek
    rustmoe --setup models list --search qwen

The search is live and newest-first. Repositories without a RustMoE profile are
shown as unsupported. RustMoE Deep currently supports the DeepSeek-V4.1
architecture; Qwen and other architectures cannot be added as local DeepSeek
engines.

The --json option is available for scripting.

## Add wizard

    rustmoe --setup models add

The wizard:

1. Shows the supported profiles and asks for one.
2. Asks for a destination. Default: models/<profile-id>
3. Queries the HuggingFace tree API for the exact GGUF files.
4. Prints file count, download size, disk free, context, and memory target.
5. Asks for confirmation.
6. Downloads each shard with HTTP Range resume.
7. Writes .models/setup-models.json and .models/model-path.txt.
8. startdeepseek.bat then uses the new pointer.

Non-interactive form:

    rustmoe --setup models add ^
      --profile deepseek-v4.1-flash-iq1-s ^
      --dest D:\models\iq1-s ^
      --yes

Preview without downloading:

    rustmoe --setup models add --profile deepseek-v4.1-flash-iq1-s --dry-run --yes

Register files that are already downloaded:

    rustmoe --setup models add ^
      --profile deepseek-v4.1-flash-iq1-s ^
      --dest D:\models\iq1-s ^
      --no-download --yes

## Resume and tokens

Downloads use a .part file and an HTTP Range request. Interrupted downloads
resume from the existing byte count. A completed file is skipped when its size
matches the HuggingFace metadata.

For gated repositories:

    set HF_TOKEN=hf_...
    rustmoe --setup models add --profile ...

The token is sent only to huggingface.co.

## Disk and memory

Sizes are estimates from the profile and are checked against the HuggingFace
tree. The wizard refuses a download when the target filesystem reports less free
space than the model.

| Profile | Approx. download | Practical host |
|---|---:|---|
| IQ1_S | 233 GB | 24 GB VRAM, 64 GB RAM, NVMe |
| Q4_K_XL | 501 GB | large NVMe; 64 GB RAM uses heavy SSD paging |
| Q8_K_XL | 509 GB | 128 GB+ RAM and large NVMe recommended |

The IQ1_S profile is the current validated low-VRAM profile. The larger profiles
are listed for completeness but are not yet validated by this project.

## Remove

    rustmoe --setup models remove deepseek-v4.1-flash-iq1-s

By default the files stay on disk and only the registry/pointer entry is
removed. Add --delete-files to remove the model directory. The command refuses
obvious unsafe paths.

## Scope

This is a model downloader and registry, not an engine installer. It does not
install CUDA, llama.cpp, ROCm, or a DeepSeek engine. For the real DeepSeek
backend, build RustMoE with --features llama as described in
[real-backend.md](real-backend.md).

Strata's custom DeepSeek pack is not a RustMoE model profile. Use Strata for
that engine; see [dual-engine.md](dual-engine.md).
