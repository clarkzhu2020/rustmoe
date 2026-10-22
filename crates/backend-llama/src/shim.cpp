#include "rustmoe_llama.h"

#include <llama.h>
#include <ggml-backend.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

static void rustmoe_log_cb(enum ggml_log_level level, const char * text, void * /*user*/) {
    if (level == GGML_LOG_LEVEL_ERROR) {
        std::fputs(text, stderr);
        std::fflush(stderr);
    }
}

struct rustmoe_llama_session {
    llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    const llama_vocab * vocab = nullptr;
    int32_t n_past = 0;
    int32_t n_batch = 512;
    int32_t n_vocab = 0;
    char error[1024] = {0};
};

static void set_error(rustmoe_llama_session * s, const char * msg, char * out, size_t out_len) {
    if (s) {
        std::snprintf(s->error, sizeof(s->error), "%s", msg ? msg : "unknown error");
    }
    if (out && out_len > 0) {
        std::snprintf(out, out_len, "%s", msg ? msg : "unknown error");
    }
}

static void set_error(rustmoe_llama_session * s, const std::exception & e, char * out, size_t out_len) {
    set_error(s, e.what(), out, out_len);
}

static llama_load_mode parse_load_mode(int mode) {
    switch (mode) {
        case 0: return LLAMA_LOAD_MODE_AUTO;
        case 1: return LLAMA_LOAD_MODE_MMAP;
        case 2: return LLAMA_LOAD_MODE_MLOCK;
        case 3: return LLAMA_LOAD_MODE_MMAP_MLOCK;
        case 4: return LLAMA_LOAD_MODE_DIRECT_IO;
        default: return LLAMA_LOAD_MODE_MMAP;
    }
}

static llama_lazy_mode parse_lazy_mode(int mode) {
    switch (mode) {
        case 0: return LLAMA_LAZY_MODE_OFF;
        case 1: return LLAMA_LAZY_MODE_AUTO;
        case 2: return LLAMA_LAZY_MODE_ON;
        default: return LLAMA_LAZY_MODE_AUTO;
    }
}

static int32_t decode_batch(rustmoe_llama_session * s, const int32_t * tokens, int32_t n_tokens) {
    if (!s || !s->ctx || !tokens || n_tokens <= 0) {
        return -1;
    }
    if (s->n_past + n_tokens > (int32_t) llama_n_ctx(s->ctx)) {
        set_error(s, "context window exhausted", nullptr, 0);
        return -2;
    }
    int32_t offset = 0;
    while (offset < n_tokens) {
        const int32_t chunk = std::min(n_tokens - offset, s->n_batch > 0 ? s->n_batch : 512);
        llama_batch batch = llama_batch_init(chunk, 0, 1);
        if (!batch.token || !batch.pos || !batch.n_seq_id || !batch.seq_id || !batch.logits) {
            llama_batch_free(batch);
            set_error(s, "failed to allocate llama batch", nullptr, 0);
            return -1;
        }
        batch.n_tokens = chunk;
        for (int32_t i = 0; i < chunk; ++i) {
            const int32_t absolute = offset + i;
            batch.token[i] = tokens[absolute];
            batch.pos[i] = s->n_past + i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = (absolute == n_tokens - 1) ? 1 : 0;
        }
        const int32_t rc = llama_decode(s->ctx, batch);
        llama_batch_free(batch);
        if (rc != 0) {
            char msg[256];
            std::snprintf(msg, sizeof(msg), "llama_decode failed with code %d", rc);
            set_error(s, msg, nullptr, 0);
            return rc;
        }
        s->n_past += chunk;
        offset += chunk;
    }
    return 0;
}

extern "C" rustmoe_llama_session * rustmoe_llama_open(
    const char * model_path,
    int32_t n_gpu_layers,
    int32_t n_ctx,
    int32_t n_batch,
    int32_t n_threads,
    int32_t n_threads_batch,
    int32_t moe_cache_mib,
    int32_t load_mode,
    int32_t lazy_mode,
    int32_t cpu_moe_layers,
    char * error,
    size_t error_len) {
    if (error && error_len > 0) {
        error[0] = 0;
    }
    if (!model_path || !*model_path) {
        if (error && error_len > 0) std::snprintf(error, error_len, "empty model path");
        return nullptr;
    }
    try {
        llama_backend_init();
        llama_log_set(rustmoe_log_cb, nullptr);
        auto * s = new rustmoe_llama_session();

        std::vector<std::string> patterns;
        if (cpu_moe_layers != 0) {
            const int32_t count = cpu_moe_layers < 0 ? 100000 : cpu_moe_layers;
            const char * suffix = "\\.ffn_(up|down|gate|gate_up)_(ch|)exps";
            if (cpu_moe_layers < 0) {
                patterns.emplace_back(suffix);
            } else {
                for (int32_t i = 0; i < count; ++i) {
                    char buf[160];
                    std::snprintf(buf, sizeof(buf), "blk\\.%d%s", i, suffix);
                    patterns.emplace_back(buf);
                }
            }
        }
        std::vector<llama_model_tensor_buft_override> overrides;
        overrides.reserve(patterns.size() + 1);
        for (const auto & p : patterns) {
            overrides.push_back({p.c_str(), ggml_backend_cpu_buffer_type()});
        }
        overrides.push_back({nullptr, nullptr});

        llama_model_params mp = llama_model_default_params();
        mp.n_gpu_layers = n_gpu_layers;
        mp.split_mode = LLAMA_SPLIT_MODE_NONE;
        mp.main_gpu = 0;
        mp.load_mode = parse_load_mode(load_mode);
        mp.lazy_mode = parse_lazy_mode(lazy_mode);
        mp.use_extra_bufts = false;
        mp.vocab_only = false;
        mp.check_tensors = false;
        mp.no_host = false;
        mp.no_alloc = false;
        mp.load_mtp = false;
        if (!overrides.empty() && overrides.size() > 1) {
            mp.tensor_buft_overrides = overrides.data();
        }

        s->model = llama_model_load_from_file(model_path, mp);
        if (!s->model) {
            set_error(s, "llama_model_load_from_file returned null", error, error_len);
            rustmoe_llama_close(s);
            return nullptr;
        }
        s->vocab = llama_model_get_vocab(s->model);
        s->n_vocab = llama_vocab_n_tokens(s->vocab);

        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = n_ctx > 0 ? (uint32_t) n_ctx : 4096;
        cp.n_batch = n_batch > 0 ? (uint32_t) n_batch : 512;
        cp.n_ubatch = std::min<uint32_t>(cp.n_batch, 512);
        s->n_batch = (int32_t) cp.n_batch;
        cp.n_seq_max = 1;
        cp.n_threads = n_threads > 0 ? n_threads : 16;
        cp.n_threads_batch = n_threads_batch > 0 ? n_threads_batch : cp.n_threads;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;
        cp.moe_cache_size = (size_t) std::max<int32_t>(0, moe_cache_mib) * 1024ull * 1024ull;
        cp.offload_kqv = true;
        cp.op_offload = true;
        cp.no_perf = false;
        cp.swa_full = false;
        cp.kv_unified = true;
        cp.n_outputs_max = 1;

        s->ctx = llama_init_from_model(s->model, cp);
        if (!s->ctx) {
            set_error(s, "llama_init_from_model returned null", error, error_len);
            rustmoe_llama_close(s);
            return nullptr;
        }
        return s;
    } catch (const std::exception & e) {
        if (error && error_len > 0) {
            std::snprintf(error, error_len, "%s", e.what());
        }
        return nullptr;
    } catch (...) {
        if (error && error_len > 0) {
            std::snprintf(error, error_len, "unknown exception while loading model");
        }
        return nullptr;
    }
}

extern "C" void rustmoe_llama_close(rustmoe_llama_session * s) {
    if (!s) return;
    if (s->ctx) {
        llama_free(s->ctx);
        s->ctx = nullptr;
    }
    if (s->model) {
        llama_model_free(s->model);
        s->model = nullptr;
    }
    delete s;
}

extern "C" int32_t rustmoe_llama_n_vocab(const rustmoe_llama_session * s) {
    return s ? s->n_vocab : 0;
}

extern "C" int32_t rustmoe_llama_n_ctx(const rustmoe_llama_session * s) {
    return (s && s->ctx) ? (int32_t) llama_n_ctx(s->ctx) : 0;
}

extern "C" int32_t rustmoe_llama_n_past(const rustmoe_llama_session * s) {
    return s ? s->n_past : 0;
}

extern "C" int32_t rustmoe_llama_tokenize(
    const rustmoe_llama_session * s,
    const char * text,
    int32_t text_len,
    int32_t * tokens,
    int32_t capacity,
    int32_t add_special) {
    if (!s || !s->vocab || !text) return -1;
    return llama_tokenize(s->vocab, text, text_len, tokens, capacity, add_special != 0, true);
}

extern "C" int32_t rustmoe_llama_token_to_piece(
    const rustmoe_llama_session * s,
    int32_t token,
    char * buffer,
    int32_t capacity) {
    if (!s || !s->vocab) return -1;
    return llama_token_to_piece(s->vocab, token, buffer, capacity, 0, true);
}

extern "C" int32_t rustmoe_llama_prefill(
    rustmoe_llama_session * s,
    const int32_t * tokens,
    int32_t n_tokens) {
    return decode_batch(s, tokens, n_tokens);
}

extern "C" int32_t rustmoe_llama_decode(rustmoe_llama_session * s, int32_t token) {
    return decode_batch(s, &token, 1);
}

extern "C" const float * rustmoe_llama_logits(rustmoe_llama_session * s) {
    if (!s || !s->ctx) return nullptr;
    return llama_get_logits_ith(s->ctx, -1);
}

extern "C" void rustmoe_llama_reset(rustmoe_llama_session * s) {
    if (!s || !s->ctx) return;
    llama_memory_clear(llama_get_memory(s->ctx), true);
    s->n_past = 0;
}

extern "C" int32_t rustmoe_llama_moe_cache_stats(
    const rustmoe_llama_session * s,
    uint64_t * hits,
    uint64_t * misses,
    uint64_t * uploaded_bytes,
    uint64_t * copied_hits,
    uint64_t * copied_bytes) {
    if (!s || !s->ctx) return -1;
    llama_moe_cache_stats st = {};
    llama_moe_cache_get_stats(s->ctx, &st);
    if (hits) *hits = st.hits;
    if (misses) *misses = st.misses;
    if (uploaded_bytes) *uploaded_bytes = st.uploaded_bytes;
    if (copied_hits) *copied_hits = st.copied_hits;
    if (copied_bytes) *copied_bytes = st.copied_bytes;
    return 0;
}

extern "C" const char * rustmoe_llama_last_error(const rustmoe_llama_session * s) {
    return s ? s->error : "no session";
}
