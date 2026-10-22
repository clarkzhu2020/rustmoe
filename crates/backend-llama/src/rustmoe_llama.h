#pragma once
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct rustmoe_llama_session rustmoe_llama_session;

rustmoe_llama_session * rustmoe_llama_open(
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
    size_t error_len);

void rustmoe_llama_close(rustmoe_llama_session * session);

int32_t rustmoe_llama_n_vocab(const rustmoe_llama_session * session);
int32_t rustmoe_llama_n_ctx(const rustmoe_llama_session * session);
int32_t rustmoe_llama_n_past(const rustmoe_llama_session * session);

int32_t rustmoe_llama_tokenize(
    const rustmoe_llama_session * session,
    const char * text,
    int32_t text_len,
    int32_t * tokens,
    int32_t capacity,
    int32_t add_special);

int32_t rustmoe_llama_token_to_piece(
    const rustmoe_llama_session * session,
    int32_t token,
    char * buffer,
    int32_t capacity);

int32_t rustmoe_llama_prefill(
    rustmoe_llama_session * session,
    const int32_t * tokens,
    int32_t n_tokens);

int32_t rustmoe_llama_decode(
    rustmoe_llama_session * session,
    int32_t token);

const float * rustmoe_llama_logits(rustmoe_llama_session * session);

void rustmoe_llama_reset(rustmoe_llama_session * session);
int32_t rustmoe_llama_moe_cache_stats(
    const rustmoe_llama_session * session,
    uint64_t * hits,
    uint64_t * misses,
    uint64_t * uploaded_bytes,
    uint64_t * copied_hits,
    uint64_t * copied_bytes);

const char * rustmoe_llama_last_error(const rustmoe_llama_session * session);

#ifdef __cplusplus
}
#endif
