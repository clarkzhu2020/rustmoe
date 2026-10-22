#![cfg(feature = "llama")]
//! Reference execution backend backed by the Unsloth DeepSeek-V4.1
//! llama.cpp branch.
//!
//! llama.cpp owns model loading, the router, MLA, Engram, and kernels. RustMoE
//! owns the API, lifecycle, metrics, and resource policy. The shim exposes only
//! a token-level interface so no llama.cpp struct layout leaks into Rust.

use rustmoe_backend::{BackendError, BackendOutput, InferenceBackend, ModelState};
use rustmoe_model::ModelDescriptor;
use serde::Serialize;
use std::ffi::{c_char, c_float, c_int, CStr, CString};
use std::ptr;

#[repr(C)]
pub struct LlamaSession {
    _private: [u8; 0],
}

extern "C" {
    fn rustmoe_llama_open(
        model_path: *const c_char,
        n_gpu_layers: c_int,
        n_ctx: c_int,
        n_batch: c_int,
        n_threads: c_int,
        n_threads_batch: c_int,
        moe_cache_mib: c_int,
        load_mode: c_int,
        lazy_mode: c_int,
        cpu_moe_layers: c_int,
        error: *mut c_char,
        error_len: usize,
    ) -> *mut LlamaSession;
    fn rustmoe_llama_close(session: *mut LlamaSession);
    fn rustmoe_llama_n_vocab(session: *const LlamaSession) -> c_int;
    fn rustmoe_llama_n_ctx(session: *const LlamaSession) -> c_int;
    fn rustmoe_llama_tokenize(
        session: *const LlamaSession,
        text: *const c_char,
        text_len: c_int,
        tokens: *mut c_int,
        capacity: c_int,
        add_special: c_int,
    ) -> c_int;
    fn rustmoe_llama_token_to_piece(
        session: *const LlamaSession,
        token: c_int,
        buffer: *mut c_char,
        capacity: c_int,
    ) -> c_int;
    fn rustmoe_llama_prefill(session: *mut LlamaSession, tokens: *const c_int, n_tokens: c_int) -> c_int;
    fn rustmoe_llama_decode(session: *mut LlamaSession, token: c_int) -> c_int;
    fn rustmoe_llama_logits(session: *mut LlamaSession) -> *const c_float;
    fn rustmoe_llama_reset(session: *mut LlamaSession);
    fn rustmoe_llama_moe_cache_stats(
        session: *const LlamaSession,
        hits: *mut u64,
        misses: *mut u64,
        uploaded_bytes: *mut u64,
        copied_hits: *mut u64,
        copied_bytes: *mut u64,
    ) -> c_int;
    fn rustmoe_llama_last_error(session: *const LlamaSession) -> *const c_char;
}

#[derive(Clone, Debug, Serialize)]
pub struct LlamaCppOptions {
    pub context_size: u32,
    pub n_gpu_layers: i32,
    pub n_batch: i32,
    pub n_threads: i32,
    pub n_threads_batch: i32,
    pub moe_cache_mib: i32,
    pub load_mode: i32,
    pub lazy_mode: i32,
    /// -1 keeps all routed expert tensors on the host, 0 keeps all on GPU, and
    /// N > 0 keeps the first N layers on the host.
    pub cpu_moe_layers: i32,
}

impl Default for LlamaCppOptions {
    fn default() -> Self {
        Self {
            context_size: 4096,
            n_gpu_layers: 999,
            n_batch: 512,
            n_threads: 16,
            n_threads_batch: 16,
            moe_cache_mib: 1024,
            load_mode: 1,
            lazy_mode: 2,
            cpu_moe_layers: -1,
        }
    }
}

pub struct LlamaCppBackend {
    session: *mut LlamaSession,
    vocab_size: usize,
    n_ctx: usize,
    options: LlamaCppOptions,
    loaded: bool,
}

unsafe impl Send for LlamaCppBackend {}

impl LlamaCppBackend {
    pub fn new(options: LlamaCppOptions) -> Self {
        Self {
            session: ptr::null_mut(),
            vocab_size: 0,
            n_ctx: 0,
            options,
            loaded: false,
        }
    }

    pub fn options(&self) -> &LlamaCppOptions {
        &self.options
    }

    pub fn n_ctx(&self) -> usize {
        self.n_ctx
    }

    fn session(&self) -> Result<*mut LlamaSession, BackendError> {
        if self.session.is_null() {
            Err(BackendError::NotLoaded)
        } else {
            Ok(self.session)
        }
    }

    fn last_error(&self) -> String {
        if self.session.is_null() {
            return "no llama.cpp session".into();
        }
        unsafe {
            let ptr = rustmoe_llama_last_error(self.session);
            if ptr.is_null() {
                "unknown llama.cpp error".into()
            } else {
                CStr::from_ptr(ptr).to_string_lossy().into_owned()
            }
        }
    }

    fn copy_logits(&self) -> Result<Vec<f32>, BackendError> {
        unsafe {
            let ptr = rustmoe_llama_logits(self.session()?);
            if ptr.is_null() {
                return Err(BackendError::Failed("llama.cpp returned null logits".into()));
            }
            Ok(std::slice::from_raw_parts(ptr, self.vocab_size).to_vec())
        }
    }

    fn check_rc(&self, rc: c_int) -> Result<(), BackendError> {
        if rc == 0 {
            Ok(())
        } else if rc == -2 {
            Err(BackendError::ContextOverflow {
                position: 0,
                tokens: 0,
                limit: self.n_ctx as u64,
            })
        } else {
            Err(BackendError::Failed(self.last_error()))
        }
    }
}

impl Drop for LlamaCppBackend {
    fn drop(&mut self) {
        if !self.session.is_null() {
            unsafe { rustmoe_llama_close(self.session) };
            self.session = ptr::null_mut();
        }
    }
}

impl InferenceBackend for LlamaCppBackend {
    fn name(&self) -> &str {
        "llama.cpp"
    }

    fn load_model(&mut self, model: &ModelDescriptor) -> Result<(), BackendError> {
        if !self.session.is_null() {
            unsafe { rustmoe_llama_close(self.session) };
            self.session = ptr::null_mut();
        }
        let path = CString::new(
            model
                .path
                .to_str()
                .ok_or_else(|| BackendError::Failed("model path is not valid UTF-8".into()))?,
        )
        .map_err(|_| BackendError::Failed("model path contains NUL".into()))?;
        let mut error = vec![0 as c_char; 2048];
        let session = unsafe {
            rustmoe_llama_open(
                path.as_ptr(),
                self.options.n_gpu_layers,
                self.options.context_size as c_int,
                self.options.n_batch,
                self.options.n_threads,
                self.options.n_threads_batch,
                self.options.moe_cache_mib,
                self.options.load_mode,
                self.options.lazy_mode,
                self.options.cpu_moe_layers,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if session.is_null() {
            let message = unsafe { CStr::from_ptr(error.as_ptr()) }
                .to_string_lossy()
                .into_owned();
            return Err(BackendError::Failed(message));
        }
        self.session = session;
        self.vocab_size = unsafe { rustmoe_llama_n_vocab(session) }.max(0) as usize;
        self.n_ctx = unsafe { rustmoe_llama_n_ctx(session) }.max(0) as usize;
        if self.vocab_size != model.config.vocab_size as usize {
            unsafe { rustmoe_llama_close(session) };
            self.session = ptr::null_mut();
            return Err(BackendError::Failed(format!(
                "llama.cpp vocab {} differs from GGUF metadata {}",
                self.vocab_size, model.config.vocab_size
            )));
        }
        self.loaded = true;
        Ok(())
    }

    fn tokenize(&self, text: &str) -> Result<Option<Vec<u32>>, BackendError> {
        let session = self.session()?;
        let text = CString::new(text).map_err(|_| BackendError::Failed("prompt contains NUL".into()))?;
        let len = text.as_bytes().len() as c_int;
        let needed = unsafe { rustmoe_llama_tokenize(session, text.as_ptr(), len, ptr::null_mut(), 0, 0) };
        if needed == 0 {
            return Ok(Some(Vec::new()));
        }
        if needed > 0 {
            // A zero-capacity call should not return a positive count, but
            // tolerate a tokenizer that does.
            let mut buf = vec![0 as c_int; needed as usize];
            let n =
                unsafe { rustmoe_llama_tokenize(session, text.as_ptr(), len, buf.as_mut_ptr(), needed, 0) };
            if n < 0 {
                return Err(BackendError::Failed(self.last_error()));
            }
            buf.truncate(n as usize);
            return Ok(Some(buf.into_iter().map(|v| v as u32).collect()));
        }
        let capacity = (-needed) as usize;
        let mut buf = vec![0 as c_int; capacity];
        let n = unsafe {
            rustmoe_llama_tokenize(
                session,
                text.as_ptr(),
                len,
                buf.as_mut_ptr(),
                capacity as c_int,
                0,
            )
        };
        if n < 0 {
            return Err(BackendError::Failed(self.last_error()));
        }
        buf.truncate(n as usize);
        Ok(Some(buf.into_iter().map(|v| v as u32).collect()))
    }

    fn decode_token_bytes(&self, token: u32) -> Option<Vec<u8>> {
        if self.session.is_null() {
            return None;
        }
        let session = self.session;
        unsafe {
            let mut buf = vec![0u8; 256];
            let n = rustmoe_llama_token_to_piece(
                session,
                token as c_int,
                buf.as_mut_ptr() as *mut c_char,
                buf.len() as c_int,
            );
            if n >= 0 {
                buf.truncate(n as usize);
                return Some(buf);
            }
            let need = (-n) as usize;
            let mut buf = vec![0u8; need];
            let n = rustmoe_llama_token_to_piece(
                session,
                token as c_int,
                buf.as_mut_ptr() as *mut c_char,
                buf.len() as c_int,
            );
            if n < 0 {
                None
            } else {
                buf.truncate(n as usize);
                Some(buf)
            }
        }
    }

    fn reports_routes(&self) -> bool {
        false
    }

    fn cache_stats(&self) -> Option<rustmoe_backend::BackendCacheStats> {
        if self.session.is_null() {
            return None;
        }
        let mut hits = 0u64;
        let mut misses = 0u64;
        let mut uploaded_bytes = 0u64;
        let mut copied_hits = 0u64;
        let mut copied_bytes = 0u64;
        let rc = unsafe {
            rustmoe_llama_moe_cache_stats(
                self.session,
                &mut hits,
                &mut misses,
                &mut uploaded_bytes,
                &mut copied_hits,
                &mut copied_bytes,
            )
        };
        if rc != 0 {
            return None;
        }
        Some(rustmoe_backend::BackendCacheStats {
            hits,
            misses,
            uploaded_bytes,
            copied_hits,
            copied_bytes,
        })
    }

    fn prefill(&mut self, tokens: &[u32], state: &mut ModelState) -> Result<BackendOutput, BackendError> {
        if tokens.is_empty() {
            return Err(BackendError::EmptyPrefill);
        }
        let session = self.session()?;
        let ids: Vec<c_int> = tokens.iter().map(|t| *t as c_int).collect();
        let rc = unsafe { rustmoe_llama_prefill(session, ids.as_ptr(), ids.len() as c_int) };
        self.check_rc(rc)?;
        state.position = state.position.saturating_add(tokens.len() as u64);
        state.prompt_tokens = state.prompt_tokens.saturating_add(tokens.len() as u64);
        state.token_history.extend_from_slice(tokens);
        let logits = self.copy_logits()?;
        Ok(BackendOutput {
            logits,
            routes: Vec::new(),
        })
    }

    fn decode_step(&mut self, token: u32, state: &mut ModelState) -> Result<BackendOutput, BackendError> {
        let session = self.session()?;
        let rc = unsafe { rustmoe_llama_decode(session, token as c_int) };
        self.check_rc(rc)?;
        state.position = state.position.saturating_add(1);
        state.generated_tokens = state.generated_tokens.saturating_add(1);
        state.token_history.push(token);
        let logits = self.copy_logits()?;
        Ok(BackendOutput {
            logits,
            routes: Vec::new(),
        })
    }

    fn reset(&mut self) {
        if !self.session.is_null() {
            unsafe { rustmoe_llama_reset(self.session) };
        }
    }
}

pub fn backend_name() -> &'static str {
    "llama.cpp"
}
