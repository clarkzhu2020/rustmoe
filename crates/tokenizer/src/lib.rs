//! GGUF GPT-2 byte-level BPE tokenizer.
//!
//! The tokenizer reads its vocabulary and merge ranks from GGUF metadata. The
//! byte-level alphabet and merge loop are exact GPT-2 rules. The pre-tokenizer
//! is a Unicode-aware GPT-2-style splitter; when a model names a pre-tokenizer
//! that RustMoE has not verified (for example the DeepSeek joyai-llm split),
//! the tokenizer reports that fact instead of claiming bit-exact tokenization.

use rustmoe_gguf::{GgufSet, Metadata, MetadataValue};
use std::collections::HashMap;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum TokenizerError {
    #[error("missing tokenizer metadata key: {0}")]
    MissingKey(String),
    #[error("tokenizer.ggml.model is not gpt2: {0}")]
    UnsupportedModel(String),
    #[error("tokenizer vocabulary is empty")]
    EmptyVocab,
    #[error("tokenizer token id {0} is out of range")]
    InvalidToken(u32),
    #[error("bad merge entry: {0}")]
    BadMerge(String),
    #[error("duplicate token in vocabulary: {0}")]
    DuplicateToken(String),
}

#[derive(Clone, Debug)]
pub struct BpeTokenizer {
    tokens: Vec<String>,
    token_types: Vec<u32>,
    token_to_id: HashMap<String, u32>,
    merges: HashMap<(String, String), u32>,
    byte_to_char: [char; 256],
    char_to_byte: HashMap<char, u8>,
    pub model: String,
    pub pre_tokenizer: String,
    pub pre_tokenizer_exact: bool,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    pub pad_token_id: Option<u32>,
    pub add_bos_token: bool,
    pub add_eos_token: bool,
    pub has_chat_template: bool,
}

impl BpeTokenizer {
    pub fn from_gguf(gguf: &GgufSet) -> Result<Self, TokenizerError> {
        Self::from_metadata(&gguf.metadata)
    }

    pub fn from_metadata(metadata: &Metadata) -> Result<Self, TokenizerError> {
        let model = metadata
            .get_str("tokenizer.ggml.model")
            .unwrap_or("gpt2")
            .to_string();
        if model != "gpt2" {
            return Err(TokenizerError::UnsupportedModel(model));
        }
        let tokens = string_array(metadata, "tokenizer.ggml.tokens")?;
        if tokens.is_empty() {
            return Err(TokenizerError::EmptyVocab);
        }
        let token_types = match metadata.get_array("tokenizer.ggml.token_type") {
            Some(values) => values.iter().map(|v| v.as_u64().unwrap_or(1) as u32).collect(),
            None => vec![1u32; tokens.len()],
        };
        if token_types.len() != tokens.len() {
            return Err(TokenizerError::MissingKey(
                "tokenizer.ggml.token_type length".into(),
            ));
        }
        let merges = string_array(metadata, "tokenizer.ggml.merges").unwrap_or_default();
        let mut tok = Self::from_parts(
            tokens,
            token_types,
            merges,
            metadata.get_str("tokenizer.ggml.pre").unwrap_or("gpt2"),
        )?;
        tok.bos_token_id = metadata.get_u32("tokenizer.ggml.bos_token_id");
        tok.eos_token_id = metadata.get_u32("tokenizer.ggml.eos_token_id");
        tok.pad_token_id = metadata
            .get_u32("tokenizer.ggml.padding_token_id")
            .or_else(|| metadata.get_u32("tokenizer.ggml.pad_token_id"));
        tok.add_bos_token = metadata.get_bool("tokenizer.ggml.add_bos_token").unwrap_or(false);
        tok.add_eos_token = metadata.get_bool("tokenizer.ggml.add_eos_token").unwrap_or(false);
        tok.has_chat_template = metadata.contains_key("tokenizer.chat_template");
        Ok(tok)
    }

    pub fn from_parts(
        tokens: Vec<String>,
        token_types: Vec<u32>,
        merge_specs: Vec<String>,
        pre_tokenizer: &str,
    ) -> Result<Self, TokenizerError> {
        if tokens.is_empty() {
            return Err(TokenizerError::EmptyVocab);
        }
        if token_types.len() != tokens.len() {
            return Err(TokenizerError::MissingKey(
                "tokenizer.ggml.token_type length".into(),
            ));
        }
        let mut token_to_id = HashMap::with_capacity(tokens.len());
        for (i, token) in tokens.iter().enumerate() {
            if token_to_id.insert(token.clone(), i as u32).is_some() {
                return Err(TokenizerError::DuplicateToken(token.clone()));
            }
        }
        let mut merges = HashMap::with_capacity(merge_specs.len());
        for (rank, spec) in merge_specs.iter().enumerate() {
            let (a, b) = spec
                .split_once(' ')
                .ok_or_else(|| TokenizerError::BadMerge(spec.clone()))?;
            merges.insert((a.to_string(), b.to_string()), rank as u32);
        }
        let (byte_to_char, char_to_byte) = byte_alphabet();
        let exact = matches!(
            pre_tokenizer,
            "gpt2" | "qwen2" | "llama-bpe" | "deepseek-llm" | "deepseek-coder"
        );
        Ok(Self {
            tokens,
            token_types,
            token_to_id,
            merges,
            byte_to_char,
            char_to_byte,
            model: "gpt2".into(),
            pre_tokenizer: pre_tokenizer.to_string(),
            pre_tokenizer_exact: exact,
            bos_token_id: None,
            eos_token_id: None,
            pad_token_id: None,
            add_bos_token: false,
            add_eos_token: false,
            has_chat_template: false,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.tokens.len()
    }

    pub fn token_text(&self, id: u32) -> Option<&str> {
        self.tokens.get(id as usize).map(String::as_str)
    }

    pub fn is_eos(&self, id: u32) -> bool {
        self.eos_token_id == Some(id)
    }

    pub fn is_special(&self, id: u32) -> bool {
        self.token_types
            .get(id as usize)
            .map(|t| *t >= 3)
            .unwrap_or(false)
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        if self.add_bos_token {
            if let Some(bos) = self.bos_token_id {
                out.push(bos);
            }
        }
        out.extend(self.encode_raw(text));
        if self.add_eos_token {
            if let Some(eos) = self.eos_token_id {
                out.push(eos);
            }
        }
        out
    }

    pub fn encode_raw(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        for piece in self.pretokenize(text) {
            let mapped: String = piece.bytes().map(|b| self.byte_to_char[b as usize]).collect();
            let symbols: Vec<String> = mapped.chars().map(|c| c.to_string()).collect();
            for id in self.bpe(symbols) {
                out.push(id);
            }
        }
        out
    }

    /// Decode token ids to raw bytes. Control/user-defined tokens are emitted
    /// as their literal UTF-8 bytes; normal tokens use the byte-level alphabet.
    pub fn decode_bytes(&self, ids: &[u32]) -> Vec<u8> {
        let mut bytes = Vec::new();
        for &id in ids {
            let Some(token) = self.tokens.get(id as usize) else {
                continue;
            };
            if self.is_special(id) {
                bytes.extend_from_slice(token.as_bytes());
                continue;
            }
            for ch in token.chars() {
                if let Some(&b) = self.char_to_byte.get(&ch) {
                    bytes.push(b);
                } else {
                    bytes.extend_from_slice(ch.to_string().as_bytes());
                }
            }
        }
        bytes
    }

    /// Decode token ids to a lossy UTF-8 string.
    pub fn decode(&self, ids: &[u32]) -> String {
        String::from_utf8_lossy(&self.decode_bytes(ids)).into_owned()
    }

    fn bpe(&self, mut symbols: Vec<String>) -> Vec<u32> {
        while symbols.len() > 1 {
            let mut best: Option<(u32, String, String)> = None;
            for i in 0..symbols.len() - 1 {
                if let Some(&rank) = self.merges.get(&(symbols[i].clone(), symbols[i + 1].clone())) {
                    if best.as_ref().map(|(r, _, _)| rank < *r).unwrap_or(true) {
                        best = Some((rank, symbols[i].clone(), symbols[i + 1].clone()));
                    }
                }
            }
            let Some((_, a, b)) = best else { break };
            let mut merged = Vec::with_capacity(symbols.len());
            let mut k = 0;
            while k < symbols.len() {
                if k + 1 < symbols.len() && symbols[k] == a && symbols[k + 1] == b {
                    merged.push(format!("{a}{b}"));
                    k += 2;
                } else {
                    merged.push(symbols[k].clone());
                    k += 1;
                }
            }
            symbols = merged;
        }
        symbols
            .into_iter()
            .filter_map(|s| {
                self.token_to_id.get(&s).copied().or_else(|| {
                    // Byte-level fallback: all 256 single bytes should exist.
                    s.chars()
                        .next()
                        .and_then(|c| self.token_to_id.get(&c.to_string()).copied())
                })
            })
            .collect()
    }

    /// A Unicode-aware approximation of the GPT-2 pre-tokenizer. It handles
    /// contractions, optional leading spaces, letter/number runs, punctuation,
    /// and whitespace. It does not claim compatibility with unknown custom
    /// pre-tokenizers.
    fn pretokenize<'a>(&self, text: &'a str) -> Vec<&'a str> {
        let mut out = Vec::new();
        let mut i = 0usize;
        let n = text.len();
        while i < n {
            let ch = text[i..].chars().next().unwrap();

            // GPT-2 contractions are separate pieces.
            if ch == '\'' {
                let rest = &text[i..];
                let lower = rest.to_ascii_lowercase();
                let mut matched = false;
                for c in ["'s", "'t", "'re", "'ve", "'m", "'ll", "'d"] {
                    if lower.starts_with(c) {
                        out.push(&text[i..i + c.len()]);
                        i += c.len();
                        matched = true;
                        break;
                    }
                }
                if matched {
                    continue;
                }
            }

            let token_start = i;
            let mut content_start = i;
            if ch == ' ' && i + 1 < n {
                let next = text[i + 1..].chars().next().unwrap();
                if !next.is_whitespace() {
                    content_start = i + 1;
                }
            }

            if content_start == i && ch.is_whitespace() {
                let mut j = i + ch.len_utf8();
                while j < n {
                    let c = text[j..].chars().next().unwrap();
                    if c.is_whitespace() {
                        j += c.len_utf8();
                    } else {
                        break;
                    }
                }
                if j < n && ch == ' ' {
                    // A run of spaces before a word leaves exactly one space
                    // for the next piece. Other whitespace (newline, tab) is
                    // emitted here; it never attaches to a word.
                    let mut cut = j - text[..j].chars().last().unwrap().len_utf8();
                    if cut > i {
                        out.push(&text[i..cut]);
                    }
                    if cut == i {
                        out.push(&text[i..i + ch.len_utf8()]);
                        cut = i + ch.len_utf8();
                    }
                    i = cut;
                } else {
                    out.push(&text[i..j]);
                    i = j;
                }
                continue;
            }

            if content_start >= n {
                out.push(&text[token_start..]);
                break;
            }
            let first = text[content_start..].chars().next().unwrap();
            if first.is_alphabetic() {
                let mut j = content_start + first.len_utf8();
                while j < n {
                    let c = text[j..].chars().next().unwrap();
                    if c.is_alphabetic() {
                        j += c.len_utf8();
                    } else {
                        break;
                    }
                }
                out.push(&text[token_start..j]);
                i = j;
                continue;
            }
            if first.is_numeric() {
                let mut j = content_start + first.len_utf8();
                while j < n {
                    let c = text[j..].chars().next().unwrap();
                    if c.is_numeric() {
                        j += c.len_utf8();
                    } else {
                        break;
                    }
                }
                out.push(&text[token_start..j]);
                i = j;
                continue;
            }
            if !first.is_whitespace() {
                let mut j = content_start + first.len_utf8();
                while j < n {
                    let c = text[j..].chars().next().unwrap();
                    if !c.is_whitespace() && !c.is_alphanumeric() {
                        j += c.len_utf8();
                    } else {
                        break;
                    }
                }
                out.push(&text[token_start..j]);
                i = j;
                continue;
            }

            // Defensive fallback for an exotic whitespace character.
            out.push(&text[i..i + ch.len_utf8()]);
            i += ch.len_utf8();
        }
        out
    }
}

fn string_array(metadata: &Metadata, key: &str) -> Result<Vec<String>, TokenizerError> {
    let values = metadata
        .get_array(key)
        .ok_or_else(|| TokenizerError::MissingKey(key.to_string()))?;
    let mut out = Vec::with_capacity(values.len());
    for v in values {
        match v {
            MetadataValue::String(s) => out.push(s.clone()),
            _ => {
                return Err(TokenizerError::MissingKey(format!(
                    "{key} element is not a string"
                )))
            }
        }
    }
    Ok(out)
}

fn byte_alphabet() -> ([char; 256], HashMap<char, u8>) {
    let mut bs: Vec<u8> = (b'!'..=b'~').chain(0xA1..=0xAC).chain(0xAE..=0xFF).collect();
    let mut cs: Vec<u32> = bs.iter().map(|&b| b as u32).collect();
    let mut n = 0u32;
    for b in 0..=255u8 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    let mut byte_to_char = ['\0'; 256];
    let mut char_to_byte = HashMap::new();
    for (b, c) in bs.iter().zip(cs.iter()) {
        let ch = char::from_u32(*c).unwrap();
        byte_to_char[*b as usize] = ch;
        char_to_byte.insert(ch, *b);
    }
    (byte_to_char, char_to_byte)
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ChatRole {
    System,
    User,
    Assistant,
}

impl ChatRole {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
        }
    }
}

#[derive(Clone, Debug)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: String,
}

/// DeepSeek-V4.1 text chat encoding for non-thinking chat mode, based on the
/// published encoding reference. Tool calls and images are not handled here.
pub fn format_chat_deepseek_v41(messages: &[ChatMessage], add_generation_prompt: bool) -> String {
    const BOS: &str = "<｜begin▁of▁sentence｜>";
    const USER: &str = "<｜User｜>";
    const ASSISTANT: &str = "<｜Assistant｜>";
    const SYSTEM: &str = "<｜System｜>";
    const EOS: &str = "<｜end▁of▁sentence｜>";
    let mut out = String::from(BOS);
    let mut first = true;
    for message in messages {
        match message.role {
            ChatRole::System => {
                if first {
                    out.push_str(&message.content);
                    out.push_str("\n\n");
                } else {
                    out.push_str(SYSTEM);
                    out.push_str(&message.content);
                    out.push_str("\n\n");
                }
            }
            ChatRole::User => {
                out.push_str(USER);
                out.push_str(&message.content);
            }
            ChatRole::Assistant => {
                out.push_str(ASSISTANT);
                out.push_str("</think>");
                out.push_str(&message.content);
                out.push_str(EOS);
            }
        }
        first = false;
    }
    if add_generation_prompt {
        out.push_str(ASSISTANT);
        out.push_str("</think>");
    }
    out
}

/// A transparent, non-official fallback chat format. The real DeepSeek-V4.1
/// chat template is a Jinja program with DSML tool syntax and thinking modes;
/// this fallback must not be presented as equivalent.
pub fn format_chat_simple(messages: &[ChatMessage], add_generation_prompt: bool) -> String {
    let mut out = String::new();
    for m in messages {
        out.push_str(m.role.as_str());
        out.push_str(": ");
        out.push_str(&m.content);
        out.push('\n');
    }
    if add_generation_prompt {
        out.push_str("assistant: ");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_tokenizer() -> BpeTokenizer {
        let mut tokens = vec![
            "h".to_string(),
            "e".to_string(),
            "l".to_string(),
            "o".to_string(),
            "he".to_string(),
            "hel".to_string(),
            "hell".to_string(),
            "hello".to_string(),
        ];
        // Include every single-byte token so the fallback is never needed.
        let (byte_to_char, _) = byte_alphabet();
        for b in 0..=255u8 {
            let s = byte_to_char[b as usize].to_string();
            if !tokens.contains(&s) {
                tokens.push(s);
            }
        }
        let types = vec![1u32; tokens.len()];
        BpeTokenizer::from_parts(
            tokens,
            types,
            vec!["h e".into(), "he l".into(), "hel l".into(), "hell o".into()],
            "gpt2",
        )
        .unwrap()
    }

    #[test]
    fn merges_and_decodes_ascii() {
        let t = tiny_tokenizer();
        let ids = t.encode("hello");
        assert_eq!(t.decode(&ids), "hello");
        assert_eq!(ids, vec![7]);
    }

    #[test]
    fn byte_roundtrip_utf8() {
        let t = tiny_tokenizer();
        let text = "héllo";
        let ids = t.encode(text);
        assert_eq!(t.decode(&ids), text);
    }

    #[test]
    fn special_tokens_decode_literally() {
        let mut t = tiny_tokenizer();
        let id = t.tokens.len() as u32;
        t.tokens.push("<|end|>".into());
        t.token_types.push(3);
        t.token_to_id.insert("<|end|>".into(), id);
        assert_eq!(t.decode(&[id]), "<|end|>");
        assert!(t.is_special(id));
    }

    #[test]
    fn pretokenizer_makes_progress_on_newlines() {
        let t = tiny_tokenizer();
        let ids = t.encode("user: Hi\nassistant: ");
        assert!(!ids.is_empty());
        assert_eq!(t.decode(&ids), "user: Hi\nassistant: ");
    }

    #[test]
    fn deepseek_chat_encoding_closes_thinking() {
        let prompt = format_chat_deepseek_v41(
            &[ChatMessage {
                role: ChatRole::User,
                content: "Hi".into(),
            }],
            true,
        );
        assert_eq!(
            prompt,
            "<｜begin▁of▁sentence｜><｜User｜>Hi<｜Assistant｜></think>"
        );
    }

    #[test]
    fn chat_fallback_is_explicit() {
        let s = format_chat_simple(
            &[ChatMessage {
                role: ChatRole::User,
                content: "hi".into(),
            }],
            true,
        );
        assert_eq!(s, "user: hi\nassistant: ");
    }
}
