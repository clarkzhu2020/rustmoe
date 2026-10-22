//! A strict, streaming GGUF v2/v3 parser.
//!
//! The parser reads only metadata and the tensor directory. It never maps or
//! reads tensor payloads. This makes a 700 GB split model inspectable with a
//! few megabytes of I/O.

mod discover;

pub use discover::{discover_split_paths, sniff_magic};

use rustmoe_quant::GgmlType;
use serde::Serialize;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use thiserror::Error;

pub const GGUF_MAGIC: [u8; 4] = *b"GGUF";
pub const DEFAULT_ALIGNMENT: u64 = 32;

#[derive(Clone, Debug)]
pub struct ParseOptions {
    pub max_header_bytes: u64,
    pub max_metadata_entries: u64,
    pub max_tensors: u64,
    pub max_string_bytes: u64,
    pub max_array_elements: u64,
    pub max_dims: u32,
}

impl Default for ParseOptions {
    fn default() -> Self {
        Self {
            max_header_bytes: 256 * 1024 * 1024,
            max_metadata_entries: 100_000,
            max_tensors: 1_000_000,
            max_string_bytes: 64 * 1024 * 1024,
            max_array_elements: 4_000_000,
            max_dims: 8,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
#[serde(untagged)]
pub enum MetadataValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    Array(Vec<MetadataValue>),
    U64(u64),
    I64(i64),
    F64(f64),
}

impl MetadataValue {
    pub fn type_name(&self) -> &'static str {
        match self {
            Self::U8(_) => "u8",
            Self::I8(_) => "i8",
            Self::U16(_) => "u16",
            Self::I16(_) => "i16",
            Self::U32(_) => "u32",
            Self::I32(_) => "i32",
            Self::F32(_) => "f32",
            Self::Bool(_) => "bool",
            Self::String(_) => "string",
            Self::Array(_) => "array",
            Self::U64(_) => "u64",
            Self::I64(_) => "i64",
            Self::F64(_) => "f64",
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Self::U8(v) => Some(*v as u64),
            Self::U16(v) => Some(*v as u64),
            Self::U32(v) => Some(*v as u64),
            Self::U64(v) => Some(*v),
            Self::I8(v) if *v >= 0 => Some(*v as u64),
            Self::I16(v) if *v >= 0 => Some(*v as u64),
            Self::I32(v) if *v >= 0 => Some(*v as u64),
            Self::I64(v) if *v >= 0 => Some(*v as u64),
            _ => None,
        }
    }

    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Self::U8(v) => Some(*v as i64),
            Self::I8(v) => Some(*v as i64),
            Self::U16(v) => Some(*v as i64),
            Self::I16(v) => Some(*v as i64),
            Self::U32(v) => Some(*v as i64),
            Self::I32(v) => Some(*v as i64),
            Self::U64(v) => i64::try_from(*v).ok(),
            Self::I64(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Self::F32(v) => Some(*v as f64),
            Self::F64(v) => Some(*v),
            Self::U8(v) => Some(*v as f64),
            Self::I8(v) => Some(*v as f64),
            Self::U16(v) => Some(*v as f64),
            Self::I16(v) => Some(*v as f64),
            Self::U32(v) => Some(*v as f64),
            Self::I32(v) => Some(*v as f64),
            Self::U64(v) => Some(*v as f64),
            Self::I64(v) => Some(*v as f64),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(v) => Some(*v),
            Self::U8(v) => Some(*v != 0),
            Self::I8(v) => Some(*v != 0),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[MetadataValue]> {
        match self {
            Self::Array(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_u32_array(&self) -> Option<Vec<u32>> {
        self.as_array()?
            .iter()
            .map(|v| v.as_u64().and_then(|n| u32::try_from(n).ok()))
            .collect()
    }

    pub fn as_u64_array(&self) -> Option<Vec<u64>> {
        self.as_array()?.iter().map(MetadataValue::as_u64).collect()
    }

    pub fn as_f32_array(&self) -> Option<Vec<f32>> {
        self.as_array()?
            .iter()
            .map(|v| v.as_f64().map(|n| n as f32))
            .collect()
    }
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct Metadata {
    entries: BTreeMap<String, MetadataValue>,
}

impl Metadata {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_entries(entries: impl IntoIterator<Item = (String, MetadataValue)>) -> Self {
        Self {
            entries: entries.into_iter().collect(),
        }
    }

    /// Insert a value. Returns the previous value when the key existed.
    pub fn insert(&mut self, key: impl Into<String>, value: MetadataValue) -> Option<MetadataValue> {
        self.entries.insert(key.into(), value)
    }

    pub fn get(&self, key: &str) -> Option<&MetadataValue> {
        self.entries.get(key)
    }

    pub fn contains_key(&self, key: &str) -> bool {
        self.entries.contains_key(key)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&String, &MetadataValue)> {
        self.entries.iter()
    }

    pub fn get_u64(&self, key: &str) -> Option<u64> {
        self.get(key).and_then(MetadataValue::as_u64)
    }

    pub fn get_u32(&self, key: &str) -> Option<u32> {
        self.get_u64(key).and_then(|v| u32::try_from(v).ok())
    }

    pub fn get_i64(&self, key: &str) -> Option<i64> {
        self.get(key).and_then(MetadataValue::as_i64)
    }

    pub fn get_f32(&self, key: &str) -> Option<f32> {
        self.get(key).and_then(MetadataValue::as_f64).map(|v| v as f32)
    }

    pub fn get_bool(&self, key: &str) -> Option<bool> {
        self.get(key).and_then(MetadataValue::as_bool)
    }

    pub fn get_str(&self, key: &str) -> Option<&str> {
        self.get(key).and_then(MetadataValue::as_str)
    }

    pub fn get_array(&self, key: &str) -> Option<&[MetadataValue]> {
        self.get(key).and_then(MetadataValue::as_array)
    }

    pub fn get_u32_array(&self, key: &str) -> Option<Vec<u32>> {
        self.get(key).and_then(MetadataValue::as_u32_array)
    }

    pub fn get_u64_array(&self, key: &str) -> Option<Vec<u64>> {
        self.get(key).and_then(MetadataValue::as_u64_array)
    }

    pub fn get_f32_array(&self, key: &str) -> Option<Vec<f32>> {
        self.get(key).and_then(MetadataValue::as_f32_array)
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct SplitInfo {
    /// Zero-based shard index stored in the file.
    pub no: u32,
    pub count: u32,
    pub tensor_count: Option<u32>,
}

#[derive(Clone, Debug, Serialize)]
pub struct TensorInfo {
    pub name: String,
    pub dims: Vec<u64>,
    pub ggml_type: GgmlType,
    /// Offset relative to this shard's data section.
    pub offset: u64,
    /// Uncompressed element count.
    pub elements: u64,
    /// Stored byte size, if the GGML type is known.
    pub nbytes: Option<u64>,
    pub shard: usize,
    pub data_offset: u64,
}

impl TensorInfo {
    pub fn absolute_offset(&self) -> Option<u64> {
        self.data_offset.checked_add(self.offset)
    }

    pub fn end(&self) -> Option<u64> {
        self.absolute_offset()?.checked_add(self.nbytes?)
    }
}

#[derive(Clone, Debug)]
pub struct GgufFile {
    pub path: PathBuf,
    pub version: u32,
    pub metadata: Metadata,
    pub tensors: Vec<TensorInfo>,
    pub data_offset: u64,
    pub header_bytes: u64,
    pub file_len: u64,
    pub alignment: u64,
    pub split: Option<SplitInfo>,
}

impl GgufFile {
    pub fn parse_path(path: impl AsRef<Path>, options: &ParseOptions) -> Result<Self, GgufError> {
        let path = path.as_ref();
        let mut file = File::open(path)?;
        let file_len = file.metadata()?.len();
        file.seek(SeekFrom::Start(0))?;
        Self::parse_reader(file, file_len, path.to_path_buf(), options)
    }

    pub fn parse_reader<R: Read>(
        reader: R,
        file_len: u64,
        path: PathBuf,
        options: &ParseOptions,
    ) -> Result<Self, GgufError> {
        let mut r = HeaderReader::new(reader, options.max_header_bytes.min(file_len.max(1)));
        let magic = r.bytes(4)?;
        if magic != GGUF_MAGIC {
            return Err(GgufError::InvalidMagic);
        }
        let version = r.u32()?;
        if version != 2 && version != 3 {
            return Err(GgufError::UnsupportedVersion(version));
        }
        let tensor_count = r.u64()?;
        let metadata_count = r.u64()?;
        if tensor_count > options.max_tensors {
            return Err(GgufError::LimitExceeded {
                what: "tensor count",
                value: tensor_count,
            });
        }
        if metadata_count > options.max_metadata_entries {
            return Err(GgufError::LimitExceeded {
                what: "metadata entry count",
                value: metadata_count,
            });
        }

        let mut metadata = Metadata::default();
        for _ in 0..metadata_count {
            let key = r.string(options.max_string_bytes)?;
            let value = r.metadata_value(options, 0)?;
            if metadata.entries.insert(key.clone(), value).is_some() {
                return Err(GgufError::DuplicateMetadata(key));
            }
        }

        let alignment = match metadata.get_u64("general.alignment") {
            Some(n) => {
                if n == 0 || !n.is_power_of_two() || n > 1024 * 1024 {
                    return Err(GgufError::InvalidAlignment(n));
                }
                n
            }
            None => DEFAULT_ALIGNMENT,
        };

        let mut tensors = Vec::new();
        tensors
            .try_reserve_exact(tensor_count as usize)
            .map_err(|_| GgufError::LimitExceeded {
                what: "tensor allocation",
                value: tensor_count,
            })?;
        for _ in 0..tensor_count {
            let name = r.string(options.max_string_bytes)?;
            let n_dims = r.u32()?;
            if n_dims == 0 || n_dims > options.max_dims {
                return Err(GgufError::InvalidDimensions {
                    tensor: name,
                    dims: n_dims,
                });
            }
            let mut dims = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                let d = r.u64()?;
                if d == 0 {
                    return Err(GgufError::ZeroDimension { tensor: name });
                }
                dims.push(d);
            }
            let type_id = r.u32()?;
            let ggml_type = GgmlType(type_id);
            let offset = r.u64()?;
            let elements = dims
                .iter()
                .try_fold(1u64, |acc, &d| acc.checked_mul(d))
                .ok_or_else(|| GgufError::TensorTooLarge { tensor: name.clone() })?;
            let nbytes = if ggml_type.is_known() {
                Some(
                    ggml_type
                        .tensor_size(&dims)
                        .map_err(|source| GgufError::Quantization {
                            tensor: name.clone(),
                            source,
                        })?,
                )
            } else {
                None
            };
            if offset % alignment != 0 {
                return Err(GgufError::MisalignedTensor {
                    tensor: name,
                    offset,
                    alignment,
                });
            }
            tensors.push(TensorInfo {
                name,
                dims,
                ggml_type,
                offset,
                elements,
                nbytes,
                shard: 0,
                data_offset: 0,
            });
        }

        let header_end = r.position();
        let data_offset = align_up(header_end, alignment);
        if !tensors.is_empty() && data_offset > file_len {
            return Err(GgufError::HeaderPastEnd {
                header_end,
                data_offset,
                file_len,
            });
        }
        for t in &mut tensors {
            t.data_offset = data_offset;
            if let (Some(_nbytes), Some(end)) = (t.nbytes, t.end()) {
                if end > file_len {
                    return Err(GgufError::TensorPastEnd {
                        tensor: t.name.clone(),
                        end,
                        file_len,
                    });
                }
            }
        }

        let split = match (metadata.get_u64("split.no"), metadata.get_u64("split.count")) {
            (Some(no), Some(count)) => {
                if count == 0 || no >= count {
                    return Err(GgufError::InvalidSplit {
                        no: no as u32,
                        count: count as u32,
                    });
                }
                let tensor_count = metadata
                    .get_u64("split.tensors.count")
                    .and_then(|v| u32::try_from(v).ok());
                Some(SplitInfo {
                    no: no as u32,
                    count: count as u32,
                    tensor_count,
                })
            }
            (None, None) => None,
            _ => return Err(GgufError::IncompleteSplitMetadata),
        };

        Ok(Self {
            path,
            version,
            metadata,
            tensors,
            data_offset,
            header_bytes: header_end,
            file_len,
            alignment,
            split,
        })
    }

    pub fn architecture(&self) -> Option<&str> {
        self.metadata.get_str("general.architecture")
    }
}

/// A complete, validated split model. Tensor payloads are still untouched.
#[derive(Clone, Debug)]
pub struct GgufSet {
    pub files: Vec<GgufFile>,
    pub primary: usize,
    pub metadata: Metadata,
    pub tensors: Vec<TensorInfo>,
    pub total_file_bytes: u64,
    pub total_tensor_bytes: u64,
}

impl GgufSet {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, GgufError> {
        Self::open_with_options(path, &ParseOptions::default())
    }

    pub fn open_with_options(path: impl AsRef<Path>, options: &ParseOptions) -> Result<Self, GgufError> {
        let path = path.as_ref();
        let paths = if path.is_dir() {
            discover::paths_in_directory(path)?
        } else {
            discover::discover_split_paths(path)?
        };
        if paths.is_empty() {
            return Err(GgufError::NoShards(path.to_path_buf()));
        }
        let mut files = Vec::with_capacity(paths.len());
        for p in &paths {
            let mut f = GgufFile::parse_path(p, options)?;
            f.tensors.shrink_to_fit();
            files.push(f);
        }
        Self::from_files(files)
    }

    pub fn from_files(mut files: Vec<GgufFile>) -> Result<Self, GgufError> {
        if files.is_empty() {
            return Err(GgufError::NoShards(PathBuf::new()));
        }
        let expected_count = files
            .iter()
            .find_map(|f| f.split.as_ref().map(|s| s.count))
            .unwrap_or(1);
        for f in &files {
            if let Some(s) = &f.split {
                if s.count != expected_count {
                    return Err(GgufError::SplitCountMismatch {
                        path: f.path.clone(),
                        expected: expected_count,
                        actual: s.count,
                    });
                }
            } else if expected_count != 1 {
                return Err(GgufError::MissingSplitMetadata(f.path.clone()));
            }
        }
        files.sort_by_key(|f| f.split.as_ref().map(|s| s.no).unwrap_or(0));
        if files.len() as u32 != expected_count {
            return Err(GgufError::SplitShardCount {
                expected: expected_count,
                actual: files.len() as u32,
            });
        }
        for (i, f) in files.iter().enumerate() {
            let no = f.split.as_ref().map(|s| s.no).unwrap_or(0);
            if no != i as u32 {
                return Err(GgufError::SplitIndexGap {
                    expected: i as u32,
                    actual: no,
                });
            }
        }

        let primary = files
            .iter()
            .enumerate()
            .max_by_key(|(_, f)| f.metadata.len())
            .map(|(i, _)| i)
            .unwrap_or(0);
        let metadata = files[primary].metadata.clone();

        let mut tensors = Vec::new();
        for (shard, f) in files.iter().enumerate() {
            let mut local = f.tensors.clone();
            for t in &mut local {
                t.shard = shard;
            }
            tensors.extend(local);
        }
        tensors.sort_by(|a, b| a.name.cmp(&b.name));
        for pair in tensors.windows(2) {
            if pair[0].name == pair[1].name {
                return Err(GgufError::DuplicateTensor(pair[0].name.clone()));
            }
        }
        if let Some(expected) = metadata.get_u64("split.tensors.count") {
            if expected != tensors.len() as u64 {
                return Err(GgufError::TensorCountMismatch {
                    expected,
                    actual: tensors.len() as u64,
                });
            }
        }
        let total_tensor_bytes = tensors
            .iter()
            .try_fold(0u64, |acc, t| acc.checked_add(t.nbytes.unwrap_or(0)))
            .ok_or(GgufError::SizeOverflow)?;
        let total_file_bytes = files
            .iter()
            .try_fold(0u64, |acc, f| acc.checked_add(f.file_len))
            .ok_or(GgufError::SizeOverflow)?;

        Ok(Self {
            files,
            primary,
            metadata,
            tensors,
            total_file_bytes,
            total_tensor_bytes,
        })
    }

    pub fn architecture(&self) -> Option<&str> {
        self.metadata.get_str("general.architecture")
    }

    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors
            .binary_search_by(|t| t.name.as_str().cmp(name))
            .ok()
            .map(|i| &self.tensors[i])
    }

    pub fn tensor_bytes_by_type(&self) -> BTreeMap<String, TypeAggregate> {
        let mut out: BTreeMap<String, TypeAggregate> = BTreeMap::new();
        for t in &self.tensors {
            let e = out.entry(t.ggml_type.name().to_string()).or_default();
            e.tensors += 1;
            e.bytes = e.bytes.saturating_add(t.nbytes.unwrap_or(0));
            e.elements = e.elements.saturating_add(t.elements);
        }
        out
    }
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct TypeAggregate {
    pub tensors: u64,
    pub bytes: u64,
    pub elements: u64,
}

pub fn align_up(value: u64, alignment: u64) -> u64 {
    if alignment <= 1 {
        return value;
    }
    let rem = value % alignment;
    if rem == 0 {
        value
    } else {
        value.saturating_add(alignment - rem)
    }
}

struct HeaderReader<R> {
    inner: R,
    pos: u64,
    limit: u64,
}

impl<R: Read> HeaderReader<R> {
    fn new(inner: R, limit: u64) -> Self {
        Self { inner, pos: 0, limit }
    }

    fn position(&self) -> u64 {
        self.pos
    }

    fn check(&self, n: u64) -> Result<(), GgufError> {
        let end = self.pos.checked_add(n).ok_or(GgufError::SizeOverflow)?;
        if end > self.limit {
            return Err(GgufError::HeaderLimit {
                position: self.pos,
                requested: n,
                limit: self.limit,
            });
        }
        Ok(())
    }

    fn bytes(&mut self, n: usize) -> Result<Vec<u8>, GgufError> {
        self.check(n as u64)?;
        let mut buf = vec![0u8; n];
        self.inner.read_exact(&mut buf)?;
        self.pos += n as u64;
        Ok(buf)
    }

    fn u8(&mut self) -> Result<u8, GgufError> {
        Ok(self.bytes(1)?[0])
    }

    fn u16(&mut self) -> Result<u16, GgufError> {
        let b = self.bytes(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }

    fn u32(&mut self) -> Result<u32, GgufError> {
        let b = self.bytes(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn u64(&mut self) -> Result<u64, GgufError> {
        let b = self.bytes(8)?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    fn string(&mut self, max: u64) -> Result<String, GgufError> {
        let len = self.u64()?;
        if len > max {
            return Err(GgufError::LimitExceeded {
                what: "string length",
                value: len,
            });
        }
        let bytes = self.bytes(len as usize)?;
        String::from_utf8(bytes).map_err(|_| GgufError::InvalidUtf8)
    }

    fn metadata_value(&mut self, options: &ParseOptions, depth: u32) -> Result<MetadataValue, GgufError> {
        if depth > 2 {
            return Err(GgufError::NestedArray);
        }
        let t = self.u32()?;
        let value = match t {
            0 => MetadataValue::U8(self.u8()?),
            1 => MetadataValue::I8(self.u8()? as i8),
            2 => MetadataValue::U16(self.u16()?),
            3 => MetadataValue::I16(self.u16()? as i16),
            4 => MetadataValue::U32(self.u32()?),
            5 => MetadataValue::I32(self.u32()? as i32),
            6 => MetadataValue::F32(f32::from_bits(self.u32()?)),
            7 => MetadataValue::Bool(self.u8()? != 0),
            8 => MetadataValue::String(self.string(options.max_string_bytes)?),
            9 => {
                let element_type = self.u32()?;
                let count = self.u64()?;
                if count > options.max_array_elements {
                    return Err(GgufError::LimitExceeded {
                        what: "array length",
                        value: count,
                    });
                }
                if element_type == 9 {
                    return Err(GgufError::NestedArray);
                }
                let mut values = Vec::new();
                values
                    .try_reserve_exact(count as usize)
                    .map_err(|_| GgufError::LimitExceeded {
                        what: "array allocation",
                        value: count,
                    })?;
                for _ in 0..count {
                    values.push(self.scalar_value(element_type, options)?);
                }
                MetadataValue::Array(values)
            }
            10 => MetadataValue::U64(self.u64()?),
            11 => MetadataValue::I64(self.u64()? as i64),
            12 => MetadataValue::F64(f64::from_bits(self.u64()?)),
            other => return Err(GgufError::UnknownMetadataType(other)),
        };
        Ok(value)
    }

    fn scalar_value(&mut self, t: u32, options: &ParseOptions) -> Result<MetadataValue, GgufError> {
        let value = match t {
            0 => MetadataValue::U8(self.u8()?),
            1 => MetadataValue::I8(self.u8()? as i8),
            2 => MetadataValue::U16(self.u16()?),
            3 => MetadataValue::I16(self.u16()? as i16),
            4 => MetadataValue::U32(self.u32()?),
            5 => MetadataValue::I32(self.u32()? as i32),
            6 => MetadataValue::F32(f32::from_bits(self.u32()?)),
            7 => MetadataValue::Bool(self.u8()? != 0),
            8 => MetadataValue::String(self.string(options.max_string_bytes)?),
            10 => MetadataValue::U64(self.u64()?),
            11 => MetadataValue::I64(self.u64()? as i64),
            12 => MetadataValue::F64(f64::from_bits(self.u64()?)),
            other => return Err(GgufError::UnknownMetadataType(other)),
        };
        Ok(value)
    }
}

#[derive(Debug, Error)]
pub enum GgufError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("not a GGUF file (bad magic)")]
    InvalidMagic,
    #[error("unsupported GGUF version {0}; RustMoE supports v2 and v3")]
    UnsupportedVersion(u32),
    #[error("{what} limit exceeded: {value}")]
    LimitExceeded { what: &'static str, value: u64 },
    #[error("header read at {position} with {requested} bytes exceeds limit {limit}")]
    HeaderLimit {
        position: u64,
        requested: u64,
        limit: u64,
    },
    #[error("invalid UTF-8 in GGUF metadata")]
    InvalidUtf8,
    #[error("unknown GGUF metadata value type {0}")]
    UnknownMetadataType(u32),
    #[error("nested metadata arrays are not supported")]
    NestedArray,
    #[error("duplicate metadata key: {0}")]
    DuplicateMetadata(String),
    #[error("invalid general.alignment value {0}")]
    InvalidAlignment(u64),
    #[error("tensor {tensor} has invalid dimensional count {dims}")]
    InvalidDimensions { tensor: String, dims: u32 },
    #[error("tensor {tensor} has a zero dimension")]
    ZeroDimension { tensor: String },
    #[error("tensor {tensor} element count overflows u64")]
    TensorTooLarge { tensor: String },
    #[error("tensor {tensor} quantization error: {source}")]
    Quantization {
        tensor: String,
        #[source]
        source: rustmoe_quant::QuantError,
    },
    #[error("tensor {tensor} offset {offset} is not aligned to {alignment}")]
    MisalignedTensor {
        tensor: String,
        offset: u64,
        alignment: u64,
    },
    #[error("header end {header_end} and aligned data start {data_offset} exceed file size {file_len}")]
    HeaderPastEnd {
        header_end: u64,
        data_offset: u64,
        file_len: u64,
    },
    #[error("tensor {tensor} ends at byte {end}, past file size {file_len}")]
    TensorPastEnd { tensor: String, end: u64, file_len: u64 },
    #[error("incomplete split metadata: split.no and split.count must both be present")]
    IncompleteSplitMetadata,
    #[error("invalid split index {no} of {count}")]
    InvalidSplit { no: u32, count: u32 },
    #[error("no GGUF shards found at {0}")]
    NoShards(PathBuf),
    #[error("split count mismatch in {path}: expected {expected}, got {actual}")]
    SplitCountMismatch {
        path: PathBuf,
        expected: u32,
        actual: u32,
    },
    #[error("split shard {0} has no split.no/split.count metadata")]
    MissingSplitMetadata(PathBuf),
    #[error("expected {expected} split shards, found {actual}")]
    SplitShardCount { expected: u32, actual: u32 },
    #[error("split index gap: expected {expected}, found {actual}")]
    SplitIndexGap { expected: u32, actual: u32 },
    #[error("duplicate tensor across shards: {0}")]
    DuplicateTensor(String),
    #[error("split.tensors.count says {expected}, found {actual}")]
    TensorCountMismatch { expected: u64, actual: u64 },
    #[error("tensor byte size overflow")]
    SizeOverflow,
    #[error("split filename error: {0}")]
    SplitFilename(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn align(v: usize, a: usize) -> usize {
        v.div_ceil(a) * a
    }

    fn put_u32(out: &mut Vec<u8>, v: u32) {
        out.extend_from_slice(&v.to_le_bytes());
    }
    fn put_u64(out: &mut Vec<u8>, v: u64) {
        out.extend_from_slice(&v.to_le_bytes());
    }
    fn put_str(out: &mut Vec<u8>, s: &str) {
        put_u64(out, s.len() as u64);
        out.extend_from_slice(s.as_bytes());
    }

    fn sample_gguf() -> Vec<u8> {
        let mut header = Vec::new();
        header.extend_from_slice(b"GGUF");
        put_u32(&mut header, 3);
        put_u64(&mut header, 1);
        put_u64(&mut header, 4);
        put_str(&mut header, "general.architecture");
        put_u32(&mut header, 8);
        put_str(&mut header, "deepseek41");
        put_str(&mut header, "general.alignment");
        put_u32(&mut header, 4);
        put_u32(&mut header, 32);
        put_str(&mut header, "demo.flag");
        put_u32(&mut header, 7);
        header.push(1);
        put_str(&mut header, "demo.ints");
        put_u32(&mut header, 9);
        put_u32(&mut header, 4);
        put_u64(&mut header, 3);
        put_u32(&mut header, 1);
        put_u32(&mut header, 2);
        put_u32(&mut header, 3);

        put_str(&mut header, "blk.0.demo.weight");
        put_u32(&mut header, 1);
        put_u64(&mut header, 32);
        put_u32(&mut header, 8); // Q8_0
        put_u64(&mut header, 0);
        let data_offset = align(header.len(), 32);
        header.resize(data_offset, 0);
        header.extend(std::iter::repeat_n(0u8, 34));
        header
    }

    #[test]
    fn parses_metadata_and_tensor_directory() {
        let bytes = sample_gguf();
        let len = bytes.len() as u64;
        let parsed = GgufFile::parse_reader(
            Cursor::new(bytes),
            len,
            PathBuf::from("fixture.gguf"),
            &ParseOptions::default(),
        )
        .unwrap();
        assert_eq!(parsed.architecture(), Some("deepseek41"));
        assert_eq!(parsed.metadata.get_bool("demo.flag"), Some(true));
        assert_eq!(parsed.metadata.get_u32_array("demo.ints"), Some(vec![1, 2, 3]));
        assert_eq!(parsed.tensors.len(), 1);
        let t = &parsed.tensors[0];
        assert_eq!(t.nbytes, Some(34));
        assert_eq!(t.end(), Some(len));
        assert_eq!(parsed.alignment, 32);
    }

    #[test]
    fn rejects_bad_magic_and_version() {
        let mut bytes = sample_gguf();
        bytes[0] = b'X';
        let len = bytes.len() as u64;
        let err = GgufFile::parse_reader(Cursor::new(bytes), len, PathBuf::new(), &ParseOptions::default())
            .unwrap_err();
        assert!(matches!(err, GgufError::InvalidMagic));

        let mut bytes = sample_gguf();
        bytes[4..8].copy_from_slice(&4u32.to_le_bytes());
        let len = bytes.len() as u64;
        let err = GgufFile::parse_reader(Cursor::new(bytes), len, PathBuf::new(), &ParseOptions::default())
            .unwrap_err();
        assert!(matches!(err, GgufError::UnsupportedVersion(4)));
    }

    #[test]
    fn rejects_tensor_past_end() {
        let mut bytes = sample_gguf();
        let len = (bytes.len() - 1) as u64;
        let err = GgufFile::parse_reader(
            Cursor::new(std::mem::take(&mut bytes)),
            len,
            PathBuf::new(),
            &ParseOptions::default(),
        )
        .unwrap_err();
        assert!(matches!(err, GgufError::TensorPastEnd { .. }));
    }

    #[test]
    fn split_discovery_parses_names() {
        let d = tempfile::tempdir().unwrap();
        let names = ["model-00001-of-00002.gguf", "model-00002-of-00002.gguf"];
        for n in names {
            std::fs::write(d.path().join(n), b"x").unwrap();
        }
        let found = discover_split_paths(&d.path().join(names[0])).unwrap();
        assert_eq!(found.len(), 2);
        assert!(found[0].ends_with(names[0]));
        assert!(found[1].ends_with(names[1]));
    }
}
