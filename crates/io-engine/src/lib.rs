//! Weight I/O for split GGUF files.
//!
//! Two explicit modes are supported:
//! - Pread: positioned reads into caller-owned buffers, with no page-fault
//!   surprises and no mapping of a 200 GB shard.
//! - Mmap: the OS page cache is allowed to decide residency.
//!
//! Expert tensors are 3-D and expert-major. The byte range for expert e is
//! base + e * bytes_per_expert. RustMoE validates that the tensor byte count
//! divides evenly by the expert count before using that formula.

use memmap2::{Mmap, MmapOptions};
use rustmoe_model::{ExpertLayout, ModelDescriptor};
use serde::Serialize;
use std::fs::File;
use std::io;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use thiserror::Error;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StoreMode {
    Pread,
    Mmap,
}

impl StoreMode {
    pub fn parse(value: &str) -> Option<Self> {
        match value.to_ascii_lowercase().as_str() {
            "pread" | "explicit" | "async" => Some(Self::Pread),
            "mmap" | "mapped" => Some(Self::Mmap),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct TensorBlob {
    pub tensor: String,
    pub shard: usize,
    pub offset: u64,
    pub len: u64,
}

impl TensorBlob {
    pub fn end(&self) -> u64 {
        self.offset.saturating_add(self.len)
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct ExpertLocation {
    pub layer: usize,
    pub expert: u32,
    pub gate: TensorBlob,
    pub down: TensorBlob,
    pub up: TensorBlob,
    pub total_bytes: u64,
}

impl ExpertLocation {
    pub fn from_layout(layout: &ExpertLayout, expert: u32) -> Result<Self, IoError> {
        if expert as u64 >= layout.expert_count {
            return Err(IoError::ExpertOutOfRange {
                layer: layout.layer,
                expert,
                count: layout.expert_count,
            });
        }
        let e = expert as u64;
        let gate = TensorBlob {
            tensor: format!("blk.{}.ffn_gate_exps.weight", layout.layer),
            shard: layout.gate_shard,
            offset: layout.gate_offset + e * layout.gate_bytes,
            len: layout.gate_bytes,
        };
        let down = TensorBlob {
            tensor: format!("blk.{}.ffn_down_exps.weight", layout.layer),
            shard: layout.down_shard,
            offset: layout.down_offset + e * layout.down_bytes,
            len: layout.down_bytes,
        };
        let up = TensorBlob {
            tensor: format!("blk.{}.ffn_up_exps.weight", layout.layer),
            shard: layout.up_shard,
            offset: layout.up_offset + e * layout.up_bytes,
            len: layout.up_bytes,
        };
        Ok(Self {
            layer: layout.layer,
            expert,
            total_bytes: gate.len + down.len + up.len,
            gate,
            down,
            up,
        })
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct StoreStats {
    pub reads: u64,
    pub bytes_read: u64,
    pub errors: u64,
}

#[derive(Default)]
struct AtomicStats {
    reads: AtomicU64,
    bytes_read: AtomicU64,
    errors: AtomicU64,
}

enum Backend {
    Pread(Vec<File>),
    Mmap(Vec<Mmap>),
}

/// Read-only access to the complete expert cold store.
pub struct WeightStore {
    model: Arc<ModelDescriptor>,
    mode: StoreMode,
    layouts: Vec<Option<ExpertLayout>>,
    backend: Backend,
    stats: AtomicStats,
    paths: Vec<PathBuf>,
}

impl WeightStore {
    pub fn open(model: Arc<ModelDescriptor>, mode: StoreMode) -> Result<Self, IoError> {
        let paths: Vec<PathBuf> = model.gguf.files.iter().map(|f| f.path.clone()).collect();
        if paths.is_empty() {
            return Err(IoError::NoShards);
        }
        let mut layouts = Vec::with_capacity(model.config.block_count);
        for layer in 0..model.config.block_count {
            let layout = model
                .expert_layout(layer)
                .ok_or(IoError::MissingExpertLayout(layer))?;
            layouts.push(Some(layout));
        }
        let backend = match mode {
            StoreMode::Pread => {
                let mut files = Vec::with_capacity(paths.len());
                for p in &paths {
                    files.push(File::open(p)?);
                }
                Backend::Pread(files)
            }
            StoreMode::Mmap => {
                let mut maps = Vec::with_capacity(paths.len());
                for p in &paths {
                    let file = File::open(p)?;
                    // SAFETY: the store is read-only by contract. Callers must
                    // not modify model files while a process is running.
                    let map = unsafe { MmapOptions::new().map(&file)? };
                    maps.push(map);
                }
                Backend::Mmap(maps)
            }
        };
        Ok(Self {
            model,
            mode,
            layouts,
            backend,
            stats: AtomicStats::default(),
            paths,
        })
    }

    pub fn mode(&self) -> StoreMode {
        self.mode
    }

    pub fn model(&self) -> &Arc<ModelDescriptor> {
        &self.model
    }

    pub fn expert_location(&self, layer: usize, expert: u32) -> Result<ExpertLocation, IoError> {
        let layout = self
            .layouts
            .get(layer)
            .and_then(Option::as_ref)
            .ok_or(IoError::MissingExpertLayout(layer))?;
        ExpertLocation::from_layout(layout, expert)
    }

    pub fn read_blob(&self, blob: &TensorBlob, dst: &mut [u8]) -> Result<(), IoError> {
        self.read_blob_inner(blob, dst).inspect_err(|_| {
            self.stats.errors.fetch_add(1, Ordering::Relaxed);
        })
    }

    fn read_blob_inner(&self, blob: &TensorBlob, dst: &mut [u8]) -> Result<(), IoError> {
        let len = blob.len as usize;
        if dst.len() < len {
            return Err(IoError::BufferTooSmall {
                need: len,
                got: dst.len(),
            });
        }
        if blob.shard >= self.paths.len() {
            return Err(IoError::ShardOutOfRange(blob.shard));
        }
        let dst = &mut dst[..len];
        match &self.backend {
            Backend::Pread(files) => read_exact_at(&files[blob.shard], blob.offset, dst)?,
            Backend::Mmap(maps) => {
                let map = &maps[blob.shard];
                let start = usize::try_from(blob.offset).map_err(|_| IoError::OffsetTooLarge)?;
                let end = start.checked_add(len).ok_or(IoError::OffsetTooLarge)?;
                let slice = map.get(start..end).ok_or(IoError::BlobOutOfFile {
                    tensor: blob.tensor.clone(),
                    end: end as u64,
                })?;
                dst.copy_from_slice(slice);
            }
        }
        self.stats.reads.fetch_add(1, Ordering::Relaxed);
        self.stats.bytes_read.fetch_add(len as u64, Ordering::Relaxed);
        Ok(())
    }

    pub fn read_expert_into(
        &self,
        layer: usize,
        expert: u32,
        gate: &mut [u8],
        down: &mut [u8],
        up: &mut [u8],
    ) -> Result<ExpertLocation, IoError> {
        let loc = self.expert_location(layer, expert)?;
        self.read_blob(&loc.gate, gate)?;
        self.read_blob(&loc.down, down)?;
        self.read_blob(&loc.up, up)?;
        Ok(loc)
    }

    pub fn read_expert_alloc(&self, layer: usize, expert: u32) -> Result<ExpertPayload, IoError> {
        let loc = self.expert_location(layer, expert)?;
        let mut gate = vec![0u8; loc.gate.len as usize];
        let mut down = vec![0u8; loc.down.len as usize];
        let mut up = vec![0u8; loc.up.len as usize];
        self.read_expert_into(layer, expert, &mut gate, &mut down, &mut up)?;
        Ok(ExpertPayload {
            layer,
            expert,
            gate,
            down,
            up,
        })
    }

    /// Async wrapper for the explicit I/O path. The blocking positioned read
    /// runs on Tokio's blocking pool; a scheduler can overlap it with other
    /// reads or GPU work.
    pub async fn read_expert_alloc_async(
        self: Arc<Self>,
        layer: usize,
        expert: u32,
    ) -> Result<ExpertPayload, IoError> {
        tokio::task::spawn_blocking(move || self.read_expert_alloc(layer, expert))
            .await
            .map_err(|e| IoError::Join(e.to_string()))?
    }

    pub fn stats(&self) -> StoreStats {
        StoreStats {
            reads: self.stats.reads.load(Ordering::Relaxed),
            bytes_read: self.stats.bytes_read.load(Ordering::Relaxed),
            errors: self.stats.errors.load(Ordering::Relaxed),
        }
    }

    pub fn total_expert_bytes(&self) -> u64 {
        self.layouts
            .iter()
            .flatten()
            .map(|l| l.expert_bytes.saturating_mul(l.expert_count))
            .sum()
    }

    pub fn shard_paths(&self) -> &[PathBuf] {
        &self.paths
    }
}

impl Clone for WeightStore {
    fn clone(&self) -> Self {
        // Reopen files/maps rather than sharing a mutable cursor. This is only
        // used for worker handles, not per-token hot paths.
        Self::open(self.model.clone(), self.mode).expect("reopen weight store")
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct ExpertPayload {
    pub layer: usize,
    pub expert: u32,
    pub gate: Vec<u8>,
    pub down: Vec<u8>,
    pub up: Vec<u8>,
}

#[derive(Debug, Error)]
pub enum IoError {
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),
    #[error("no GGUF shards in model")]
    NoShards,
    #[error("missing expert layout for layer {0}")]
    MissingExpertLayout(usize),
    #[error("expert {expert} out of range for layer {layer} (count {count})")]
    ExpertOutOfRange { layer: usize, expert: u32, count: u64 },
    #[error("shard index {0} is out of range")]
    ShardOutOfRange(usize),
    #[error("buffer too small: need {need}, got {got}")]
    BufferTooSmall { need: usize, got: usize },
    #[error("file offset does not fit usize")]
    OffsetTooLarge,
    #[error("tensor {tensor} ends past mapped file at byte {end}")]
    BlobOutOfFile { tensor: String, end: u64 },
    #[error("blocking I/O task failed: {0}")]
    Join(String),
}

impl Clone for IoError {
    fn clone(&self) -> Self {
        match self {
            Self::Io(e) => Self::Io(io::Error::new(e.kind(), e.to_string())),
            Self::NoShards => Self::NoShards,
            Self::MissingExpertLayout(l) => Self::MissingExpertLayout(*l),
            Self::ExpertOutOfRange { layer, expert, count } => Self::ExpertOutOfRange {
                layer: *layer,
                expert: *expert,
                count: *count,
            },
            Self::ShardOutOfRange(s) => Self::ShardOutOfRange(*s),
            Self::BufferTooSmall { need, got } => Self::BufferTooSmall {
                need: *need,
                got: *got,
            },
            Self::OffsetTooLarge => Self::OffsetTooLarge,
            Self::BlobOutOfFile { tensor, end } => Self::BlobOutOfFile {
                tensor: tensor.clone(),
                end: *end,
            },
            Self::Join(message) => Self::Join(message.clone()),
        }
    }
}

#[cfg(windows)]
fn read_exact_at(file: &File, mut offset: u64, mut buf: &mut [u8]) -> io::Result<()> {
    use std::os::windows::fs::FileExt;
    while !buf.is_empty() {
        match file.seek_read(buf, offset) {
            Ok(0) => {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "short positioned read",
                ))
            }
            Ok(n) => {
                offset += n as u64;
                buf = &mut buf[n..];
            }
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

#[cfg(unix)]
fn read_exact_at(file: &File, mut offset: u64, mut buf: &mut [u8]) -> io::Result<()> {
    use std::os::unix::fs::FileExt;
    while !buf.is_empty() {
        match file.read_at(buf, offset) {
            Ok(0) => {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "short positioned read",
                ))
            }
            Ok(n) => {
                offset += n as u64;
                buf = &mut buf[n..];
            }
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn positioned_read_handles_offsets() {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(b"0123456789abcdef").unwrap();
        f.flush().unwrap();
        let file = f.reopen().unwrap();
        let mut buf = [0u8; 4];
        read_exact_at(&file, 5, &mut buf).unwrap();
        assert_eq!(&buf, b"5678");
        let err = read_exact_at(&file, 14, &mut buf).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::UnexpectedEof);
    }

    #[test]
    fn expert_location_offsets() {
        let layout = ExpertLayout {
            layer: 7,
            expert_count: 384,
            gate_bytes: 1000,
            down_bytes: 2000,
            up_bytes: 3000,
            expert_bytes: 6000,
            gate_offset: 10_000,
            down_offset: 20_000,
            up_offset: 30_000,
            gate_shard: 1,
            down_shard: 2,
            up_shard: 3,
        };
        let loc = ExpertLocation::from_layout(&layout, 6).unwrap();
        assert_eq!(loc.gate.offset, 16_000);
        assert_eq!(loc.down.offset, 32_000);
        assert_eq!(loc.up.offset, 48_000);
        assert!(ExpertLocation::from_layout(&layout, 384).is_err());
    }
}
