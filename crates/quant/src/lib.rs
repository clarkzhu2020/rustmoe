//! GGML quantization descriptors and a small, auditable set of reference
//! dequantizers.
//!
//! GGUF tensor names and quantization names are not interchangeable. A file
//! called Q4_K_XL may contain MXFP4 experts, Q8_0 dense tensors, F32 norms,
//! and an MXFP8 lookup table. Always dispatch on the numeric ggml_type stored
//! in the tensor directory.

use serde::{Deserialize, Serialize};
use std::fmt;
use thiserror::Error;

/// One quantization block layout, as stored in a GGUF tensor directory.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct GgmlType(pub u32);

impl GgmlType {
    pub const F32: Self = Self(0);
    pub const F16: Self = Self(1);
    pub const Q4_0: Self = Self(2);
    pub const Q4_1: Self = Self(3);
    pub const Q5_0: Self = Self(6);
    pub const Q5_1: Self = Self(7);
    pub const Q8_0: Self = Self(8);
    pub const Q8_1: Self = Self(9);
    pub const Q2_K: Self = Self(10);
    pub const Q3_K: Self = Self(11);
    pub const Q4_K: Self = Self(12);
    pub const Q5_K: Self = Self(13);
    pub const Q6_K: Self = Self(14);
    pub const Q8_K: Self = Self(15);
    pub const IQ2_XXS: Self = Self(16);
    pub const IQ2_XS: Self = Self(17);
    pub const IQ3_XXS: Self = Self(18);
    pub const IQ1_S: Self = Self(19);
    pub const IQ4_NL: Self = Self(20);
    pub const IQ3_S: Self = Self(21);
    pub const IQ2_S: Self = Self(22);
    pub const IQ4_XS: Self = Self(23);
    pub const I8: Self = Self(24);
    pub const I16: Self = Self(25);
    pub const I32: Self = Self(26);
    pub const I64: Self = Self(27);
    pub const F64: Self = Self(28);
    pub const IQ1_M: Self = Self(29);
    pub const BF16: Self = Self(30);
    pub const TQ1_0: Self = Self(34);
    pub const TQ2_0: Self = Self(35);
    pub const MXFP4: Self = Self(39);
    pub const NVFP4: Self = Self(40);
    pub const Q1_0: Self = Self(41);
    pub const Q2_0: Self = Self(42);
    /// MXFP8 as used by the DeepSeek-V4.1 Engram table in Unsloth GGUF
    /// releases. It is a GGUF extension at the time of writing.
    pub const MXFP8: Self = Self(43);
    /// Unsloth DeepSeek-V4.1 fork-local IQ1 variants.
    pub const IQ1_XS: Self = Self(64);
    pub const IQ1_XXS: Self = Self(65);
    pub const IQ1_XXXS: Self = Self(66);

    pub fn name(self) -> &'static str {
        static_info(self.0).map(|i| i.name).unwrap_or("unknown")
    }

    pub fn block_elems(self) -> Option<u64> {
        static_info(self.0).map(|i| i.block_elems)
    }

    pub fn block_bytes(self) -> Option<u64> {
        static_info(self.0).map(|i| i.block_bytes)
    }

    pub fn bits_per_weight(self) -> Option<f64> {
        let info = static_info(self.0)?;
        Some((info.block_bytes as f64 * 8.0) / info.block_elems as f64)
    }

    pub fn is_known(self) -> bool {
        static_info(self.0).is_some()
    }

    pub fn is_quantized(self) -> bool {
        static_info(self.0).map(|i| i.quantized).unwrap_or(false)
    }

    /// Whether RustMoE has a reference dequantizer for this type. Layout
    /// validation and byte accounting work for all known types, even when a
    /// native kernel has not been written yet.
    pub fn has_reference_dequant(self) -> bool {
        matches!(
            self,
            Self::F32
                | Self::F16
                | Self::BF16
                | Self::Q4_0
                | Self::Q8_0
                | Self::Q4_K
                | Self::MXFP4
                | Self::MXFP8
        )
    }

    pub fn row_size(self, elements: u64) -> Result<u64, QuantError> {
        let info = static_info(self.0).ok_or(QuantError::UnknownType(self.0))?;
        if elements % info.block_elems != 0 {
            return Err(QuantError::MisalignedRow {
                type_name: info.name,
                elements,
                block_elems: info.block_elems,
            });
        }
        elements
            .checked_div(info.block_elems)
            .and_then(|blocks| blocks.checked_mul(info.block_bytes))
            .ok_or(QuantError::SizeOverflow)
    }

    /// Byte size of a multi-dimensional tensor. GGUF dimension 0 is the
    /// contiguous row dimension.
    pub fn tensor_size(self, dims: &[u64]) -> Result<u64, QuantError> {
        let (&row, rest) = dims.split_first().ok_or(QuantError::EmptyShape)?;
        let row_bytes = self.row_size(row)?;
        let rows = rest
            .iter()
            .try_fold(1u64, |acc, &d| acc.checked_mul(d))
            .ok_or(QuantError::SizeOverflow)?;
        row_bytes.checked_mul(rows).ok_or(QuantError::SizeOverflow)
    }

    pub fn dequantize(self, src: &[u8], elements: u64, dst: &mut Vec<f32>) -> Result<(), QuantError> {
        let info = static_info(self.0).ok_or(QuantError::UnknownType(self.0))?;
        let expected = self.row_size(elements)? as usize;
        if src.len() != expected {
            return Err(QuantError::InputLength {
                expected,
                actual: src.len(),
            });
        }
        dst.clear();
        dst.try_reserve_exact(elements as usize)
            .map_err(|_| QuantError::SizeOverflow)?;
        match self {
            Self::F32 => dequant_f32(src, dst),
            Self::F16 => dequant_f16(src, dst),
            Self::BF16 => dequant_bf16(src, dst),
            Self::Q4_0 => dequant_q4_0(src, dst),
            Self::Q8_0 => dequant_q8_0(src, dst),
            Self::Q4_K => dequant_q4_k(src, dst),
            Self::MXFP4 => dequant_mxfp4(src, dst),
            Self::MXFP8 => dequant_mxfp8(src, dst),
            _ => {
                return Err(QuantError::NoDequantizer {
                    type_name: info.name,
                    id: self.0,
                })
            }
        }
        Ok(())
    }
}

impl fmt::Display for GgmlType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_known() {
            write!(f, "{} ({})", self.name(), self.0)
        } else {
            write!(f, "unknown ({})", self.0)
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct TypeInfo {
    pub id: u32,
    pub name: &'static str,
    pub block_elems: u64,
    pub block_bytes: u64,
    pub quantized: bool,
}

macro_rules! info {
    ($id:expr, $name:expr, $elems:expr, $bytes:expr, $q:expr) => {
        TypeInfo {
            id: $id,
            name: $name,
            block_elems: $elems,
            block_bytes: $bytes,
            quantized: $q,
        }
    };
}

static TYPES: &[TypeInfo] = &[
    info!(0, "f32", 1, 4, false),
    info!(1, "f16", 1, 2, false),
    info!(2, "q4_0", 32, 18, true),
    info!(3, "q4_1", 32, 20, true),
    info!(6, "q5_0", 32, 22, true),
    info!(7, "q5_1", 32, 24, true),
    info!(8, "q8_0", 32, 34, true),
    info!(9, "q8_1", 32, 36, true),
    info!(10, "q2_K", 256, 84, true),
    info!(11, "q3_K", 256, 110, true),
    info!(12, "q4_K", 256, 144, true),
    info!(13, "q5_K", 256, 176, true),
    info!(14, "q6_K", 256, 210, true),
    info!(15, "q8_K", 256, 292, true),
    info!(16, "iq2_xxs", 256, 66, true),
    info!(17, "iq2_xs", 256, 74, true),
    info!(18, "iq3_xxs", 256, 98, true),
    info!(19, "iq1_s", 256, 50, true),
    info!(20, "iq4_nl", 32, 18, true),
    info!(21, "iq3_s", 256, 112, true),
    info!(22, "iq2_s", 256, 82, true),
    info!(23, "iq4_xs", 256, 136, true),
    info!(24, "i8", 1, 1, false),
    info!(25, "i16", 1, 2, false),
    info!(26, "i32", 1, 4, false),
    info!(27, "i64", 1, 8, false),
    info!(28, "f64", 1, 8, false),
    info!(29, "iq1_m", 256, 56, true),
    info!(30, "bf16", 1, 2, false),
    info!(34, "tq1_0", 256, 54, true),
    info!(35, "tq2_0", 256, 66, true),
    info!(39, "mxfp4", 32, 17, true),
    info!(40, "nvfp4", 64, 36, true),
    info!(41, "q1_0", 128, 18, true),
    info!(42, "q2_0", 64, 18, true),
    info!(43, "mxfp8", 32, 33, true),
    info!(64, "iq1_xs", 256, 46, true),
    info!(65, "iq1_xxs", 256, 42, true),
    info!(66, "iq1_xxxs", 256, 38, true),
];

fn static_info(id: u32) -> Option<TypeInfo> {
    TYPES.iter().copied().find(|i| i.id == id)
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum QuantError {
    #[error("unknown GGML tensor type {0}")]
    UnknownType(u32),
    #[error("row of {elements} elements is not a multiple of {type_name} block size {block_elems}")]
    MisalignedRow {
        type_name: &'static str,
        elements: u64,
        block_elems: u64,
    },
    #[error("empty tensor shape")]
    EmptyShape,
    #[error("tensor byte size overflows u64")]
    SizeOverflow,
    #[error("input buffer has {actual} bytes, expected {expected}")]
    InputLength { expected: usize, actual: usize },
    #[error("no reference dequantizer for {type_name} (GGML type {id})")]
    NoDequantizer { type_name: &'static str, id: u32 },
}

fn dequant_f32(src: &[u8], dst: &mut Vec<f32>) {
    for c in src.chunks_exact(4) {
        dst.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
    }
}

pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let frac = (bits & 0x03ff) as u32;
    let out = if exp == 0 {
        if frac == 0 {
            sign << 31
        } else {
            // Subnormal half: normalize into a normal f32.
            let mut e = 127 - 15 + 1i32;
            let mut f = frac;
            while f & 0x0400 == 0 {
                f <<= 1;
                e -= 1;
            }
            f &= 0x03ff;
            (sign << 31) | ((e as u32) << 23) | (f << 13)
        }
    } else if exp == 0x1f {
        (sign << 31) | 0x7f80_0000 | (frac << 13)
    } else {
        (sign << 31) | ((exp + 127 - 15) << 23) | (frac << 13)
    };
    f32::from_bits(out)
}

pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

fn dequant_f16(src: &[u8], dst: &mut Vec<f32>) {
    for c in src.chunks_exact(2) {
        dst.push(f16_to_f32(u16::from_le_bytes([c[0], c[1]])));
    }
}

fn dequant_bf16(src: &[u8], dst: &mut Vec<f32>) {
    for c in src.chunks_exact(2) {
        dst.push(bf16_to_f32(u16::from_le_bytes([c[0], c[1]])));
    }
}

fn dequant_q8_0(src: &[u8], dst: &mut Vec<f32>) {
    for block in src.chunks_exact(34) {
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        for &q in &block[2..34] {
            dst.push(d * (q as i8) as f32);
        }
    }
}

fn dequant_q4_0(src: &[u8], dst: &mut Vec<f32>) {
    for block in src.chunks_exact(18) {
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        for &byte in &block[2..18] {
            dst.push(d * ((byte & 0x0f) as i32 - 8) as f32);
        }
        for &byte in &block[2..18] {
            dst.push(d * ((byte >> 4) as i32 - 8) as f32);
        }
    }
}

fn dequant_mxfp4(src: &[u8], dst: &mut Vec<f32>) {
    // E2M1 magnitudes. Multiplying by 2^(e-127) directly is equivalent to
    // llama.cpp's doubled table with a half scale.
    const TABLE: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    for block in src.chunks_exact(17) {
        let scale = e8m0_to_f32(block[0]);
        for &byte in &block[1..17] {
            let lo = (byte & 0x0f) as usize;
            let val = if lo & 8 != 0 { -TABLE[lo & 7] } else { TABLE[lo] };
            dst.push(scale * val);
        }
        for &byte in &block[1..17] {
            let hi = (byte >> 4) as usize;
            let val = if hi & 8 != 0 { -TABLE[hi & 7] } else { TABLE[hi] };
            dst.push(scale * val);
        }
    }
}

fn dequant_mxfp8(src: &[u8], dst: &mut Vec<f32>) {
    for block in src.chunks_exact(33) {
        let scale = e8m0_to_f32(block[0]);
        for &q in &block[1..33] {
            dst.push(scale * e4m3_to_f32(q));
        }
    }
}

fn e8m0_to_f32(e: u8) -> f32 {
    if e == 255 {
        f32::NAN
    } else {
        2.0f32.powi(e as i32 - 127)
    }
}

fn e4m3_to_f32(q: u8) -> f32 {
    let sign = if q & 0x80 != 0 { -1.0 } else { 1.0 };
    let exp = ((q >> 3) & 0x0f) as i32;
    let mant = (q & 0x07) as f32;
    if exp == 0 {
        sign * (mant / 8.0) * 2.0f32.powi(1 - 7)
    } else if exp == 0x0f && mant == 0.0 {
        sign * f32::NAN
    } else {
        sign * (1.0 + mant / 8.0) * 2.0f32.powi(exp - 7)
    }
}

fn dequant_q4_k(src: &[u8], dst: &mut Vec<f32>) {
    for block in src.chunks_exact(144) {
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
        let scales = &block[4..16];
        let qs = &block[16..144];
        for group in 0..8 {
            let (sc, m) = scale_min_k4(group, scales);
            let dd = d * sc as f32;
            let mm = dmin * m as f32;
            let base = (group / 2) * 32;
            for j in 0..32 {
                let q = if group & 1 == 0 {
                    qs[base + j] & 0x0f
                } else {
                    qs[base + j] >> 4
                };
                dst.push(dd * q as f32 - mm);
            }
        }
    }
}

fn scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        (
            (q[j + 4] & 0x0f) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_layout_sizes() {
        assert_eq!(GgmlType::MXFP4.row_size(32).unwrap(), 17);
        assert_eq!(GgmlType::MXFP4.row_size(5120).unwrap(), 2720);
        assert_eq!(GgmlType::MXFP8.row_size(256).unwrap(), 264);
        assert_eq!(GgmlType::Q8_0.row_size(5120).unwrap(), 5440);
        assert_eq!(GgmlType::F32.row_size(5120).unwrap(), 20480);
    }

    #[test]
    fn mxfp4_layout_is_exact_for_deepseek_experts() {
        let bytes = GgmlType::MXFP4.tensor_size(&[5120, 2304, 384]).unwrap();
        assert_eq!(bytes, 2_406_481_920);
    }

    #[test]
    fn mxfp4_unpacking_matches_known_values() {
        let mut block = vec![127u8]; // scale 1.0
        block.push(0x11);
        block.extend(std::iter::repeat_n(0u8, 14));
        block.push(0xff);
        assert_eq!(block.len(), 17);
        let mut out = Vec::new();
        GgmlType::MXFP4.dequantize(&block, 32, &mut out).unwrap();
        assert_eq!(out.len(), 32);
        assert_eq!(out[0], 0.5);
        assert_eq!(out[15], -6.0);
        assert_eq!(out[16], 0.5);
        assert_eq!(out[31], -6.0);
    }

    #[test]
    fn mxfp8_scale_and_e4m3() {
        let mut block = vec![127u8];
        block.push(0x38); // 1.0
        block.extend(std::iter::repeat_n(0u8, 31));
        let mut out = Vec::new();
        GgmlType::MXFP8.dequantize(&block, 32, &mut out).unwrap();
        assert_eq!(out[0], 1.0);
        assert_eq!(out[1], 0.0);
    }

    #[test]
    fn q8_0_decode() {
        let mut block = Vec::new();
        block.extend_from_slice(&0x3c00u16.to_le_bytes()); // 1.0 in f16
        for i in 0..32i8 {
            block.push((i - 16) as u8);
        }
        let mut out = Vec::new();
        GgmlType::Q8_0.dequantize(&block, 32, &mut out).unwrap();
        assert_eq!(out[0], -16.0);
        assert_eq!(out[31], 15.0);
    }

    #[test]
    fn iq1_fork_types_are_accounted() {
        assert_eq!(GgmlType::IQ1_XS.row_size(256).unwrap(), 46);
        assert_eq!(GgmlType::IQ1_XXS.row_size(256).unwrap(), 42);
        assert_eq!(GgmlType::IQ1_XXXS.row_size(256).unwrap(), 38);
        assert_eq!(GgmlType::IQ1_XXXS.name(), "iq1_xxxs");
    }

    #[test]
    fn rejects_unaligned_rows_and_unknown_types() {
        assert!(GgmlType::Q8_0.row_size(31).is_err());
        assert_eq!(GgmlType(999).name(), "unknown");
        assert!(GgmlType(999).tensor_size(&[1]).is_err());
    }
}
