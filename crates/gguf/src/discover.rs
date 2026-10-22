use super::{GgufError, GGUF_MAGIC};
use std::path::{Path, PathBuf};

/// Expand a split entry path such as model-00001-of-00009.gguf into all
/// expected shard paths. A split filename is a contract: every shard must be
/// present, and no silently missing shard is tolerated.
pub fn discover_split_paths(path: &Path) -> Result<Vec<PathBuf>, GgufError> {
    if !path.exists() {
        return Err(GgufError::NoShards(path.to_path_buf()));
    }
    let name = path
        .file_name()
        .and_then(|s| s.to_str())
        .ok_or_else(|| GgufError::SplitFilename(path.display().to_string()))?;
    let Some(parts) = parse_split_name(name) else {
        return Ok(vec![path.to_path_buf()]);
    };
    if parts.count == 0 || parts.index == 0 || parts.index > parts.count {
        return Err(GgufError::SplitFilename(name.to_string()));
    }
    let dir = path.parent().unwrap_or_else(|| Path::new("."));
    let mut out = Vec::with_capacity(parts.count as usize);
    for i in 1..=parts.count {
        let candidate = dir.join(format!("{}-{i:05}-of-{:05}.gguf", parts.prefix, parts.count));
        if !candidate.exists() {
            return Err(GgufError::SplitFilename(format!(
                "missing shard {i} of {}: {}",
                parts.count,
                candidate.display()
            )));
        }
        out.push(candidate);
    }
    Ok(out)
}

/// Resolve a directory containing exactly one logical GGUF model.
pub fn paths_in_directory(dir: &Path) -> Result<Vec<PathBuf>, GgufError> {
    let mut files = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_file()
            && path
                .extension()
                .and_then(|e| e.to_str())
                .map(|e| e.eq_ignore_ascii_case("gguf"))
                .unwrap_or(false)
        {
            files.push(path);
        }
    }
    files.sort();
    if files.is_empty() {
        return Err(GgufError::NoShards(dir.to_path_buf()));
    }
    if files.len() == 1 {
        return Ok(files);
    }

    let mut sets = std::collections::BTreeSet::new();
    for f in &files {
        let name = f
            .file_name()
            .and_then(|s| s.to_str())
            .ok_or_else(|| GgufError::SplitFilename(f.display().to_string()))?;
        if let Some(p) = parse_split_name(name) {
            sets.insert((p.prefix.clone(), p.count));
        } else {
            return Err(GgufError::SplitFilename(format!(
                "multiple GGUF files in {}; pass the entry shard explicitly",
                dir.display()
            )));
        }
    }
    if sets.len() != 1 {
        return Err(GgufError::SplitFilename(format!(
            "{} contains {} different GGUF split sets; pass the entry shard explicitly",
            dir.display(),
            sets.len()
        )));
    }
    let entry = files
        .iter()
        .find(|p| {
            p.file_name()
                .and_then(|s| s.to_str())
                .and_then(parse_split_name)
                .map(|p| p.index == 1)
                .unwrap_or(false)
        })
        .ok_or_else(|| GgufError::SplitFilename("no shard 1 in split set".into()))?;
    discover_split_paths(entry)
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SplitName {
    prefix: String,
    index: u32,
    count: u32,
}

fn parse_split_name(name: &str) -> Option<SplitName> {
    let stem = name.strip_suffix(".gguf")?;
    let pos = stem.rfind("-of-")?;
    if pos < 6 || pos + 4 + 5 != stem.len() {
        return None;
    }
    let idx_text = &stem[pos - 5..pos];
    let count_text = &stem[pos + 4..];
    if !idx_text.bytes().all(|b| b.is_ascii_digit()) || !count_text.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    if stem.as_bytes().get(pos - 6) != Some(&b'-') {
        return None;
    }
    Some(SplitName {
        prefix: stem[..pos - 6].to_string(),
        index: idx_text.parse().ok()?,
        count: count_text.parse().ok()?,
    })
}

/// Quick check for the GGUF magic without reading the entire header.
pub fn sniff_magic(path: &Path) -> Result<bool, GgufError> {
    use std::io::Read;
    let mut f = std::fs::File::open(path)?;
    let mut magic = [0u8; 4];
    match f.read_exact(&mut magic) {
        Ok(()) => Ok(magic == GGUF_MAGIC),
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => Ok(false),
        Err(e) => Err(e.into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_split_names() {
        assert_eq!(
            parse_split_name("DeepSeek-V4.1-UD-Q4_K_XL-00001-of-00009.gguf"),
            Some(SplitName {
                prefix: "DeepSeek-V4.1-UD-Q4_K_XL".into(),
                index: 1,
                count: 9
            })
        );
        assert_eq!(parse_split_name("plain.gguf"), None);
        assert_eq!(parse_split_name("model-1-of-9.gguf"), None);
    }
}
