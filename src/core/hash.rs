use crate::error::WsiError;
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::ErrorKind;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

pub struct Quickhash1 {
    hasher: Sha256,
    enabled: bool,
}

impl Quickhash1 {
    pub fn new() -> Self {
        Self {
            hasher: Sha256::new(),
            enabled: true,
        }
    }

    pub fn update(&mut self, data: &[u8]) {
        if self.enabled && !data.is_empty() {
            self.hasher.update(data);
        }
    }

    /// Hash string including a null terminator for compatibility hashing.
    pub fn hash_string(&mut self, s: &str) {
        if self.enabled {
            self.hasher.update(s.as_bytes());
            self.hasher.update([0u8]);
        }
    }

    /// Hash `size` bytes from `path` starting at `offset`. None = to end of file.
    pub fn hash_file_part(
        &mut self,
        path: &Path,
        offset: u64,
        size: Option<u64>,
    ) -> Result<(), WsiError> {
        if !self.enabled {
            return Ok(());
        }
        let mut f = File::open(path)?;
        let file_len = f.metadata()?.len();
        if offset > file_len {
            return Err(std::io::Error::new(
                ErrorKind::UnexpectedEof,
                format!("offset {offset} exceeds file length {file_len}"),
            )
            .into());
        }
        let available = file_len - offset;
        let actual_size = match size {
            Some(s) if s > available => {
                return Err(std::io::Error::new(
                    ErrorKind::UnexpectedEof,
                    format!(
                        "requested {s} bytes at offset {offset}, but only {available} bytes remain"
                    ),
                )
                .into());
            }
            Some(s) => s,
            None => available,
        };
        if offset > 0 {
            f.seek(SeekFrom::Start(offset))?;
        }
        let mut remaining = actual_size;
        let mut buf = [0u8; 4096];
        while remaining > 0 {
            let to_read = (remaining as usize).min(buf.len());
            let n = f.read(&mut buf[..to_read])?;
            if n == 0 {
                break;
            }
            self.hasher.update(&buf[..n]);
            remaining -= n as u64;
        }
        Ok(())
    }

    pub fn hash_file(&mut self, path: &Path) -> Result<(), WsiError> {
        self.hash_file_part(path, 0, None)
    }

    pub fn disable(&mut self) {
        self.enabled = false;
    }

    pub fn finish(self) -> Option<String> {
        if self.enabled {
            Some(format!("{:x}", self.hasher.finalize()))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn hash_data_produces_hex_string() {
        let mut h = Quickhash1::new();
        h.update(b"hello world");
        let result = h.finish().unwrap();
        // SHA-256 hex is always 64 characters
        assert_eq!(result.len(), 64);
        // Verify it's valid hex
        assert!(result.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn hash_string_includes_null_terminator() {
        // hash_string("abc") should equal update(b"abc\0")
        let mut h1 = Quickhash1::new();
        h1.hash_string("abc");
        let r1 = h1.finish().unwrap();

        let mut h2 = Quickhash1::new();
        h2.update(b"abc\0");
        let r2 = h2.finish().unwrap();

        assert_eq!(r1, r2);
    }

    #[test]
    fn disabled_hash_returns_none() {
        let mut h = Quickhash1::new();
        h.update(b"data");
        h.disable();
        assert!(h.finish().is_none());
    }

    #[test]
    fn hash_file_part() {
        // Write "0123456789" to temp file, hash bytes 2..7 ("23456")
        let mut tmp = NamedTempFile::new().unwrap();
        tmp.write_all(b"0123456789").unwrap();
        tmp.flush().unwrap();

        let mut h1 = Quickhash1::new();
        h1.hash_file_part(tmp.path(), 2, Some(5)).unwrap();
        let r1 = h1.finish().unwrap();

        // Compare with direct hash of "23456"
        let mut h2 = Quickhash1::new();
        h2.update(b"23456");
        let r2 = h2.finish().unwrap();

        assert_eq!(r1, r2);
    }

    #[test]
    fn hash_file_part_offset_past_eof_errors() {
        let mut tmp = NamedTempFile::new().unwrap();
        tmp.write_all(b"0123456789").unwrap();
        tmp.flush().unwrap();

        let mut h = Quickhash1::new();
        let err = h.hash_file_part(tmp.path(), 20, Some(1)).unwrap_err();
        assert!(err.to_string().contains("offset 20 exceeds file length 10"));
    }

    #[test]
    fn hash_file_part_range_past_eof_errors() {
        let mut tmp = NamedTempFile::new().unwrap();
        tmp.write_all(b"0123456789").unwrap();
        tmp.flush().unwrap();

        let mut h = Quickhash1::new();
        let err = h.hash_file_part(tmp.path(), 8, Some(5)).unwrap_err();
        assert!(err.to_string().contains("only 2 bytes remain"));
    }
}
