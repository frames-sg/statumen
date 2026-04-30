use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::error::WsiError;

/// Unique identity for an IFD, derived from its byte offset in the file.
/// Defined here (not in container.rs) to avoid circular dependency with the error type.
#[derive(Clone, Copy, Hash, Eq, PartialEq, Debug)]
pub(crate) struct IfdId(pub u64);

impl fmt::Display for IfdId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "IFD@{}", self.0)
    }
}

#[derive(Clone, Debug, thiserror::Error)]
pub(crate) enum TiffParseError {
    #[error("I/O error ({kind}): {source}")]
    Io {
        kind: std::io::ErrorKind,
        #[source]
        source: Arc<std::io::Error>,
        path: Option<Arc<PathBuf>>,
    },

    #[error("invalid tag in IFD at offset {ifd_offset}, tag {tag}: {message}")]
    InvalidTag {
        ifd_offset: u64,
        tag: u16,
        message: String,
    },

    #[error("out of bounds: offset {offset}, len {len}")]
    Bounds { offset: u64, len: u64 },

    #[error("invalid TIFF structure: {0}")]
    Structure(String),

    #[error("IFD not found: {0}")]
    IfdNotFound(IfdId),

    #[error("tag not found: IFD at offset {ifd_offset}, tag {tag}")]
    TagNotFound { ifd_offset: u64, tag: u16 },
}

impl From<std::io::Error> for TiffParseError {
    fn from(e: std::io::Error) -> Self {
        let kind = e.kind();
        TiffParseError::Io {
            kind,
            source: Arc::new(e),
            path: None,
        }
    }
}

impl TiffParseError {
    /// Convert to WsiError at the module boundary. Requires the file path
    /// because TiffParseError does not always carry one.
    pub fn into_wsi_error(self, path: &Path) -> WsiError {
        match self {
            TiffParseError::Io {
                source,
                path: io_path,
                ..
            } => WsiError::IoWithPath {
                source,
                path: io_path
                    .map(|p| p.as_ref().clone())
                    .unwrap_or_else(|| path.to_path_buf()),
            },
            other => WsiError::Tiff {
                path: path.to_path_buf(),
                message: other.to_string(),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_io_error() {
        let err = TiffParseError::Io {
            kind: std::io::ErrorKind::NotFound,
            source: Arc::new(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "file not found",
            )),
            path: None,
        };
        let s = err.to_string();
        assert!(s.contains("I/O error"), "got: {}", s);
        assert!(s.contains("file not found"), "got: {}", s);
    }

    #[test]
    fn display_invalid_tag() {
        let err = TiffParseError::InvalidTag {
            ifd_offset: 1024,
            tag: 256,
            message: "expected LONG, got ASCII".into(),
        };
        let s = err.to_string();
        assert!(s.contains("1024"), "got: {}", s);
        assert!(s.contains("256"), "got: {}", s);
        assert!(s.contains("expected LONG"), "got: {}", s);
    }

    #[test]
    fn display_bounds() {
        let err = TiffParseError::Bounds {
            offset: 999999,
            len: 4096,
        };
        let s = err.to_string();
        assert!(s.contains("999999"), "got: {}", s);
        assert!(s.contains("4096"), "got: {}", s);
    }

    #[test]
    fn display_structure() {
        let err = TiffParseError::Structure("IFD chain loop detected".into());
        assert!(err.to_string().contains("loop detected"));
    }

    #[test]
    fn display_ifd_not_found() {
        let err = TiffParseError::IfdNotFound(IfdId(8192));
        let s = err.to_string();
        assert!(s.contains("8192"), "got: {}", s);
    }

    #[test]
    fn display_tag_not_found() {
        let err = TiffParseError::TagNotFound {
            ifd_offset: 512,
            tag: 322,
        };
        let s = err.to_string();
        assert!(s.contains("512"), "got: {}", s);
        assert!(s.contains("322"), "got: {}", s);
    }

    #[test]
    fn from_io_error() {
        let io_err = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "access denied");
        let tpe: TiffParseError = io_err.into();
        match &tpe {
            TiffParseError::Io { kind, source, .. } => {
                assert_eq!(*kind, std::io::ErrorKind::PermissionDenied);
                assert_eq!(source.kind(), std::io::ErrorKind::PermissionDenied);
                assert!(source.to_string().contains("access denied"));
            }
            other => panic!("expected Io variant, got: {:?}", other),
        }
    }

    #[test]
    fn into_wsi_error_conversion() {
        let tpe = TiffParseError::Structure("bad header".into());
        let path = Path::new("/tmp/slide.svs");
        let wsi_err = tpe.into_wsi_error(path);
        match wsi_err {
            WsiError::Tiff {
                path: p,
                message: m,
            } => {
                assert_eq!(p, PathBuf::from("/tmp/slide.svs"));
                assert!(m.contains("bad header"), "got: {}", m);
            }
            other => panic!("expected Tiff variant, got: {:?}", other),
        }
    }

    #[test]
    fn into_wsi_error_io_routes_to_io_with_path() {
        let tpe = TiffParseError::Io {
            kind: std::io::ErrorKind::NotFound,
            source: Arc::new(std::io::Error::new(std::io::ErrorKind::NotFound, "gone")),
            path: None,
        };
        let wsi_err = tpe.into_wsi_error(Path::new("/tmp/test.ndpi"));
        match wsi_err {
            WsiError::IoWithPath { source, path: p } => {
                assert_eq!(source.kind(), std::io::ErrorKind::NotFound);
                assert!(source.to_string().contains("gone"), "got: {}", source);
                assert_eq!(p, PathBuf::from("/tmp/test.ndpi"));
            }
            other => panic!("expected IoWithPath, got: {:?}", other),
        }
    }

    #[test]
    fn ifd_id_equality_and_display() {
        let a = IfdId(100);
        let b = IfdId(100);
        let c = IfdId(200);
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert!(a.to_string().contains("100"));
    }
}
