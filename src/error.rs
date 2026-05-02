use std::path::PathBuf;
use std::sync::Arc;

#[derive(Debug, thiserror::Error)]
pub enum WsiError {
    #[error("unsupported format: {0}")]
    UnsupportedFormat(String),
    #[error("TIFF error in {path}: {message}")]
    Tiff { path: PathBuf, message: String },
    #[error("JPEG decode error: {0}")]
    Jpeg(String),
    #[error("JPEG2000 decode error: {0}")]
    Jp2k(String),
    #[error("XML parse error: {0}")]
    Xml(String),
    #[error("invalid slide {path}: {message}")]
    InvalidSlide { path: PathBuf, message: String },
    #[error("tile read failed at ({col}, {row}) level {level}: {reason}")]
    TileRead {
        col: i64,
        row: i64,
        level: u32,
        reason: String,
    },
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("I/O error at {path}: {source}")]
    IoWithPath {
        #[source]
        source: Arc<std::io::Error>,
        path: PathBuf,
    },

    // --- New variants for multi-dimensional engine ---
    #[error("scene index {index} out of range (dataset has {count} scenes)")]
    SceneOutOfRange { index: usize, count: usize },

    #[error("series index {index} out of range (scene has {count} series)")]
    SeriesOutOfRange { index: usize, count: usize },

    #[error("level {level} out of range (series has {count} levels)")]
    LevelOutOfRange { level: u32, count: u32 },

    #[error("plane axis {axis} value {value} exceeds max {max}")]
    PlaneOutOfRange { axis: String, value: u32, max: u32 },

    #[error("associated image not found: {0}")]
    AssociatedImageNotFound(String),

    #[error("display conversion error: {0}")]
    DisplayConversion(String),

    /// Codec-layer error from signinum or the transitional facade.
    #[error("codec error in {codec}: {source}")]
    Codec {
        codec: &'static str,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    /// Operation is intentionally unsupported on this path.
    #[error("unsupported: {reason}")]
    Unsupported { reason: String },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_display_formats() {
        let err = WsiError::Tiff {
            path: "/tmp/test.svs".into(),
            message: "bad IFD".into(),
        };
        assert!(err.to_string().contains("test.svs"));
        assert!(err.to_string().contains("bad IFD"));
    }

    #[test]
    fn io_error_converts() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "gone");
        let wsi_err: WsiError = io_err.into();
        assert!(matches!(wsi_err, WsiError::Io(_)));
    }

    #[test]
    fn scene_out_of_range_display() {
        let err = WsiError::SceneOutOfRange { index: 2, count: 1 };
        assert!(err.to_string().contains("2"));
        assert!(err.to_string().contains("1"));
    }

    #[test]
    fn series_out_of_range_display() {
        let err = WsiError::SeriesOutOfRange { index: 3, count: 2 };
        assert!(err.to_string().contains("3"));
    }

    #[test]
    fn plane_out_of_range_display() {
        let err = WsiError::PlaneOutOfRange {
            axis: "z".into(),
            value: 5,
            max: 3,
        };
        assert!(err.to_string().contains("z"));
        assert!(err.to_string().contains("5"));
    }

    #[test]
    fn level_out_of_range_display() {
        let err = WsiError::LevelOutOfRange {
            level: 10,
            count: 5,
        };
        assert!(err.to_string().contains("10"));
    }

    #[test]
    fn associated_image_not_found_display() {
        let err = WsiError::AssociatedImageNotFound("label".into());
        assert!(err.to_string().contains("label"));
    }

    #[test]
    fn display_conversion_display() {
        let err = WsiError::DisplayConversion("non-uint8 requires windowing".into());
        assert!(err.to_string().contains("windowing"));
    }

    #[test]
    fn io_with_path_display() {
        let err = WsiError::IoWithPath {
            source: Arc::new(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "file not found",
            )),
            path: "/tmp/slide.svs".into(),
        };
        let msg = err.to_string();
        assert!(msg.contains("/tmp/slide.svs"), "got: {msg}");
        assert!(msg.contains("file not found"), "got: {msg}");
    }

    #[test]
    fn codec_display_includes_codec_and_source() {
        let inner: Box<dyn std::error::Error + Send + Sync> = "boom".into();
        let err = WsiError::Codec {
            codec: "jpeg",
            source: inner,
        };
        let msg = err.to_string();
        assert!(msg.contains("jpeg"), "got: {msg}");
        assert!(msg.contains("boom"), "got: {msg}");
    }

    #[test]
    fn codec_pattern_match_round_trips() {
        let err = WsiError::Codec {
            codec: "j2k",
            source: "decode failed".into(),
        };
        match err {
            WsiError::Codec { codec, source: _ } => assert_eq!(codec, "j2k"),
            other => panic!("expected Codec, got {other:?}"),
        }
    }

    #[test]
    fn unsupported_display() {
        let err = WsiError::Unsupported {
            reason: "device backend unavailable".into(),
        };
        assert!(err.to_string().contains("device backend unavailable"));
    }
}
