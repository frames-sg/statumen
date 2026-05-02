use signinum_jpeg::{ColorTransform, DecodeOptions};

#[test]
fn ndpi_phase3_prereq_color_transform_settable() {
    let mut opts = DecodeOptions::default();
    opts.set_color_transform(ColorTransform::ForceRgb);
    assert!(matches!(opts.color_transform(), ColorTransform::ForceRgb));
}
