#![cfg(all(target_os = "macos", feature = "metal"))]

use signinum_core::{BackendKind, BackendRequest, DeviceSurface, ImageDecodeDevice, PixelFormat};

const JPEG_FIXTURE: &[u8] = include_bytes!("fixtures/jpeg/baseline_420_16x16.jpg");

const _: () = {
    fn assert_send<T: Send>() {}
    let _ = assert_send::<signinum_jpeg_metal::Surface>;
    let _ = assert_send::<signinum_j2k_metal::Surface>;
};

#[test]
fn metal_surface_accessors_are_public_for_jpeg_and_j2k() {
    let mut jpeg_decoder = signinum_jpeg_metal::Decoder::new(JPEG_FIXTURE).expect("jpeg decoder");
    let jpeg_surface = jpeg_decoder
        .decode_to_device(PixelFormat::Rgb8, BackendRequest::Metal)
        .expect("jpeg metal surface");
    describe_jpeg_surface("jpeg-metal", &jpeg_surface);
    let (jpeg_buffer, jpeg_offset) = jpeg_surface.metal_buffer().expect("jpeg metal buffer");
    assert_eq!(jpeg_surface.backend_kind(), BackendKind::Metal);
    assert_eq!(jpeg_offset, 0);
    assert!(jpeg_buffer.length() as usize >= jpeg_surface.byte_len());

    let mut jpeg_cpu_decoder =
        signinum_jpeg_metal::Decoder::new(JPEG_FIXTURE).expect("jpeg cpu decoder");
    let jpeg_cpu_surface = jpeg_cpu_decoder
        .decode_to_device(PixelFormat::Rgb8, BackendRequest::Cpu)
        .expect("jpeg cpu surface");
    describe_jpeg_surface("jpeg-cpu", &jpeg_cpu_surface);
    assert_eq!(jpeg_cpu_surface.backend_kind(), BackendKind::Cpu);
    assert!(jpeg_cpu_surface.metal_buffer().is_none());

    let j2k_bytes = fixture_gray8_j2k();
    let mut j2k_decoder = signinum_j2k_metal::J2kDecoder::new(&j2k_bytes).expect("j2k decoder");
    let j2k_surface = j2k_decoder
        .decode_to_device(PixelFormat::Gray8, BackendRequest::Metal)
        .expect("j2k metal surface");
    describe_j2k_surface("j2k-metal", &j2k_surface);
    let (j2k_buffer, j2k_offset) = j2k_surface.metal_buffer().expect("j2k metal buffer");
    assert_eq!(j2k_surface.backend_kind(), BackendKind::Metal);
    assert_eq!(j2k_offset, 0);
    assert!(j2k_buffer.length() as usize >= j2k_surface.byte_len());

    let mut j2k_cpu_decoder =
        signinum_j2k_metal::J2kDecoder::new(&j2k_bytes).expect("j2k cpu decoder");
    let j2k_cpu_surface = j2k_cpu_decoder
        .decode_to_device(PixelFormat::Gray8, BackendRequest::Cpu)
        .expect("j2k cpu surface");
    describe_j2k_surface("j2k-cpu", &j2k_cpu_surface);
    assert_eq!(j2k_cpu_surface.backend_kind(), BackendKind::Cpu);
    assert!(j2k_cpu_surface.metal_buffer().is_none());
}

fn describe_jpeg_surface(label: &str, surface: &signinum_jpeg_metal::Surface) {
    let buffer = surface
        .metal_buffer()
        .map(|(buffer, byte_offset)| (buffer.length(), byte_offset));
    println!(
        "{label}: dimensions={:?} pitch_bytes={} pixel_format={:?} backend={:?} metal_buffer={buffer:?}",
        surface.dimensions(),
        surface.pitch_bytes(),
        surface.pixel_format(),
        surface.backend_kind(),
    );
}

fn describe_j2k_surface(label: &str, surface: &signinum_j2k_metal::Surface) {
    let buffer = surface
        .metal_buffer()
        .map(|(buffer, byte_offset)| (buffer.length(), byte_offset));
    println!(
        "{label}: dimensions={:?} pitch_bytes={} pixel_format={:?} backend={:?} metal_buffer={buffer:?}",
        surface.dimensions(),
        surface.pitch_bytes(),
        surface.pixel_format(),
        surface.backend_kind(),
    );
}

fn fixture_gray8_j2k() -> Vec<u8> {
    let pixels: Vec<u8> = (0..16).collect();
    let options = signinum_j2k_native::EncodeOptions {
        reversible: true,
        num_decomposition_levels: 1,
        ..signinum_j2k_native::EncodeOptions::default()
    };
    signinum_j2k_native::encode(&pixels, 4, 4, 1, 8, false, &options)
        .expect("encode gray8 j2k fixture")
}
