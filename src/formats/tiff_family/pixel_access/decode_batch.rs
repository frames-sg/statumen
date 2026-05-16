use super::*;

pub(super) enum CodecBatchJob<'a> {
    Jpeg(JpegDecodeJob<'a>),
    Jp2k(Jp2kDecodeJob<'a>),
}

#[derive(Clone, Copy, Debug)]
pub(super) struct TiffJpegDecodeOptions {
    pub(super) force_dimensions: bool,
    pub(super) color_transform: SigninumColorTransform,
}

pub(super) fn decode_one_jpeg(job: JpegDecodeJob<'_>) -> Result<CpuTile, WsiError> {
    decode_batch_jpeg(&[job])
        .into_iter()
        .next()
        .expect("1-element JPEG batch")
}

pub(super) fn decode_one_jp2k(job: Jp2kDecodeJob<'_>) -> Result<CpuTile, WsiError> {
    decode_batch_jp2k(&[job])
        .into_iter()
        .next()
        .expect("1-element JP2K batch")
}

pub(super) fn decode_mixed_batch(jobs: Vec<CodecBatchJob<'_>>) -> Vec<Result<CpuTile, WsiError>> {
    let mut jpeg_jobs = Vec::new();
    let mut jpeg_slots = Vec::new();
    let mut jp2k_jobs = Vec::new();
    let mut jp2k_slots = Vec::new();

    for (slot, job) in jobs.into_iter().enumerate() {
        match job {
            CodecBatchJob::Jpeg(job) => {
                jpeg_slots.push(slot);
                jpeg_jobs.push(job);
            }
            CodecBatchJob::Jp2k(job) => {
                jp2k_slots.push(slot);
                jp2k_jobs.push(job);
            }
        }
    }

    let total = jpeg_slots.len() + jp2k_slots.len();
    let mut out: Vec<Option<Result<CpuTile, WsiError>>> = (0..total).map(|_| None).collect();
    for (slot, result) in jpeg_slots.into_iter().zip(decode_batch_jpeg(&jpeg_jobs)) {
        out[slot] = Some(result);
    }
    for (slot, result) in jp2k_slots.into_iter().zip(decode_batch_jp2k(&jp2k_jobs)) {
        out[slot] = Some(result);
    }

    out.into_iter()
        .map(|result| result.expect("every mixed batch slot filled"))
        .collect()
}
