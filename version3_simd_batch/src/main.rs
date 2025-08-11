use std::collections::HashMap;
use std::error::Error;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;
use timsrust::{converters::ConvertableDomain, readers::{FrameReader, MetadataReader}, MSLevel};
use rayon::prelude::*;

const NUM_THREADS: usize = 32;
const BATCH_SIZE: usize = 8;

#[derive(Debug, Clone)]
pub struct TimsTOFData {
    pub rt_values_min: Vec<f32>,
    pub mobility_values: Vec<f32>,
    pub mz_values: Vec<f32>,
    pub intensity_values: Vec<u32>,
    pub frame_indices: Vec<u32>,
    pub scan_indices: Vec<u32>,
}

impl TimsTOFData {
    pub fn new() -> Self {
        TimsTOFData {
            rt_values_min: Vec::new(),
            mobility_values: Vec::new(),
            mz_values: Vec::new(),
            intensity_values: Vec::new(),
            frame_indices: Vec::new(),
            scan_indices: Vec::new(),
        }
    }
    
    pub fn with_capacity(capacity: usize) -> Self {
        let aligned_cap = ((capacity + 15) / 16) * 16;
        Self {
            rt_values_min: Vec::with_capacity(aligned_cap),
            mobility_values: Vec::with_capacity(aligned_cap),
            mz_values: Vec::with_capacity(aligned_cap),
            intensity_values: Vec::with_capacity(aligned_cap),
            frame_indices: Vec::with_capacity(aligned_cap),
            scan_indices: Vec::with_capacity(aligned_cap),
        }
    }
    
    fn merge_from(&mut self, other: &mut Self) {
        self.rt_values_min.append(&mut other.rt_values_min);
        self.mobility_values.append(&mut other.mobility_values);
        self.mz_values.append(&mut other.mz_values);
        self.intensity_values.append(&mut other.intensity_values);
        self.frame_indices.append(&mut other.frame_indices);
        self.scan_indices.append(&mut other.scan_indices);
    }
}

#[derive(Debug, Clone)]
pub struct TimsTOFRawData {
    pub ms1_data: TimsTOFData,
    pub ms2_windows: Vec<((f32, f32), TimsTOFData)>,
}

struct FrameSplit {
    pub ms1: TimsTOFData,
    pub ms2: Vec<((u32, u32), TimsTOFData)>,
}

#[inline(always)]
fn quantize_batch(values: &[f32]) -> Vec<u32> {
    values.iter().map(|&x| (x * 10_000.0).round() as u32).collect()
}

#[inline(always)]
fn find_scan_for_index_binary(index: usize, scan_offsets: &[usize]) -> usize {
    let mut left = 0;
    let mut right = scan_offsets.len() - 1;
    
    while left < right {
        let mid = left + (right - left + 1) / 2;
        if scan_offsets[mid] <= index {
            left = mid;
        } else {
            right = mid - 1;
        }
    }
    left
}

fn process_peaks_batch(
    tof_batch: &[u32],
    intensity_batch: &[u32],
    indices: &[usize],
    scan_offsets: &[usize],
    rt_min: f32,
    frame_index: u32,
    mz_cv: &impl ConvertableDomain,
    im_cv: &impl ConvertableDomain,
    output: &mut TimsTOFData,
) {
    let batch_size = tof_batch.len();
    
    let mut mz_buffer = Vec::with_capacity(batch_size);
    let mut im_buffer = Vec::with_capacity(batch_size);
    let mut scan_buffer = Vec::with_capacity(batch_size);
    
    for (i, &tof) in tof_batch.iter().enumerate() {
        mz_buffer.push(mz_cv.convert(tof as f64) as f32);
        let scan = find_scan_for_index_binary(indices[i], scan_offsets);
        scan_buffer.push(scan as u32);
        im_buffer.push(im_cv.convert(scan as f64) as f32);
    }
    
    output.rt_values_min.extend(vec![rt_min; batch_size]);
    output.mobility_values.extend(im_buffer);
    output.mz_values.extend(mz_buffer);
    output.intensity_values.extend_from_slice(intensity_batch);
    output.frame_indices.extend(vec![frame_index; batch_size]);
    output.scan_indices.extend(scan_buffer);
}

pub fn read_timstof_data(d_folder: &Path) -> Result<TimsTOFRawData, Box<dyn Error>> {
    let total_start = Instant::now();
    
    println!("Initializing metadata readers...");
    let meta_start = Instant::now();
    let tdf_path = d_folder.join("analysis.tdf");
    let meta = MetadataReader::new(&tdf_path)?;
    let mz_cv = Arc::new(meta.mz_converter);
    let im_cv = Arc::new(meta.im_converter);
    println!("  Metadata initialization: {:.3}s", meta_start.elapsed().as_secs_f32());
    
    println!("Initializing frame reader...");
    let frame_reader_start = Instant::now();
    let frames = FrameReader::new(d_folder)?;
    let n_frames = frames.len();
    println!("  Frame reader initialization: {:.3}s", frame_reader_start.elapsed().as_secs_f32());
    println!("  Total frames to process: {}", n_frames);
    
    println!("Processing frames with SIMD batch processing ({} threads, batch size {})...", 
            NUM_THREADS, BATCH_SIZE);
    let process_start = Instant::now();
    
    let splits: Vec<FrameSplit> = (0..n_frames).into_par_iter().map(|idx| {
        let frame = frames.get(idx).expect("frame read");
        let rt_min = frame.rt_in_seconds as f32 / 60.0;
        let mut ms1 = TimsTOFData::new();
        let mut ms2_pairs: Vec<((u32,u32), TimsTOFData)> = Vec::new();
        
        match frame.ms_level {
            MSLevel::MS1 => {
                let n_peaks = frame.tof_indices.len();
                ms1 = TimsTOFData::with_capacity(n_peaks);
                
                let mut i = 0;
                while i < n_peaks {
                    let batch_end = std::cmp::min(i + BATCH_SIZE, n_peaks);
                    let batch_indices: Vec<usize> = (i..batch_end).collect();
                    
                    process_peaks_batch(
                        &frame.tof_indices[i..batch_end],
                        &frame.intensities[i..batch_end],
                        &batch_indices,
                        &frame.scan_offsets,
                        rt_min,
                        frame.index as u32,
                        &*mz_cv,
                        &*im_cv,
                        &mut ms1,
                    );
                    
                    i = batch_end;
                }
            }
            MSLevel::MS2 => {
                let qs = &frame.quadrupole_settings;
                ms2_pairs.reserve(qs.isolation_mz.len());
                
                for win in 0..qs.isolation_mz.len() {
                    if win >= qs.isolation_width.len() { break; }
                    let prec_mz = qs.isolation_mz[win] as f32;
                    let width = qs.isolation_width[win] as f32;
                    let low = prec_mz - width * 0.5;
                    let high = prec_mz + width * 0.5;
                    let key = ((low * 10_000.0).round() as u32, (high * 10_000.0).round() as u32);
                    
                    let mut td = TimsTOFData::new();
                    let mut batch_tof = Vec::with_capacity(BATCH_SIZE);
                    let mut batch_intensity = Vec::with_capacity(BATCH_SIZE);
                    let mut batch_indices = Vec::with_capacity(BATCH_SIZE);
                    
                    for (p_idx, (&tof, &intensity)) in frame.tof_indices.iter()
                        .zip(frame.intensities.iter()).enumerate() {
                        let scan = find_scan_for_index_binary(p_idx, &frame.scan_offsets);
                        if scan < qs.scan_starts[win] || scan > qs.scan_ends[win] { continue; }
                        
                        batch_tof.push(tof);
                        batch_intensity.push(intensity);
                        batch_indices.push(p_idx);
                        
                        if batch_tof.len() == BATCH_SIZE {
                            process_peaks_batch(
                                &batch_tof,
                                &batch_intensity,
                                &batch_indices,
                                &frame.scan_offsets,
                                rt_min,
                                frame.index as u32,
                                &*mz_cv,
                                &*im_cv,
                                &mut td,
                            );
                            batch_tof.clear();
                            batch_intensity.clear();
                            batch_indices.clear();
                        }
                    }
                    
                    if !batch_tof.is_empty() {
                        process_peaks_batch(
                            &batch_tof,
                            &batch_intensity,
                            &batch_indices,
                            &frame.scan_offsets,
                            rt_min,
                            frame.index as u32,
                            &*mz_cv,
                            &*im_cv,
                            &mut td,
                        );
                    }
                    
                    if !td.mz_values.is_empty() {
                        ms2_pairs.push((key, td));
                    }
                }
            }
            _ => {}
        }
        FrameSplit { ms1, ms2: ms2_pairs }
    }).collect();
    
    println!("  Frame processing: {:.3}s", process_start.elapsed().as_secs_f32());
    
    println!("Merging data...");
    let merge_start = Instant::now();
    
    let ms1_size_estimate: usize = splits.par_iter().map(|s| s.ms1.mz_values.len()).sum();
    let mut global_ms1 = TimsTOFData::with_capacity(ms1_size_estimate);
    let mut ms2_hash: HashMap<(u32,u32), TimsTOFData> = HashMap::new();
    
    for mut split in splits {
        global_ms1.merge_from(&mut split.ms1);
        
        for (key, mut td) in split.ms2 {
            ms2_hash.entry(key).or_insert_with(TimsTOFData::new).merge_from(&mut td);
        }
    }
    
    let mut ms2_vec = Vec::with_capacity(ms2_hash.len());
    for ((q_low, q_high), td) in ms2_hash {
        let low = q_low as f32 / 10_000.0;
        let high = q_high as f32 / 10_000.0;
        ms2_vec.push(((low, high), td));
    }
    
    println!("  Data merging: {:.3}s", merge_start.elapsed().as_secs_f32());
    
    println!("\n========== Data Summary ==========");
    println!("MS1 data points: {}", global_ms1.mz_values.len());
    println!("MS2 windows: {}", ms2_vec.len());
    
    let total_ms2_points: usize = ms2_vec.iter().map(|(_, td)| td.mz_values.len()).sum();
    println!("MS2 data points: {}", total_ms2_points);
    println!("Total processing time: {:.3}s", total_start.elapsed().as_secs_f32());
    
    Ok(TimsTOFRawData {
        ms1_data: global_ms1,
        ms2_windows: ms2_vec,
    })
}

fn main() -> Result<(), Box<dyn Error>> {
    rayon::ThreadPoolBuilder::new()
        .num_threads(NUM_THREADS)
        .build_global()
        .unwrap();
    
    let d_folder_path = "/storage/guotiannanLab/wangshuaiyao/006.DIABERT_TimsTOF_Rust/test_data/CAD20220207yuel_TPHP_DIA_pool1_Slot2-54_1_4382.d";
    
    let d_path = Path::new(d_folder_path);
    if !d_path.exists() {
        return Err(format!("Folder {:?} not found", d_path).into());
    }
    
    println!("========== TimsTOF .d File Reader V3 (SIMD + Batch Processing) ==========");
    println!("Data folder: {}", d_folder_path);
    println!("Parallel threads: {}", NUM_THREADS);
    println!("Batch size: {}", BATCH_SIZE);
    println!();
    
    let _raw_data = read_timstof_data(d_path)?;
    
    println!("\n========== Processing Complete ==========");
    
    Ok(())
}