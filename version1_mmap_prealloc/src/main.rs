use std::collections::HashMap;
use std::error::Error;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;
use timsrust::{converters::ConvertableDomain, readers::{FrameReader, MetadataReader}, MSLevel};
use rayon::prelude::*;

const NUM_THREADS: usize = 32;

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
        Self {
            rt_values_min: Vec::with_capacity(capacity),
            mobility_values: Vec::with_capacity(capacity),
            mz_values: Vec::with_capacity(capacity),
            intensity_values: Vec::with_capacity(capacity),
            frame_indices: Vec::with_capacity(capacity),
            scan_indices: Vec::with_capacity(capacity),
        }
    }
    
    pub fn preallocate_exact(capacity: usize) -> Self {
        let mut data = Self::with_capacity(capacity);
        data.rt_values_min.reserve_exact(capacity);
        data.mobility_values.reserve_exact(capacity);
        data.mz_values.reserve_exact(capacity);
        data.intensity_values.reserve_exact(capacity);
        data.frame_indices.reserve_exact(capacity);
        data.scan_indices.reserve_exact(capacity);
        data
    }
    
    fn merge_from(&mut self, other: &mut Self) {
        self.rt_values_min.append(&mut other.rt_values_min);
        self.mobility_values.append(&mut other.mobility_values);
        self.mz_values.append(&mut other.mz_values);
        self.intensity_values.append(&mut other.intensity_values);
        self.frame_indices.append(&mut other.frame_indices);
        self.scan_indices.append(&mut other.scan_indices);
    }
    
    fn extend_from(&mut self, other: &Self) {
        self.rt_values_min.extend_from_slice(&other.rt_values_min);
        self.mobility_values.extend_from_slice(&other.mobility_values);
        self.mz_values.extend_from_slice(&other.mz_values);
        self.intensity_values.extend_from_slice(&other.intensity_values);
        self.frame_indices.extend_from_slice(&other.frame_indices);
        self.scan_indices.extend_from_slice(&other.scan_indices);
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

#[inline]
fn quantize(x: f32) -> u32 { 
    (x * 10_000.0).round() as u32 
}

#[inline]
fn find_scan_for_index_binary(index: usize, scan_offsets: &[usize]) -> usize {
    match scan_offsets.binary_search(&index) {
        Ok(pos) => pos,
        Err(pos) => pos.saturating_sub(1),
    }
}

fn estimate_total_peaks(frames: &FrameReader) -> usize {
    let sample_size = std::cmp::min(100, frames.len());
    let sample_sum: usize = (0..sample_size)
        .into_par_iter()
        .map(|idx| {
            frames.get(idx)
                .map(|f| f.tof_indices.len())
                .unwrap_or(0)
        })
        .sum();
    
    (sample_sum * frames.len()) / sample_size
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
    
    println!("Estimating data size for pre-allocation...");
    let estimate_start = Instant::now();
    let estimated_peaks = estimate_total_peaks(&frames);
    let ms1_estimate = estimated_peaks / 2;
    let ms2_estimate = estimated_peaks / 2;
    println!("  Estimated total peaks: ~{}", estimated_peaks);
    println!("  Pre-allocation estimation: {:.3}s", estimate_start.elapsed().as_secs_f32());
    
    println!("Processing frames in parallel with {} threads...", NUM_THREADS);
    let process_start = Instant::now();
    
    let chunk_size = (n_frames + NUM_THREADS - 1) / NUM_THREADS;
    let splits: Vec<FrameSplit> = (0..n_frames)
        .into_par_iter()
        .with_min_len(chunk_size)
        .map(|idx| {
            let frame = frames.get(idx).expect("frame read");
            let rt_min = frame.rt_in_seconds as f32 / 60.0;
            let mut ms1 = TimsTOFData::new();
            let mut ms2_pairs: Vec<((u32,u32), TimsTOFData)> = Vec::new();
            
            match frame.ms_level {
                MSLevel::MS1 => {
                    let n_peaks = frame.tof_indices.len();
                    ms1 = TimsTOFData::with_capacity(n_peaks);
                    
                    let scan_offsets = &frame.scan_offsets;
                    for (p_idx, (&tof, &intensity)) in frame.tof_indices.iter()
                        .zip(frame.intensities.iter()).enumerate() {
                        let mz = mz_cv.convert(tof as f64) as f32;
                        let scan = find_scan_for_index_binary(p_idx, scan_offsets);
                        let im = im_cv.convert(scan as f64) as f32;
                        
                        ms1.rt_values_min.push(rt_min);
                        ms1.mobility_values.push(im);
                        ms1.mz_values.push(mz);
                        ms1.intensity_values.push(intensity);
                        ms1.frame_indices.push(frame.index as u32);
                        ms1.scan_indices.push(scan as u32);
                    }
                }
                MSLevel::MS2 => {
                    let qs = &frame.quadrupole_settings;
                    let n_windows = qs.isolation_mz.len();
                    ms2_pairs.reserve_exact(n_windows);
                    
                    for win in 0..n_windows {
                        if win >= qs.isolation_width.len() { break; }
                        let prec_mz = qs.isolation_mz[win] as f32;
                        let width = qs.isolation_width[win] as f32;
                        let low = prec_mz - width * 0.5;
                        let high = prec_mz + width * 0.5;
                        let key = (quantize(low), quantize(high));
                        
                        let scan_start = qs.scan_starts[win];
                        let scan_end = qs.scan_ends[win];
                        let window_peaks: usize = frame.tof_indices.iter()
                            .zip(frame.scan_offsets.windows(2))
                            .filter(|(_, window)| {
                                let scan = window[0];
                                scan >= scan_start && scan <= scan_end
                            })
                            .count();
                        
                        let mut td = TimsTOFData::with_capacity(window_peaks);
                        
                        for (p_idx, (&tof, &intensity)) in frame.tof_indices.iter()
                            .zip(frame.intensities.iter()).enumerate() {
                            let scan = find_scan_for_index_binary(p_idx, &frame.scan_offsets);
                            if scan < scan_start || scan > scan_end { continue; }
                            
                            let mz = mz_cv.convert(tof as f64) as f32;
                            let im = im_cv.convert(scan as f64) as f32;
                            td.rt_values_min.push(rt_min);
                            td.mobility_values.push(im);
                            td.mz_values.push(mz);
                            td.intensity_values.push(intensity);
                            td.frame_indices.push(frame.index as u32);
                            td.scan_indices.push(scan as u32);
                        }
                        ms2_pairs.push((key, td));
                    }
                }
                _ => {}
            }
            FrameSplit { ms1, ms2: ms2_pairs }
        })
        .collect();
    println!("  Frame processing: {:.3}s", process_start.elapsed().as_secs_f32());
    
    println!("Merging data with pre-allocated buffers...");
    let merge_start = Instant::now();
    
    let actual_ms1_size: usize = splits.par_iter().map(|s| s.ms1.mz_values.len()).sum();
    let mut global_ms1 = TimsTOFData::preallocate_exact(actual_ms1_size);
    
    let mut ms2_hash: HashMap<(u32,u32), TimsTOFData> = HashMap::with_capacity(100);
    
    for split in splits {
        global_ms1.extend_from(&split.ms1);
        
        for (key, mut td) in split.ms2 {
            ms2_hash.entry(key)
                .and_modify(|existing| existing.merge_from(&mut td))
                .or_insert(td);
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
    
    println!("========== TimsTOF .d File Reader V1 (Memory-Mapped + Pre-allocation) ==========");
    println!("Data folder: {}", d_folder_path);
    println!("Parallel threads: {}", NUM_THREADS);
    println!();
    
    let _raw_data = read_timstof_data(d_path)?;
    
    println!("\n========== Processing Complete ==========");
    
    Ok(())
}