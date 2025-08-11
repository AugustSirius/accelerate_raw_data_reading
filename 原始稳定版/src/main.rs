use std::collections::HashMap;
use std::error::Error;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;
use timsrust::{converters::ConvertableDomain, readers::{FrameReader, MetadataReader}, MSLevel};
use rayon::prelude::*;

// Data structure for raw TimsTOF data
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

// Helper functions
#[inline]
fn quantize(x: f32) -> u32 { 
    (x * 10_000.0).round() as u32 
}

fn find_scan_for_index(index: usize, scan_offsets: &[usize]) -> usize {
    for (scan, window) in scan_offsets.windows(2).enumerate() {
        if index >= window[0] && index < window[1] {
            return scan;
        }
    }
    scan_offsets.len() - 1
}

/// Read TimsTOF .d folder and return raw data with detailed timing
pub fn read_timstof_data(d_folder: &Path) -> Result<TimsTOFRawData, Box<dyn Error>> {
    let total_start = Instant::now();
    
    // Initialize metadata readers
    println!("Initializing metadata readers...");
    let meta_start = Instant::now();
    let tdf_path = d_folder.join("analysis.tdf");
    let meta = MetadataReader::new(&tdf_path)?;
    let mz_cv = Arc::new(meta.mz_converter);
    let im_cv = Arc::new(meta.im_converter);
    println!("  Metadata initialization: {:.3}s", meta_start.elapsed().as_secs_f32());
    
    // Initialize frame reader
    println!("Initializing frame reader...");
    let frame_reader_start = Instant::now();
    let frames = FrameReader::new(d_folder)?;
    let n_frames = frames.len();
    println!("  Frame reader initialization: {:.3}s", frame_reader_start.elapsed().as_secs_f32());
    println!("  Total frames to process: {}", n_frames);
    
    // Process frames in parallel
    println!("Processing frames in parallel...");
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
                for (p_idx, (&tof, &intensity)) in frame.tof_indices.iter().zip(frame.intensities.iter()).enumerate() {
                    let mz = mz_cv.convert(tof as f64) as f32;
                    let scan = find_scan_for_index(p_idx, &frame.scan_offsets);
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
                ms2_pairs.reserve(qs.isolation_mz.len());
                for win in 0..qs.isolation_mz.len() {
                    if win >= qs.isolation_width.len() { break; }
                    let prec_mz = qs.isolation_mz[win] as f32;
                    let width = qs.isolation_width[win] as f32;
                    let low = prec_mz - width * 0.5;
                    let high = prec_mz + width * 0.5;
                    let key = (quantize(low), quantize(high));
                    
                    let mut td = TimsTOFData::new();
                    for (p_idx, (&tof, &intensity)) in frame.tof_indices.iter().zip(frame.intensities.iter()).enumerate() {
                        let scan = find_scan_for_index(p_idx, &frame.scan_offsets);
                        if scan < qs.scan_starts[win] || scan > qs.scan_ends[win] { continue; }
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
    }).collect();
    println!("  Frame processing: {:.3}s", process_start.elapsed().as_secs_f32());
    
    // Merge data
    println!("Merging data...");
    let merge_start = Instant::now();
    let ms1_size_estimate: usize = splits.par_iter().map(|s| s.ms1.mz_values.len()).sum();
    let mut global_ms1 = TimsTOFData::with_capacity(ms1_size_estimate);
    let mut ms2_hash: HashMap<(u32,u32), TimsTOFData> = HashMap::new();
    
    for split in splits {
        global_ms1.rt_values_min.extend(split.ms1.rt_values_min);
        global_ms1.mobility_values.extend(split.ms1.mobility_values);
        global_ms1.mz_values.extend(split.ms1.mz_values);
        global_ms1.intensity_values.extend(split.ms1.intensity_values);
        global_ms1.frame_indices.extend(split.ms1.frame_indices);
        global_ms1.scan_indices.extend(split.ms1.scan_indices);
        
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
    
    // Print summary statistics
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
    // Configure parallel processing
    rayon::ThreadPoolBuilder::new()
        .num_threads(16)
        .build_global()
        .unwrap();
    
    // Set path to your .d folder
    let d_folder_path = if cfg!(target_os = "macos") {
        "/Users/augustsirius/Desktop/DIA_peak_group_extraction/输入数据文件/raw_data/CAD20220207yuel_TPHP_DIA_pool1_Slot2-54_1_4382.d"
    } else {
        "/storage/guotiannanLab/wangshuaiyao/006.DIABERT_TimsTOF_Rust/test_data/CAD20220207yuel_TPHP_DIA_pool1_Slot2-54_1_4382.d"
    };
    
    let d_path = Path::new(d_folder_path);
    if !d_path.exists() {
        return Err(format!("Folder {:?} not found", d_path).into());
    }
    
    println!("========== TimsTOF .d File Reader ==========");
    println!("Data folder: {}", d_folder_path);
    println!("Parallel threads: 16");
    println!();
    
    // Read the data with timing
    let _raw_data = read_timstof_data(d_path)?;
    
    println!("\n========== Processing Complete ==========");
    
    Ok(())
}

// Cargo.toml dependencies needed:
/*
[dependencies]
timsrust = "0.4"
rayon = "1.7"

[profile.release]
opt-level = 3
lto = true
codegen-units = 1
*/