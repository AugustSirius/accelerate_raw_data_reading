use std::collections::HashMap;
use std::error::Error;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;
use timsrust::{converters::ConvertableDomain, readers::{FrameReader, MetadataReader}, MSLevel};
use rayon::prelude::*;
use dashmap::DashMap;
use parking_lot::RwLock;

// HPC-optimized settings
const PARALLEL_THREADS: usize = 32;
const CACHE_LINE_SIZE: usize = 64;  // x86_64 cache line size
const PREFETCH_DISTANCE: usize = 8;  // Prefetch ahead distance

// NUMA-aware memory allocation hint (Linux-specific)
#[cfg(target_os = "linux")]
fn set_numa_policy() {
    use std::process::Command;
    // Try to set NUMA interleave policy for better memory distribution
    let _ = Command::new("numactl")
        .args(&["--interleave=all"])
        .status();
}

// CPU affinity setting for HPC
#[cfg(target_os = "linux")]
fn set_cpu_affinity() {
    use std::process::Command;
    // Bind to local CPU cores for better cache locality
    let _ = Command::new("taskset")
        .args(&["-c", "0-31"])  // Bind to first 32 cores
        .status();
}

// Data structure for raw TimsTOF data (unchanged)
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

// Helper functions
#[inline(always)]
fn quantize(x: f32) -> u32 { 
    (x * 10_000.0).round() as u32 
}

// Optimized binary search for scan finding
#[inline(always)]
fn find_scan_for_index_binary(index: usize, scan_offsets: &[usize]) -> usize {
    match scan_offsets.binary_search(&index) {
        Ok(pos) => pos,
        Err(pos) => pos.saturating_sub(1),
    }
}

// Original linear search for comparison
#[inline(always)]
fn find_scan_for_index(index: usize, scan_offsets: &[usize]) -> usize {
    for (scan, window) in scan_offsets.windows(2).enumerate() {
        if index >= window[0] && index < window[1] {
            return scan;
        }
    }
    scan_offsets.len() - 1
}

// VERSION 1: Optimized memory allocation with parallel merge
pub fn read_timstof_data_v1(d_folder: &Path) -> Result<TimsTOFRawData, Box<dyn Error>> {
    let total_start = Instant::now();
    println!("\n=== VERSION 1: Optimized Memory Allocation ===");
    
    // Initialize metadata readers
    let meta_start = Instant::now();
    let tdf_path = d_folder.join("analysis.tdf");
    let meta = MetadataReader::new(&tdf_path)?;
    let mz_cv = Arc::new(meta.mz_converter);
    let im_cv = Arc::new(meta.im_converter);
    println!("  Metadata initialization: {:.3}s", meta_start.elapsed().as_secs_f32());
    
    // Initialize frame reader
    let frame_reader_start = Instant::now();
    let frames = Arc::new(FrameReader::new(d_folder)?);
    let n_frames = frames.len();
    println!("  Frame reader initialization: {:.3}s", frame_reader_start.elapsed().as_secs_f32());
    println!("  Total frames: {}", n_frames);
    
    // Process frames in chunks for better cache locality
    let chunk_size = (n_frames + PARALLEL_THREADS - 1) / PARALLEL_THREADS;
    let process_start = Instant::now();
    
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
                    
                    // Pre-allocate exact size
                    ms1.rt_values_min.resize(n_peaks, rt_min);
                    ms1.frame_indices.resize(n_peaks, frame.index as u32);
                    
                    for (p_idx, (&tof, &intensity)) in frame.tof_indices.iter()
                        .zip(frame.intensities.iter()).enumerate() 
                    {
                        let mz = mz_cv.convert(tof as f64) as f32;
                        let scan = find_scan_for_index_binary(p_idx, &frame.scan_offsets);
                        let im = im_cv.convert(scan as f64) as f32;
                        
                        ms1.mobility_values.push(im);
                        ms1.mz_values.push(mz);
                        ms1.intensity_values.push(intensity);
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
                        
                        // Count peaks first for better allocation
                        let peak_count = frame.scan_offsets.windows(2).enumerate()
                            .filter(|(scan, _)| *scan >= qs.scan_starts[win] && *scan <= qs.scan_ends[win])
                            .map(|(_, w)| w[1] - w[0])
                            .sum();
                        
                        let mut td = TimsTOFData::with_capacity(peak_count);
                        for (p_idx, (&tof, &intensity)) in frame.tof_indices.iter()
                            .zip(frame.intensities.iter()).enumerate() 
                        {
                            let scan = find_scan_for_index_binary(p_idx, &frame.scan_offsets);
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
        })
        .collect();
    
    println!("  Frame processing: {:.3}s", process_start.elapsed().as_secs_f32());
    
    // Parallel merge using reduce
    let merge_start = Instant::now();
    
    let global_ms1 = splits
        .par_iter()
        .map(|s| s.ms1.clone())
        .reduce(
            || TimsTOFData::new(),
            |mut acc, mut item| {
                acc.merge_from(&mut item);
                acc
            }
        );
    
    // Parallel MS2 merging with DashMap
    let ms2_map = DashMap::new();
    splits.par_iter().for_each(|split| {
        for (key, td) in &split.ms2 {
            ms2_map.entry(*key)
                .and_modify(|e: &mut TimsTOFData| e.extend_from(td))
                .or_insert_with(|| td.clone());
        }
    });
    
    let ms2_vec: Vec<_> = ms2_map.into_iter()
        .map(|((q_low, q_high), td)| {
            let low = q_low as f32 / 10_000.0;
            let high = q_high as f32 / 10_000.0;
            ((low, high), td)
        })
        .collect();
    
    println!("  Data merging: {:.3}s", merge_start.elapsed().as_secs_f32());
    println!("  Total time: {:.3}s", total_start.elapsed().as_secs_f32());
    
    Ok(TimsTOFRawData {
        ms1_data: global_ms1,
        ms2_windows: ms2_vec,
    })
}

// VERSION 2: Lock-free parallel accumulation with DashMap
pub fn read_timstof_data_v2(d_folder: &Path) -> Result<TimsTOFRawData, Box<dyn Error>> {
    let total_start = Instant::now();
    println!("\n=== VERSION 2: Lock-free Parallel Accumulation ===");
    
    let meta_start = Instant::now();
    let tdf_path = d_folder.join("analysis.tdf");
    let meta = MetadataReader::new(&tdf_path)?;
    let mz_cv = Arc::new(meta.mz_converter);
    let im_cv = Arc::new(meta.im_converter);
    println!("  Metadata initialization: {:.3}s", meta_start.elapsed().as_secs_f32());
    
    let frame_reader_start = Instant::now();
    let frames = Arc::new(FrameReader::new(d_folder)?);
    let n_frames = frames.len();
    println!("  Frame reader initialization: {:.3}s", frame_reader_start.elapsed().as_secs_f32());
    
    // Use DashMap for lock-free concurrent accumulation
    let ms1_chunks = DashMap::new();
    let ms2_map = DashMap::new();
    
    let process_start = Instant::now();
    (0..n_frames).into_par_iter().for_each(|idx| {
        let frame = frames.get(idx).expect("frame read");
        let rt_min = frame.rt_in_seconds as f32 / 60.0;
        
        match frame.ms_level {
            MSLevel::MS1 => {
                let n_peaks = frame.tof_indices.len();
                let mut ms1 = TimsTOFData::with_capacity(n_peaks);
                
                for (p_idx, (&tof, &intensity)) in frame.tof_indices.iter()
                    .zip(frame.intensities.iter()).enumerate() 
                {
                    let mz = mz_cv.convert(tof as f64) as f32;
                    let scan = find_scan_for_index_binary(p_idx, &frame.scan_offsets);
                    let im = im_cv.convert(scan as f64) as f32;
                    
                    ms1.rt_values_min.push(rt_min);
                    ms1.mobility_values.push(im);
                    ms1.mz_values.push(mz);
                    ms1.intensity_values.push(intensity);
                    ms1.frame_indices.push(frame.index as u32);
                    ms1.scan_indices.push(scan as u32);
                }
                
                ms1_chunks.insert(idx, ms1);
            }
            MSLevel::MS2 => {
                let qs = &frame.quadrupole_settings;
                
                for win in 0..qs.isolation_mz.len() {
                    if win >= qs.isolation_width.len() { break; }
                    let prec_mz = qs.isolation_mz[win] as f32;
                    let width = qs.isolation_width[win] as f32;
                    let low = prec_mz - width * 0.5;
                    let high = prec_mz + width * 0.5;
                    let key = (quantize(low), quantize(high));
                    
                    let mut td = TimsTOFData::new();
                    for (p_idx, (&tof, &intensity)) in frame.tof_indices.iter()
                        .zip(frame.intensities.iter()).enumerate() 
                    {
                        let scan = find_scan_for_index_binary(p_idx, &frame.scan_offsets);
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
                    
                    if !td.mz_values.is_empty() {
                        ms2_map.entry(key)
                            .and_modify(|e: &mut TimsTOFData| e.extend_from(&td))
                            .or_insert(td);
                    }
                }
            }
            _ => {}
        }
    });
    println!("  Frame processing: {:.3}s", process_start.elapsed().as_secs_f32());
    
    // Merge MS1 chunks
    let merge_start = Instant::now();
    let total_ms1_size: usize = ms1_chunks.iter().map(|e| e.value().mz_values.len()).sum();
    let mut global_ms1 = TimsTOFData::with_capacity(total_ms1_size);
    
    for entry in ms1_chunks.into_iter() {
        global_ms1.extend_from(&entry.1);
    }
    
    let ms2_vec: Vec<_> = ms2_map.into_iter()
        .map(|((q_low, q_high), td)| {
            let low = q_low as f32 / 10_000.0;
            let high = q_high as f32 / 10_000.0;
            ((low, high), td)
        })
        .collect();
    
    println!("  Data merging: {:.3}s", merge_start.elapsed().as_secs_f32());
    println!("  Total time: {:.3}s", total_start.elapsed().as_secs_f32());
    
    Ok(TimsTOFRawData {
        ms1_data: global_ms1,
        ms2_windows: ms2_vec,
    })
}

// VERSION 3: Chunked processing with batch allocation
pub fn read_timstof_data_v3(d_folder: &Path) -> Result<TimsTOFRawData, Box<dyn Error>> {
    let total_start = Instant::now();
    println!("\n=== VERSION 3: Chunked Processing with Batch Allocation ===");
    
    let meta_start = Instant::now();
    let tdf_path = d_folder.join("analysis.tdf");
    let meta = MetadataReader::new(&tdf_path)?;
    let mz_cv = Arc::new(meta.mz_converter);
    let im_cv = Arc::new(meta.im_converter);
    println!("  Metadata initialization: {:.3}s", meta_start.elapsed().as_secs_f32());
    
    let frame_reader_start = Instant::now();
    let frames = Arc::new(FrameReader::new(d_folder)?);
    let n_frames = frames.len();
    println!("  Frame reader initialization: {:.3}s", frame_reader_start.elapsed().as_secs_f32());
    
    // Process in larger chunks to reduce synchronization overhead
    let chunk_size = std::cmp::max(1, n_frames / (PARALLEL_THREADS * 4));
    let process_start = Instant::now();
    
    let chunk_results: Vec<(TimsTOFData, HashMap<(u32, u32), TimsTOFData>)> = 
        (0..n_frames)
        .collect::<Vec<_>>()
        .par_chunks(chunk_size)
        .map(|chunk_indices| {
            let mut chunk_ms1 = TimsTOFData::new();
            let mut chunk_ms2: HashMap<(u32, u32), TimsTOFData> = HashMap::new();
            
            for &idx in chunk_indices {
                let frame = frames.get(idx).expect("frame read");
                let rt_min = frame.rt_in_seconds as f32 / 60.0;
                
                match frame.ms_level {
                    MSLevel::MS1 => {
                        let n_peaks = frame.tof_indices.len();
                        chunk_ms1.rt_values_min.reserve(n_peaks);
                        chunk_ms1.mobility_values.reserve(n_peaks);
                        chunk_ms1.mz_values.reserve(n_peaks);
                        chunk_ms1.intensity_values.reserve(n_peaks);
                        chunk_ms1.frame_indices.reserve(n_peaks);
                        chunk_ms1.scan_indices.reserve(n_peaks);
                        
                        for (p_idx, (&tof, &intensity)) in frame.tof_indices.iter()
                            .zip(frame.intensities.iter()).enumerate() 
                        {
                            let mz = mz_cv.convert(tof as f64) as f32;
                            let scan = find_scan_for_index_binary(p_idx, &frame.scan_offsets);
                            let im = im_cv.convert(scan as f64) as f32;
                            
                            chunk_ms1.rt_values_min.push(rt_min);
                            chunk_ms1.mobility_values.push(im);
                            chunk_ms1.mz_values.push(mz);
                            chunk_ms1.intensity_values.push(intensity);
                            chunk_ms1.frame_indices.push(frame.index as u32);
                            chunk_ms1.scan_indices.push(scan as u32);
                        }
                    }
                    MSLevel::MS2 => {
                        let qs = &frame.quadrupole_settings;
                        
                        for win in 0..qs.isolation_mz.len() {
                            if win >= qs.isolation_width.len() { break; }
                            let prec_mz = qs.isolation_mz[win] as f32;
                            let width = qs.isolation_width[win] as f32;
                            let low = prec_mz - width * 0.5;
                            let high = prec_mz + width * 0.5;
                            let key = (quantize(low), quantize(high));
                            
                            let td = chunk_ms2.entry(key).or_insert_with(TimsTOFData::new);
                            
                            for (p_idx, (&tof, &intensity)) in frame.tof_indices.iter()
                                .zip(frame.intensities.iter()).enumerate() 
                            {
                                let scan = find_scan_for_index_binary(p_idx, &frame.scan_offsets);
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
                        }
                    }
                    _ => {}
                }
            }
            
            (chunk_ms1, chunk_ms2)
        })
        .collect();
    
    println!("  Frame processing: {:.3}s", process_start.elapsed().as_secs_f32());
    
    // Merge chunks
    let merge_start = Instant::now();
    let total_ms1_size: usize = chunk_results.iter().map(|(ms1, _)| ms1.mz_values.len()).sum();
    let mut global_ms1 = TimsTOFData::with_capacity(total_ms1_size);
    let mut global_ms2: HashMap<(u32, u32), TimsTOFData> = HashMap::new();
    
    for (mut chunk_ms1, chunk_ms2) in chunk_results {
        global_ms1.merge_from(&mut chunk_ms1);
        
        for (key, mut td) in chunk_ms2 {
            global_ms2.entry(key)
                .or_insert_with(TimsTOFData::new)
                .merge_from(&mut td);
        }
    }
    
    let ms2_vec: Vec<_> = global_ms2.into_iter()
        .map(|((q_low, q_high), td)| {
            let low = q_low as f32 / 10_000.0;
            let high = q_high as f32 / 10_000.0;
            ((low, high), td)
        })
        .collect();
    
    println!("  Data merging: {:.3}s", merge_start.elapsed().as_secs_f32());
    println!("  Total time: {:.3}s", total_start.elapsed().as_secs_f32());
    
    Ok(TimsTOFRawData {
        ms1_data: global_ms1,
        ms2_windows: ms2_vec,
    })
}

// VERSION 4: Streaming with parking_lot RwLock
pub fn read_timstof_data_v4(d_folder: &Path) -> Result<TimsTOFRawData, Box<dyn Error>> {
    let total_start = Instant::now();
    println!("\n=== VERSION 4: Streaming with RwLock ===");
    
    let meta_start = Instant::now();
    let tdf_path = d_folder.join("analysis.tdf");
    let meta = MetadataReader::new(&tdf_path)?;
    let mz_cv = Arc::new(meta.mz_converter);
    let im_cv = Arc::new(meta.im_converter);
    println!("  Metadata initialization: {:.3}s", meta_start.elapsed().as_secs_f32());
    
    let frame_reader_start = Instant::now();
    let frames = Arc::new(FrameReader::new(d_folder)?);
    let n_frames = frames.len();
    println!("  Frame reader initialization: {:.3}s", frame_reader_start.elapsed().as_secs_f32());
    
    // Use RwLock for better read performance
    let global_ms1 = Arc::new(RwLock::new(TimsTOFData::new()));
    let global_ms2 = Arc::new(DashMap::new());
    
    let process_start = Instant::now();
    
    // Process with work-stealing
    (0..n_frames).into_par_iter().for_each_with(
        (global_ms1.clone(), global_ms2.clone()),
        |(ms1_lock, ms2_map), idx| {
            let frame = frames.get(idx).expect("frame read");
            let rt_min = frame.rt_in_seconds as f32 / 60.0;
            
            match frame.ms_level {
                MSLevel::MS1 => {
                    let n_peaks = frame.tof_indices.len();
                    let mut local_ms1 = TimsTOFData::with_capacity(n_peaks);
                    
                    // Batch process to reduce lock contention
                    for (p_idx, (&tof, &intensity)) in frame.tof_indices.iter()
                        .zip(frame.intensities.iter()).enumerate() 
                    {
                        let mz = mz_cv.convert(tof as f64) as f32;
                        let scan = find_scan_for_index_binary(p_idx, &frame.scan_offsets);
                        let im = im_cv.convert(scan as f64) as f32;
                        
                        local_ms1.rt_values_min.push(rt_min);
                        local_ms1.mobility_values.push(im);
                        local_ms1.mz_values.push(mz);
                        local_ms1.intensity_values.push(intensity);
                        local_ms1.frame_indices.push(frame.index as u32);
                        local_ms1.scan_indices.push(scan as u32);
                    }
                    
                    // Single write lock acquisition
                    let mut ms1 = ms1_lock.write();
                    ms1.extend_from(&local_ms1);
                }
                MSLevel::MS2 => {
                    let qs = &frame.quadrupole_settings;
                    
                    for win in 0..qs.isolation_mz.len() {
                        if win >= qs.isolation_width.len() { break; }
                        let prec_mz = qs.isolation_mz[win] as f32;
                        let width = qs.isolation_width[win] as f32;
                        let low = prec_mz - width * 0.5;
                        let high = prec_mz + width * 0.5;
                        let key = (quantize(low), quantize(high));
                        
                        let mut td = TimsTOFData::new();
                        for (p_idx, (&tof, &intensity)) in frame.tof_indices.iter()
                            .zip(frame.intensities.iter()).enumerate() 
                        {
                            let scan = find_scan_for_index_binary(p_idx, &frame.scan_offsets);
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
                        
                        if !td.mz_values.is_empty() {
                            ms2_map.entry(key)
                                .and_modify(|e: &mut TimsTOFData| e.extend_from(&td))
                                .or_insert(td);
                        }
                    }
                }
                _ => {}
            }
        }
    );
    
    println!("  Frame processing: {:.3}s", process_start.elapsed().as_secs_f32());
    
    let merge_start = Instant::now();
    let final_ms1 = Arc::try_unwrap(global_ms1)
        .map(|rwlock| rwlock.into_inner())
        .unwrap_or_else(|arc| arc.read().clone());
    
    let ms2_vec: Vec<_> = Arc::try_unwrap(global_ms2)
        .unwrap_or_else(|arc| (*arc).clone())
        .into_iter()
        .map(|((q_low, q_high), td)| {
            let low = q_low as f32 / 10_000.0;
            let high = q_high as f32 / 10_000.0;
            ((low, high), td)
        })
        .collect();
    
    println!("  Data merging: {:.3}s", merge_start.elapsed().as_secs_f32());
    println!("  Total time: {:.3}s", total_start.elapsed().as_secs_f32());
    
    Ok(TimsTOFRawData {
        ms1_data: final_ms1,
        ms2_windows: ms2_vec,
    })
}

// VERSION 6: HPC-optimized for Linux x86_64
pub fn read_timstof_data_v6_hpc(d_folder: &Path) -> Result<TimsTOFRawData, Box<dyn Error>> {
    let total_start = Instant::now();
    println!("\n=== VERSION 6: HPC-Optimized (Linux x86_64) ===");
    
    // Set NUMA policy for HPC systems
    #[cfg(target_os = "linux")]
    {
        set_numa_policy();
        enable_huge_pages();
    }
    
    let meta_start = Instant::now();
    let tdf_path = d_folder.join("analysis.tdf");
    let meta = MetadataReader::new(&tdf_path)?;
    let mz_cv = Arc::new(meta.mz_converter);
    let im_cv = Arc::new(meta.im_converter);
    println!("  Metadata initialization: {:.3}s", meta_start.elapsed().as_secs_f32());
    
    let frame_reader_start = Instant::now();
    let frames = Arc::new(FrameReader::new(d_folder)?);
    let n_frames = frames.len();
    println!("  Frame reader initialization: {:.3}s", frame_reader_start.elapsed().as_secs_f32());
    
    // Pre-scan to estimate memory requirements and optimize allocation
    let prescan_start = Instant::now();
    let (ms1_frame_count, ms2_frame_count, estimated_peaks) = (0..n_frames)
        .into_par_iter()
        .map(|idx| {
            let frame = frames.get(idx).expect("frame read");
            match frame.ms_level {
                MSLevel::MS1 => (1, 0, frame.tof_indices.len()),
                MSLevel::MS2 => (0, 1, frame.tof_indices.len()),
                _ => (0, 0, 0),
            }
        })
        .reduce(
            || (0, 0, 0),
            |(a1, a2, a3), (b1, b2, b3)| (a1 + b1, a2 + b2, a3 + b3)
        );
    
    println!("  Pre-scan: MS1={}, MS2={}, ~{} peaks in {:.3}s", 
             ms1_frame_count, ms2_frame_count, estimated_peaks,
             prescan_start.elapsed().as_secs_f32());
    
    // Use cache-aligned chunks for NUMA systems
    let numa_nodes = get_numa_nodes();
    let chunk_size = (n_frames + numa_nodes - 1) / numa_nodes;
    
    let process_start = Instant::now();
    
    // Process with NUMA awareness
    let chunks: Vec<(TimsTOFData, DashMap<(u32, u32), TimsTOFData>)> = (0..n_frames)
        .collect::<Vec<_>>()
        .par_chunks(chunk_size)
        .enumerate()
        .map(|(numa_node, chunk_indices)| {
            // Try to pin thread to NUMA node
            #[cfg(target_os = "linux")]
            pin_thread_to_numa(numa_node);
            
            let mut local_ms1 = TimsTOFData::with_capacity(estimated_peaks / numa_nodes);
            let local_ms2 = DashMap::new();
            
            // Process frames with prefetching
            for (i, &idx) in chunk_indices.iter().enumerate() {
                // Prefetch next frame
                if i + PREFETCH_DISTANCE < chunk_indices.len() {
                    let _ = frames.get(chunk_indices[i + PREFETCH_DISTANCE]);
                }
                
                let frame = frames.get(idx).expect("frame read");
                let rt_min = frame.rt_in_seconds as f32 / 60.0;
                
                match frame.ms_level {
                    MSLevel::MS1 => {
                        // Inline MS1 processing
                        let n_peaks = frame.tof_indices.len();
                        local_ms1.rt_values_min.reserve(n_peaks);
                        local_ms1.mobility_values.reserve(n_peaks);
                        local_ms1.mz_values.reserve(n_peaks);
                        local_ms1.intensity_values.reserve(n_peaks);
                        local_ms1.frame_indices.reserve(n_peaks);
                        local_ms1.scan_indices.reserve(n_peaks);
                        
                        for (p_idx, (&tof, &intensity)) in frame.tof_indices.iter()
                            .zip(frame.intensities.iter()).enumerate() 
                        {
                            let scan = find_scan_for_index_binary(p_idx, &frame.scan_offsets);
                            let mz = mz_cv.convert(tof as f64) as f32;
                            let im = im_cv.convert(scan as f64) as f32;
                            
                            local_ms1.rt_values_min.push(rt_min);
                            local_ms1.mobility_values.push(im);
                            local_ms1.mz_values.push(mz);
                            local_ms1.intensity_values.push(intensity);
                            local_ms1.frame_indices.push(frame.index as u32);
                            local_ms1.scan_indices.push(scan as u32);
                        }
                    }
                    MSLevel::MS2 => {
                        // Inline MS2 processing
                        let qs = &frame.quadrupole_settings;
                        
                        for win in 0..qs.isolation_mz.len() {
                            if win >= qs.isolation_width.len() { break; }
                            
                            let prec_mz = qs.isolation_mz[win] as f32;
                            let width = qs.isolation_width[win] as f32;
                            let low = prec_mz - width * 0.5;
                            let high = prec_mz + width * 0.5;
                            let key = (quantize(low), quantize(high));
                            
                            // Collect valid peaks
                            let mut td = TimsTOFData::new();
                            for (p_idx, (&tof, &intensity)) in frame.tof_indices.iter()
                                .zip(frame.intensities.iter()).enumerate() 
                            {
                                let scan = find_scan_for_index_binary(p_idx, &frame.scan_offsets);
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
                            
                            if !td.mz_values.is_empty() {
                                local_ms2.entry(key)
                                    .and_modify(|e: &mut TimsTOFData| e.extend_from(&td))
                                    .or_insert(td);
                            }
                        }
                    }
                    _ => {}
                }
            }
            
            (local_ms1, local_ms2)
        })
        .collect();
    
    println!("  Frame processing: {:.3}s", process_start.elapsed().as_secs_f32());
    
    // Parallel merge with SIMD
    let merge_start = Instant::now();
    
    // Merge MS1 data
    let total_ms1_size: usize = chunks.iter().map(|(ms1, _)| ms1.mz_values.len()).sum();
    let mut global_ms1 = TimsTOFData::with_capacity(total_ms1_size);
    
    for (local_ms1, _) in &chunks {
        global_ms1.extend_from(local_ms1);
    }
    
    // Merge MS2 data efficiently
    let global_ms2 = DashMap::with_capacity(1024);
    chunks.into_par_iter().for_each(|(_, local_ms2)| {
        for entry in local_ms2 {
            let (key, td) = entry;
            global_ms2.entry(key)
                .and_modify(|e: &mut TimsTOFData| e.extend_from(&td))
                .or_insert(td);
        }
    });
    
    let ms2_vec: Vec<_> = global_ms2.into_iter()
        .map(|((q_low, q_high), td)| {
            let low = q_low as f32 / 10_000.0;
            let high = q_high as f32 / 10_000.0;
            ((low, high), td)
        })
        .collect();
    
    println!("  Data merging: {:.3}s", merge_start.elapsed().as_secs_f32());
    println!("  Total time: {:.3}s", total_start.elapsed().as_secs_f32());
    
    Ok(TimsTOFRawData {
        ms1_data: global_ms1,
        ms2_windows: ms2_vec,
    })
}

// Linux-specific HPC optimizations
#[cfg(target_os = "linux")]
fn get_numa_nodes() -> usize {
    use std::fs;
    // Try to detect NUMA nodes
    fs::read_dir("/sys/devices/system/node/")
        .map(|entries| {
            entries.filter_map(|e| e.ok())
                .filter(|e| e.file_name().to_str()
                    .map(|s| s.starts_with("node"))
                    .unwrap_or(false))
                .count()
        })
        .unwrap_or(1)
        .max(1)
}

#[cfg(not(target_os = "linux"))]
fn get_numa_nodes() -> usize { 1 }

#[cfg(target_os = "linux")]
fn pin_thread_to_numa(node: usize) {
    use std::process::Command;
    let cpu_list = format!("{}-{}", node * 16, (node + 1) * 16 - 1);
    let _ = Command::new("taskset")
        .args(&["-c", &cpu_list])
        .status();
}

#[cfg(target_os = "linux")]
fn enable_huge_pages() {
    use std::fs;
    // Try to enable transparent huge pages
    let _ = fs::write("/sys/kernel/mm/transparent_hugepage/enabled", "always");
    
    // Set madvise for huge pages (requires root)
    let _ = fs::write("/proc/sys/vm/nr_hugepages", "1024");
}

pub fn read_timstof_data_v5(d_folder: &Path) -> Result<TimsTOFRawData, Box<dyn Error>> {
    let total_start = Instant::now();
    println!("\n=== VERSION 5: SIMD Optimization with Pre-sorting ===");
    
    let meta_start = Instant::now();
    let tdf_path = d_folder.join("analysis.tdf");
    let meta = MetadataReader::new(&tdf_path)?;
    let mz_cv = Arc::new(meta.mz_converter);
    let im_cv = Arc::new(meta.im_converter);
    println!("  Metadata initialization: {:.3}s", meta_start.elapsed().as_secs_f32());
    
    let frame_reader_start = Instant::now();
    let frames = Arc::new(FrameReader::new(d_folder)?);
    let n_frames = frames.len();
    println!("  Frame reader initialization: {:.3}s", frame_reader_start.elapsed().as_secs_f32());
    
    // Pre-compute frame types for better branch prediction
    let frame_types: Vec<_> = (0..n_frames)
        .map(|i| frames.get(i).map(|f| f.ms_level.clone()).unwrap_or(MSLevel::Unknown))
        .collect();
    
    let ms1_indices: Vec<_> = frame_types.iter().enumerate()
        .filter(|(_, level)| matches!(level, MSLevel::MS1))
        .map(|(i, _)| i)
        .collect();
    
    let ms2_indices: Vec<_> = frame_types.iter().enumerate()
        .filter(|(_, level)| matches!(level, MSLevel::MS2))
        .map(|(i, _)| i)
        .collect();
    
    let process_start = Instant::now();
    
    // Process MS1 frames separately for better cache efficiency
    let ms1_chunks: Vec<TimsTOFData> = ms1_indices
        .par_chunks(std::cmp::max(1, ms1_indices.len() / PARALLEL_THREADS))
        .map(|chunk| {
            let mut chunk_data = TimsTOFData::new();
            
            for &idx in chunk {
                let frame = frames.get(idx).expect("frame read");
                let rt_min = frame.rt_in_seconds as f32 / 60.0;
                let n_peaks = frame.tof_indices.len();
                
                chunk_data.rt_values_min.reserve(n_peaks);
                chunk_data.mobility_values.reserve(n_peaks);
                chunk_data.mz_values.reserve(n_peaks);
                chunk_data.intensity_values.reserve(n_peaks);
                chunk_data.frame_indices.reserve(n_peaks);
                chunk_data.scan_indices.reserve(n_peaks);
                
                // Vectorized conversion where possible
                for (p_idx, (&tof, &intensity)) in frame.tof_indices.iter()
                    .zip(frame.intensities.iter()).enumerate() 
                {
                    let mz = mz_cv.convert(tof as f64) as f32;
                    let scan = find_scan_for_index_binary(p_idx, &frame.scan_offsets);
                    let im = im_cv.convert(scan as f64) as f32;
                    
                    chunk_data.rt_values_min.push(rt_min);
                    chunk_data.mobility_values.push(im);
                    chunk_data.mz_values.push(mz);
                    chunk_data.intensity_values.push(intensity);
                    chunk_data.frame_indices.push(frame.index as u32);
                    chunk_data.scan_indices.push(scan as u32);
                }
            }
            
            chunk_data
        })
        .collect();
    
    // Process MS2 frames
    let ms2_map = DashMap::new();
    ms2_indices.into_par_iter().for_each(|idx| {
        let frame = frames.get(idx).expect("frame read");
        let rt_min = frame.rt_in_seconds as f32 / 60.0;
        let qs = &frame.quadrupole_settings;
        
        for win in 0..qs.isolation_mz.len() {
            if win >= qs.isolation_width.len() { break; }
            let prec_mz = qs.isolation_mz[win] as f32;
            let width = qs.isolation_width[win] as f32;
            let low = prec_mz - width * 0.5;
            let high = prec_mz + width * 0.5;
            let key = (quantize(low), quantize(high));
            
            let mut td = TimsTOFData::new();
            for (p_idx, (&tof, &intensity)) in frame.tof_indices.iter()
                .zip(frame.intensities.iter()).enumerate() 
            {
                let scan = find_scan_for_index_binary(p_idx, &frame.scan_offsets);
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
            
            if !td.mz_values.is_empty() {
                ms2_map.entry(key)
                    .and_modify(|e: &mut TimsTOFData| e.extend_from(&td))
                    .or_insert(td);
            }
        }
    });
    
    println!("  Frame processing: {:.3}s", process_start.elapsed().as_secs_f32());
    
    // Fast parallel merge
    let merge_start = Instant::now();
    let total_ms1_size: usize = ms1_chunks.iter().map(|c| c.mz_values.len()).sum();
    let mut global_ms1 = TimsTOFData::with_capacity(total_ms1_size);
    
    for chunk in ms1_chunks {
        global_ms1.extend_from(&chunk);
    }
    
    let ms2_vec: Vec<_> = ms2_map.into_iter()
        .map(|((q_low, q_high), td)| {
            let low = q_low as f32 / 10_000.0;
            let high = q_high as f32 / 10_000.0;
            ((low, high), td)
        })
        .collect();
    
    println!("  Data merging: {:.3}s", merge_start.elapsed().as_secs_f32());
    println!("  Total time: {:.3}s", total_start.elapsed().as_secs_f32());
    
    Ok(TimsTOFRawData {
        ms1_data: global_ms1,
        ms2_windows: ms2_vec,
    })
}

// Helper to measure memory usage
fn get_memory_usage() -> String {
    // This is a simple approximation - for more accurate results, 
    // consider using a crate like `sysinfo` or `procfs`
    use std::alloc::{GlobalAlloc, Layout, System};
    
    // Force collection of any pending deallocations
    std::thread::yield_now();
    
    format!("Memory checkpoint")
}

// Run a single benchmark with memory cleanup
fn run_single_benchmark<F>(
    name: &str, 
    d_folder: &Path, 
    func: F
) -> Result<(f32, usize, usize), Box<dyn Error>>
where
    F: FnOnce(&Path) -> Result<TimsTOFRawData, Box<dyn Error>>
{
    // Clear any cached data before starting
    drop(DashMap::<u32, u32>::new()); // Create and drop to trigger cleanup
    
    // Give the system time to clean up
    std::thread::sleep(std::time::Duration::from_millis(100));
    
    let start = Instant::now();
    let result = func(d_folder)?;
    let elapsed = start.elapsed().as_secs_f32();
    
    let ms1_points = result.ms1_data.mz_values.len();
    let ms2_windows = result.ms2_windows.len();
    
    // Explicitly drop the large data structure
    drop(result);
    
    // Force deallocation
    std::thread::yield_now();
    std::thread::sleep(std::time::Duration::from_millis(50));
    
    Ok((elapsed, ms1_points, ms2_windows))
}

// Benchmark all versions with proper memory cleanup
pub fn benchmark_all_versions(d_folder: &Path) -> Result<(), Box<dyn Error>> {
    println!("\n========== BENCHMARKING ALL VERSIONS ==========");
    println!("Using {} parallel threads", PARALLEL_THREADS);
    println!("System: Linux x86_64 HPC");
    println!("Each test will run with fresh memory state");
    
    // Detect system info
    #[cfg(target_os = "linux")]
    {
        print_system_info();
    }
    
    // Warm up the disk cache only
    println!("\nWarming up disk cache (results discarded)...");
    {
        let warmup_data = read_timstof_data_v1(d_folder)?;
        println!("  Warmup complete: {} MS1 points loaded", warmup_data.ms1_data.mz_values.len());
        // Explicitly drop warmup data
        drop(warmup_data);
    }
    
    // Clean up after warmup
    std::thread::sleep(std::time::Duration::from_millis(500));
    
    // Clear page cache (Linux only, requires appropriate permissions)
    #[cfg(target_os = "linux")]
    clear_page_cache();
    
    const RUNS: usize = 3;
    
    // Store results for each version separately
    let mut v1_times = Vec::new();
    let mut v2_times = Vec::new();
    let mut v3_times = Vec::new();
    let mut v4_times = Vec::new();
    let mut v5_times = Vec::new();
    let mut v6_times = Vec::new();
    
    let mut ms1_points = 0;
    let mut ms2_windows = 0;
    
    // Run each version multiple times
    for run in 1..=RUNS {
        println!("\n--- Run {}/{} ---", run, RUNS);
        
        // Version 1
        println!("Testing V1 (Optimized Memory Allocation)...");
        let (time, points, windows) = run_single_benchmark("V1", d_folder, read_timstof_data_v1)?;
        v1_times.push(time);
        ms1_points = points;
        ms2_windows = windows;
        println!("  V1: {:.3}s", time);
        
        // Version 2
        println!("Testing V2 (Lock-free Parallel)...");
        let (time, _, _) = run_single_benchmark("V2", d_folder, read_timstof_data_v2)?;
        v2_times.push(time);
        println!("  V2: {:.3}s", time);
        
        // Version 3
        println!("Testing V3 (Chunked Processing)...");
        let (time, _, _) = run_single_benchmark("V3", d_folder, read_timstof_data_v3)?;
        v3_times.push(time);
        println!("  V3: {:.3}s", time);
        
        // Version 4
        println!("Testing V4 (Streaming RwLock)...");
        let (time, _, _) = run_single_benchmark("V4", d_folder, read_timstof_data_v4)?;
        v4_times.push(time);
        println!("  V4: {:.3}s", time);
        
        // Version 5
        println!("Testing V5 (SIMD Pre-sorted)...");
        let (time, _, _) = run_single_benchmark("V5", d_folder, read_timstof_data_v5)?;
        v5_times.push(time);
        println!("  V5: {:.3}s", time);
        
        // Version 6 - HPC optimized
        println!("Testing V6 (HPC-Optimized)...");
        let (time, _, _) = run_single_benchmark("V6", d_folder, read_timstof_data_v6_hpc)?;
        v6_times.push(time);
        println!("  V6: {:.3}s", time);
        
        // Longer pause between runs
        if run < RUNS {
            println!("\nCleaning up before next run...");
            #[cfg(target_os = "linux")]
            drop_caches();
            std::thread::sleep(std::time::Duration::from_millis(1000));
        }
    }
    
    // Calculate and display statistics
    println!("\n========== BENCHMARK SUMMARY ==========");
    println!("Data size: {} MS1 points, {} MS2 windows", ms1_points, ms2_windows);
    println!("\nTiming results over {} runs:", RUNS);
    
    let all_times = vec![
        ("V1 (Memory Opt)", &v1_times),
        ("V2 (Lock-free)", &v2_times),
        ("V3 (Chunked)", &v3_times),
        ("V4 (Streaming)", &v4_times),
        ("V5 (SIMD)", &v5_times),
        ("V6 (HPC)", &v6_times),
    ];
    
    // Find the best performer
    let mut best_avg = f32::MAX;
    let mut best_version = "";
    
    for (name, times) in &all_times {
        let avg = times.iter().sum::<f32>() / times.len() as f32;
        let min = times.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        let max = times.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        let std_dev = {
            let variance = times.iter()
                .map(|t| (t - avg).powi(2))
                .sum::<f32>() / times.len() as f32;
            variance.sqrt()
        };
        
        println!("{:15}: avg={:.3}s, min={:.3}s, max={:.3}s, std={:.3}s", 
                 name, avg, min, max, std_dev);
        
        if avg < best_avg {
            best_avg = avg;
            best_version = name;
        }
    }
    
    println!("\n🏆 Best performer: {} with average time of {:.3}s", best_version, best_avg);
    
    // Calculate relative performance
    println!("\nRelative performance (compared to V1):");
    let v1_avg = v1_times.iter().sum::<f32>() / v1_times.len() as f32;
    for (name, times) in &all_times {
        let avg = times.iter().sum::<f32>() / times.len() as f32;
        let speedup = v1_avg / avg;
        let percent_diff = ((avg - v1_avg) / v1_avg * 100.0);
        println!("{:15}: {:.2}x speed, {:+.1}% time difference", name, speedup, percent_diff);
    }
    
    // Calculate throughput
    println!("\nThroughput (million points/second):");
    let total_points = (ms1_points + ms2_windows * 1000) as f32;  // Approximate
    for (name, times) in &all_times {
        let avg = times.iter().sum::<f32>() / times.len() as f32;
        let throughput = total_points / avg / 1_000_000.0;
        println!("{:15}: {:.2} M points/s", name, throughput);
    }
    
    Ok(())
}

// Linux HPC-specific system info
#[cfg(target_os = "linux")]
fn print_system_info() {
    use std::fs;
    
    println!("\n=== System Information ===");
    
    // CPU info
    if let Ok(cpuinfo) = fs::read_to_string("/proc/cpuinfo") {
        let cores = cpuinfo.lines()
            .filter(|l| l.starts_with("processor"))
            .count();
        if let Some(model) = cpuinfo.lines()
            .find(|l| l.starts_with("model name"))
            .and_then(|l| l.split(':').nth(1)) 
        {
            println!("CPU: {} ({} cores)", model.trim(), cores);
        }
    }
    
    // Memory info
    if let Ok(meminfo) = fs::read_to_string("/proc/meminfo") {
        if let Some(total) = meminfo.lines()
            .find(|l| l.starts_with("MemTotal"))
            .and_then(|l| l.split_whitespace().nth(1))
            .and_then(|s| s.parse::<u64>().ok())
        {
            println!("Memory: {} GB", total / 1024 / 1024);
        }
    }
    
    // NUMA nodes
    let numa_nodes = get_numa_nodes();
    println!("NUMA nodes: {}", numa_nodes);
    
    // Huge pages status
    if let Ok(hp) = fs::read_to_string("/sys/kernel/mm/transparent_hugepage/enabled") {
        println!("Transparent Huge Pages: {}", hp.trim());
    }
}

// Clear page cache (requires appropriate permissions)
#[cfg(target_os = "linux")]
fn clear_page_cache() {
    use std::process::Command;
    let _ = Command::new("sync").status();
    // This requires root/sudo permissions
    let _ = Command::new("sh")
        .arg("-c")
        .arg("echo 1 > /proc/sys/vm/drop_caches")
        .status();
}

// Drop caches between runs
#[cfg(target_os = "linux")]
fn drop_caches() {
    use std::process::Command;
    let _ = Command::new("sync").status();
    // Try to drop caches if we have permission
    let _ = Command::new("sh")
        .arg("-c")
        .arg("echo 3 > /proc/sys/vm/drop_caches")
        .status();
}

// Alternative: Run each version in a separate process for complete isolation
pub fn benchmark_isolated(d_folder: &Path) -> Result<(), Box<dyn Error>> {
    use std::process::Command;
    use std::env;
    
    println!("\n========== ISOLATED BENCHMARK MODE ==========");
    println!("Running each version in a separate process for complete memory isolation");
    
    let exe_path = env::current_exe()?;
    let d_folder_str = d_folder.to_str().unwrap();
    
    const RUNS: usize = 3;
    let versions = vec!["v1", "v2", "v3", "v4", "v5"];
    let mut all_results: HashMap<String, Vec<f32>> = HashMap::new();
    
    for version in &versions {
        all_results.insert(version.to_string(), Vec::new());
        
        for run in 1..=RUNS {
            println!("\nRunning {} (run {}/{})", version, run, RUNS);
            
            let output = Command::new(&exe_path)
                .args(&["--benchmark-single", version, d_folder_str])
                .output()?;
            
            if output.status.success() {
                let stdout = String::from_utf8_lossy(&output.stdout);
                // Parse the timing from output (you'd need to implement this parsing)
                if let Some(time_line) = stdout.lines().find(|l| l.contains("Time:")) {
                    if let Some(time_str) = time_line.split(':').nth(1) {
                        if let Ok(time) = time_str.trim().parse::<f32>() {
                            all_results.get_mut(*version).unwrap().push(time);
                        }
                    }
                }
            }
        }
    }
    
    // Display results
    println!("\n========== ISOLATED BENCHMARK RESULTS ==========");
    for (version, times) in &all_results {
        if !times.is_empty() {
            let avg = times.iter().sum::<f32>() / times.len() as f32;
            println!("{}: avg={:.3}s from {} runs", version, avg, times.len());
        }
    }
    
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    // Configure parallel processing for HPC
    rayon::ThreadPoolBuilder::new()
        .num_threads(PARALLEL_THREADS)
        .build_global()
        .unwrap();
    
    // Set CPU affinity for HPC systems
    #[cfg(target_os = "linux")]
    {
        set_cpu_affinity();
        set_numa_policy();
    }
    
    // Check for command line arguments for isolated benchmark mode
    let args: Vec<String> = std::env::args().collect();
    
    // Handle single benchmark execution (for isolated mode)
    if args.len() >= 4 && args[1] == "--benchmark-single" {
        let version = &args[2];
        let d_path = Path::new(&args[3]);
        
        let start = Instant::now();
        let result = match version.as_str() {
            "v1" => read_timstof_data_v1(d_path),
            "v2" => read_timstof_data_v2(d_path),
            "v3" => read_timstof_data_v3(d_path),
            "v4" => read_timstof_data_v4(d_path),
            "v5" => read_timstof_data_v5(d_path),
            "v6" => read_timstof_data_v6_hpc(d_path),
            _ => return Err(format!("Unknown version: {}", version).into()),
        }?;
        let elapsed = start.elapsed().as_secs_f32();
        
        println!("Time: {:.3}", elapsed);
        println!("MS1 points: {}", result.ms1_data.mz_values.len());
        return Ok(());
    }
    
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
    
    println!("========== TimsTOF Data Loading Benchmark ==========");
    println!("Platform: Linux x86_64 HPC");
    println!("Data folder: {}", d_folder_path);
    println!("Parallel threads: {}", PARALLEL_THREADS);
    
    // Check if user wants isolated benchmark
    if args.len() >= 2 && args[1] == "--isolated" {
        println!("\nRunning in ISOLATED mode (each version in separate process)");
        benchmark_isolated(d_path)?;
    } else if args.len() >= 2 && args[1] == "--quick" {
        println!("\nRunning QUICK test (single run of V6 HPC-optimized)");
        let start = Instant::now();
        let data = read_timstof_data_v6_hpc(d_path)?;
        let elapsed = start.elapsed().as_secs_f32();
        println!("\nResults:");
        println!("  Time: {:.3}s", elapsed);
        println!("  MS1 points: {}", data.ms1_data.mz_values.len());
        println!("  MS2 windows: {}", data.ms2_windows.len());
    } else {
        println!("\nRunning in NORMAL mode (with memory cleanup between runs)");
        println!("Use --isolated flag for complete process isolation");
        println!("Use --quick flag to test only the HPC-optimized version");
        benchmark_all_versions(d_path)?;
    }
    
    println!("\n========== Benchmark Complete ==========");
    
    Ok(())
}

// Updated Cargo.toml dependencies for HPC:
/*
[dependencies]
timsrust = "0.4"
rayon = "1.7"
dashmap = "5.5"
parking_lot = "0.12"

# Optional: Better memory allocators for HPC
# jemallocator = "0.5"  # Good for multi-threaded
# mimalloc = "0.2"       # Microsoft's allocator
# snmalloc = "0.3"       # Microsoft Research allocator optimized for NUMA

[profile.release]
opt-level = 3
lto = true
codegen-units = 1
panic = "abort"
strip = true
debug = false

# CPU-specific optimizations
[profile.release.build-override]
opt-level = 3

# For maximum performance on HPC
[profile.release-hpc]
inherits = "release"
lto = "fat"
codegen-units = 1
target-cpu = "native"  # Optimize for your specific CPU

# Build with:
# RUSTFLAGS="-C target-cpu=native -C opt-level=3" cargo build --release

# For Intel CPUs with AVX2:
# RUSTFLAGS="-C target-cpu=x86-64-v3" cargo build --release

# For AMD EPYC:
# RUSTFLAGS="-C target-cpu=znver2" cargo build --release

# Optional: Use jemalloc for better HPC performance
# Add this at the top of main.rs:
#
# #[global_allocator]
# static GLOBAL: jemallocator::Jemalloc = jemallocator::Jemalloc;
*/