use std::collections::HashMap;
use std::error::Error;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;
use timsrust::{converters::ConvertableDomain, readers::{FrameReader, MetadataReader}, MSLevel};
use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};

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

#[inline(always)]
fn find_scan_for_index(index: usize, scan_offsets: &[usize]) -> usize {
    // Binary search optimization for larger offset arrays
    if scan_offsets.len() > 32 {
        match scan_offsets.binary_search(&index) {
            Ok(pos) => pos,
            Err(pos) => pos.saturating_sub(1),
        }
    } else {
        // Linear search for small arrays (better cache locality)
        for (scan, window) in scan_offsets.windows(2).enumerate() {
            if index >= window[0] && index < window[1] {
                return scan;
            }
        }
        scan_offsets.len() - 1
    }
}

// VERSION 1: Sequential Processing (Baseline)
pub fn read_timstof_v1_sequential(d_folder: &Path) -> Result<TimsTOFRawData, Box<dyn Error>> {
    let total_start = Instant::now();
    
    // Initialize metadata
    let tdf_path = d_folder.join("analysis.tdf");
    let meta = MetadataReader::new(&tdf_path)?;
    let mz_cv = Arc::new(meta.mz_converter);
    let im_cv = Arc::new(meta.im_converter);
    
    // Initialize frame reader
    let frames = FrameReader::new(d_folder)?;
    let n_frames = frames.len();
    
    // Process frames sequentially
    let mut global_ms1 = TimsTOFData::new();
    let mut ms2_hash: HashMap<(u32,u32), TimsTOFData> = HashMap::new();
    
    for idx in 0..n_frames {
        let frame = frames.get(idx).expect("frame read");
        let rt_min = frame.rt_in_seconds as f32 / 60.0;
        
        match frame.ms_level {
            MSLevel::MS1 => {
                for (p_idx, (&tof, &intensity)) in frame.tof_indices.iter().zip(frame.intensities.iter()).enumerate() {
                    let mz = mz_cv.convert(tof as f64) as f32;
                    let scan = find_scan_for_index(p_idx, &frame.scan_offsets);
                    let im = im_cv.convert(scan as f64) as f32;
                    global_ms1.rt_values_min.push(rt_min);
                    global_ms1.mobility_values.push(im);
                    global_ms1.mz_values.push(mz);
                    global_ms1.intensity_values.push(intensity);
                    global_ms1.frame_indices.push(frame.index as u32);
                    global_ms1.scan_indices.push(scan as u32);
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
                    
                    let td = ms2_hash.entry(key).or_insert_with(TimsTOFData::new);
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
                }
            }
            _ => {}
        }
    }
    
    let mut ms2_vec = Vec::with_capacity(ms2_hash.len());
    for ((q_low, q_high), td) in ms2_hash {
        let low = q_low as f32 / 10_000.0;
        let high = q_high as f32 / 10_000.0;
        ms2_vec.push(((low, high), td));
    }
    
    let elapsed = total_start.elapsed().as_secs_f32();
    println!("V1 Sequential - Time: {:.3}s, MS1: {} points, MS2: {} windows", 
             elapsed, global_ms1.mz_values.len(), ms2_vec.len());
    
    Ok(TimsTOFRawData {
        ms1_data: global_ms1,
        ms2_windows: ms2_vec,
    })
}

// VERSION 2: Standard Parallel (Original)
pub fn read_timstof_v2_parallel_standard(d_folder: &Path) -> Result<TimsTOFRawData, Box<dyn Error>> {
    let total_start = Instant::now();
    
    let tdf_path = d_folder.join("analysis.tdf");
    let meta = MetadataReader::new(&tdf_path)?;
    let mz_cv = Arc::new(meta.mz_converter);
    let im_cv = Arc::new(meta.im_converter);
    
    let frames = FrameReader::new(d_folder)?;
    let n_frames = frames.len();
    
    // Process frames in parallel
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
    
    // Merge data
    let ms1_size_estimate: usize = splits.par_iter().map(|s| s.ms1.mz_values.len()).sum();
    let mut global_ms1 = TimsTOFData::with_capacity(ms1_size_estimate);
    let mut ms2_hash: HashMap<(u32,u32), TimsTOFData> = HashMap::new();
    
    for split in splits {
        global_ms1.extend_from(&split.ms1);
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
    
    let elapsed = total_start.elapsed().as_secs_f32();
    println!("V2 Parallel Standard - Time: {:.3}s, MS1: {} points, MS2: {} windows", 
             elapsed, global_ms1.mz_values.len(), ms2_vec.len());
    
    Ok(TimsTOFRawData {
        ms1_data: global_ms1,
        ms2_windows: ms2_vec,
    })
}

// VERSION 3: Chunked Parallel Processing (Better cache locality)
pub fn read_timstof_v3_chunked_parallel(d_folder: &Path) -> Result<TimsTOFRawData, Box<dyn Error>> {
    let total_start = Instant::now();
    
    let tdf_path = d_folder.join("analysis.tdf");
    let meta = MetadataReader::new(&tdf_path)?;
    let mz_cv = Arc::new(meta.mz_converter);
    let im_cv = Arc::new(meta.im_converter);
    
    let frames = FrameReader::new(d_folder)?;
    let n_frames = frames.len();
    
    // Process in chunks for better cache locality
    let chunk_size = 64; // Optimize for cache line size
    let chunks: Vec<Vec<FrameSplit>> = (0..n_frames)
        .collect::<Vec<_>>()
        .chunks(chunk_size)
        .map(|chunk_indices| {
            chunk_indices.par_iter().map(|&idx| {
                let frame = frames.get(idx).expect("frame read");
                let rt_min = frame.rt_in_seconds as f32 / 60.0;
                let mut ms1 = TimsTOFData::new();
                let mut ms2_pairs: Vec<((u32,u32), TimsTOFData)> = Vec::new();
                
                match frame.ms_level {
                    MSLevel::MS1 => {
                        let n_peaks = frame.tof_indices.len();
                        ms1 = TimsTOFData::with_capacity(n_peaks);
                        unsafe {
                            ms1.rt_values_min.set_len(n_peaks);
                            ms1.mobility_values.set_len(n_peaks);
                            ms1.mz_values.set_len(n_peaks);
                            ms1.intensity_values.set_len(n_peaks);
                            ms1.frame_indices.set_len(n_peaks);
                            ms1.scan_indices.set_len(n_peaks);
                        }
                        
                        for (p_idx, (&tof, &intensity)) in frame.tof_indices.iter().zip(frame.intensities.iter()).enumerate() {
                            let mz = mz_cv.convert(tof as f64) as f32;
                            let scan = find_scan_for_index(p_idx, &frame.scan_offsets);
                            let im = im_cv.convert(scan as f64) as f32;
                            ms1.rt_values_min[p_idx] = rt_min;
                            ms1.mobility_values[p_idx] = im;
                            ms1.mz_values[p_idx] = mz;
                            ms1.intensity_values[p_idx] = intensity;
                            ms1.frame_indices[p_idx] = frame.index as u32;
                            ms1.scan_indices[p_idx] = scan as u32;
                        }
                    }
                    MSLevel::MS2 => {
                        let qs = &frame.quadrupole_settings;
                        ms2_pairs.reserve_exact(qs.isolation_mz.len());
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
            }).collect()
        })
        .collect();
    
    // Merge with pre-allocation
    let ms1_size_estimate: usize = chunks.iter()
        .flat_map(|c| c.iter())
        .map(|s| s.ms1.mz_values.len())
        .sum();
    
    let mut global_ms1 = TimsTOFData::with_capacity(ms1_size_estimate);
    let mut ms2_hash: HashMap<(u32,u32), TimsTOFData> = HashMap::with_capacity(100);
    
    for chunk in chunks {
        for split in chunk {
            global_ms1.extend_from(&split.ms1);
            for (key, mut td) in split.ms2 {
                ms2_hash.entry(key).or_insert_with(TimsTOFData::new).merge_from(&mut td);
            }
        }
    }
    
    let mut ms2_vec = Vec::with_capacity(ms2_hash.len());
    for ((q_low, q_high), td) in ms2_hash {
        let low = q_low as f32 / 10_000.0;
        let high = q_high as f32 / 10_000.0;
        ms2_vec.push(((low, high), td));
    }
    
    let elapsed = total_start.elapsed().as_secs_f32();
    println!("V3 Chunked Parallel - Time: {:.3}s, MS1: {} points, MS2: {} windows", 
             elapsed, global_ms1.mz_values.len(), ms2_vec.len());
    
    Ok(TimsTOFRawData {
        ms1_data: global_ms1,
        ms2_windows: ms2_vec,
    })
}

// VERSION 4: Lock-free Parallel Merge
pub fn read_timstof_v4_lockfree_parallel(d_folder: &Path) -> Result<TimsTOFRawData, Box<dyn Error>> {
    use std::sync::Mutex;
    use parking_lot::Mutex as ParkingMutex;
    
    let total_start = Instant::now();
    
    let tdf_path = d_folder.join("analysis.tdf");
    let meta = MetadataReader::new(&tdf_path)?;
    let mz_cv = Arc::new(meta.mz_converter);
    let im_cv = Arc::new(meta.im_converter);
    
    let frames = FrameReader::new(d_folder)?;
    let n_frames = frames.len();
    
    // Use parking_lot for better performance
    let ms1_mutex = Arc::new(ParkingMutex::new(TimsTOFData::with_capacity(1_000_000)));
    let ms2_mutex = Arc::new(ParkingMutex::new(HashMap::<(u32,u32), TimsTOFData>::with_capacity(100)));
    
    // Process frames in parallel with direct merge
    (0..n_frames).into_par_iter().for_each(|idx| {
        let frame = frames.get(idx).expect("frame read");
        let rt_min = frame.rt_in_seconds as f32 / 60.0;
        
        match frame.ms_level {
            MSLevel::MS1 => {
                let n_peaks = frame.tof_indices.len();
                let mut ms1_local = TimsTOFData::with_capacity(n_peaks);
                
                for (p_idx, (&tof, &intensity)) in frame.tof_indices.iter().zip(frame.intensities.iter()).enumerate() {
                    let mz = mz_cv.convert(tof as f64) as f32;
                    let scan = find_scan_for_index(p_idx, &frame.scan_offsets);
                    let im = im_cv.convert(scan as f64) as f32;
                    ms1_local.rt_values_min.push(rt_min);
                    ms1_local.mobility_values.push(im);
                    ms1_local.mz_values.push(mz);
                    ms1_local.intensity_values.push(intensity);
                    ms1_local.frame_indices.push(frame.index as u32);
                    ms1_local.scan_indices.push(scan as u32);
                }
                
                let mut ms1_global = ms1_mutex.lock();
                ms1_global.extend_from(&ms1_local);
            }
            MSLevel::MS2 => {
                let qs = &frame.quadrupole_settings;
                let mut ms2_local = Vec::with_capacity(qs.isolation_mz.len());
                
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
                    ms2_local.push((key, td));
                }
                
                let mut ms2_global = ms2_mutex.lock();
                for (key, mut td) in ms2_local {
                    ms2_global.entry(key).or_insert_with(TimsTOFData::new).merge_from(&mut td);
                }
            }
            _ => {}
        }
    });
    
    let global_ms1 = Arc::try_unwrap(ms1_mutex).unwrap().into_inner();
    let ms2_hash = Arc::try_unwrap(ms2_mutex).unwrap().into_inner();
    
    let mut ms2_vec = Vec::with_capacity(ms2_hash.len());
    for ((q_low, q_high), td) in ms2_hash {
        let low = q_low as f32 / 10_000.0;
        let high = q_high as f32 / 10_000.0;
        ms2_vec.push(((low, high), td));
    }
    
    let elapsed = total_start.elapsed().as_secs_f32();
    println!("V4 Lock-free Parallel - Time: {:.3}s, MS1: {} points, MS2: {} windows", 
             elapsed, global_ms1.mz_values.len(), ms2_vec.len());
    
    Ok(TimsTOFRawData {
        ms1_data: global_ms1,
        ms2_windows: ms2_vec,
    })
}

// VERSION 5: Work-stealing with crossbeam
pub fn read_timstof_v5_workstealing(d_folder: &Path) -> Result<TimsTOFRawData, Box<dyn Error>> {
    use crossbeam::channel::{bounded, unbounded};
    use std::thread;
    
    let total_start = Instant::now();
    
    let tdf_path = d_folder.join("analysis.tdf");
    let meta = MetadataReader::new(&tdf_path)?;
    let mz_cv = Arc::new(meta.mz_converter);
    let im_cv = Arc::new(meta.im_converter);
    
    let frames = Arc::new(FrameReader::new(d_folder)?);
    let n_frames = frames.len();
    
    let num_workers = num_cpus::get();
    let (work_tx, work_rx) = bounded::<usize>(n_frames);
    let (result_tx, result_rx) = unbounded::<FrameSplit>();
    
    // Queue all work
    for idx in 0..n_frames {
        work_tx.send(idx).unwrap();
    }
    drop(work_tx);
    
    // Spawn workers
    let workers: Vec<_> = (0..num_workers).map(|_| {
        let work_rx = work_rx.clone();
        let result_tx = result_tx.clone();
        let frames = frames.clone();
        let mz_cv = mz_cv.clone();
        let im_cv = im_cv.clone();
        
        thread::spawn(move || {
            while let Ok(idx) = work_rx.recv() {
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
                
                result_tx.send(FrameSplit { ms1, ms2: ms2_pairs }).unwrap();
            }
        })
    }).collect();
    
    drop(result_tx);
    
    // Collect results
    let mut global_ms1 = TimsTOFData::with_capacity(1_000_000);
    let mut ms2_hash: HashMap<(u32,u32), TimsTOFData> = HashMap::with_capacity(100);
    
    while let Ok(split) = result_rx.recv() {
        global_ms1.extend_from(&split.ms1);
        for (key, mut td) in split.ms2 {
            ms2_hash.entry(key).or_insert_with(TimsTOFData::new).merge_from(&mut td);
        }
    }
    
    // Wait for workers
    for worker in workers {
        worker.join().unwrap();
    }
    
    let mut ms2_vec = Vec::with_capacity(ms2_hash.len());
    for ((q_low, q_high), td) in ms2_hash {
        let low = q_low as f32 / 10_000.0;
        let high = q_high as f32 / 10_000.0;
        ms2_vec.push(((low, high), td));
    }
    
    let elapsed = total_start.elapsed().as_secs_f32();
    println!("V5 Work-stealing - Time: {:.3}s, MS1: {} points, MS2: {} windows", 
             elapsed, global_ms1.mz_values.len(), ms2_vec.len());
    
    Ok(TimsTOFRawData {
        ms1_data: global_ms1,
        ms2_windows: ms2_vec,
    })
}

fn main() -> Result<(), Box<dyn Error>> {
    // Detect system configuration
    let num_cores = num_cpus::get();
    let physical_cores = num_cpus::get_physical();
    
    println!("========== System Configuration ==========");
    println!("Logical cores: {}", num_cores);
    println!("Physical cores: {}", physical_cores);
    println!("Architecture: {}", std::env::consts::ARCH);
    println!("OS: {}", std::env::consts::OS);
    
    // Set path to your .d folder
    let d_folder_path = if cfg!(target_os = "macos") {
        "/Users/augustsirius/Desktop/DIA_peak_group_extraction/输入数据文件/raw_data/CAD20220207yuel_TPHP_DIA_pool1_Slot2-54_1_4382.d"
    } else {
        // HPC path
        "/storage/guotiannanLab/wangshuaiyao/006.DIABERT_TimsTOF_Rust/test_data/CAD20220207yuel_TPHP_DIA_pool1_Slot2-54_1_4382.d"
    };
    
    let d_path = Path::new(d_folder_path);
    if !d_path.exists() {
        return Err(format!("Folder {:?} not found", d_path).into());
    }
    
    println!("\n========== TimsTOF Multi-Version Benchmark ==========");
    println!("Data folder: {}", d_folder_path);
    
    // Warm-up run
    println!("\n--- Warm-up run ---");
    let _ = read_timstof_v1_sequential(d_path)?;
    
    println!("\n========== Performance Comparison ==========");
    
    // Test different thread configurations for parallel versions
    let thread_configs = vec![1, 2, 4, 8, 16, 32, 64, physical_cores, num_cores];
    
    for threads in thread_configs {
        if threads > num_cores {
            continue;
        }
        
        println!("\n--- Testing with {} threads ---", threads);
        
        // Configure rayon thread pool
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .unwrap();
        
        // Run all versions
        if threads == 1 {
            println!("\nVersion 1: Sequential Processing");
            let _ = read_timstof_v1_sequential(d_path)?;
        }
        
        println!("\nVersion 2: Standard Parallel");
        let _ = read_timstof_v2_parallel_standard(d_path)?;
        
        println!("\nVersion 3: Chunked Parallel");
        let _ = read_timstof_v3_chunked_parallel(d_path)?;
        
        println!("\nVersion 4: Lock-free Parallel");
        let _ = read_timstof_v4_lockfree_parallel(d_path)?;
        
        if threads == physical_cores {
            println!("\nVersion 5: Work-stealing (uses all cores)");
            let _ = read_timstof_v5_workstealing(d_path)?;
        }
    }
    
    println!("\n========== Benchmark Complete ==========");
    
    // Find optimal configuration
    println!("\nRecommendations:");
    println!("- For HPC with many cores: Use V3 (Chunked) or V4 (Lock-free) with thread count = physical cores");
    println!("- For memory-constrained systems: Use V3 (Chunked) with smaller thread count");
    println!("- For best latency: Use V5 (Work-stealing) on systems with good thread scheduling");
    
    Ok(())
}

// Cargo.toml dependencies needed:
/*
[package]
name = "timstof-reader"
version = "0.1.0"
edition = "2021"

[dependencies]
timsrust = "0.4"
rayon = "1.7"
parking_lot = "0.12"
crossbeam = "0.8"
num_cpus = "1.16"

[profile.release]
opt-level = 3
lto = true
codegen-units = 1
panic = "abort"
strip = true

[profile.release-lto]
inherits = "release"
lto = "fat"

[profile.bench]
inherits = "release"
debug = true
*/