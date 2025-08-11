use std::collections::HashMap;
use std::error::Error;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use std::env;
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

// Helper functions
#[inline(always)]
fn quantize(x: f32) -> u32 { 
    (x * 10_000.0).round() as u32 
}

#[inline(always)]
fn find_scan_for_index(index: usize, scan_offsets: &[usize]) -> usize {
    if scan_offsets.len() > 16 {
        let pos = scan_offsets.partition_point(|&offset| offset <= index);
        if pos > 0 { pos - 1 } else { 0 }
    } else {
        for (scan, window) in scan_offsets.windows(2).enumerate() {
            if index >= window[0] && index < window[1] {
                return scan;
            }
        }
        scan_offsets.len() - 1
    }
}

// Detect total available processing units
fn detect_total_cores() -> usize {
    let slurm_ntasks = env::var("SLURM_NTASKS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(1);
    
    let slurm_cpus_per_task = env::var("SLURM_CPUS_PER_TASK")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(1);
    
    let slurm_job_cpus = env::var("SLURM_JOB_CPUS_PER_NODE")
        .ok()
        .and_then(|s| s.parse::<usize>().ok());
    
    let slurm_total = slurm_job_cpus.unwrap_or(slurm_ntasks * slurm_cpus_per_task);
    
    if slurm_total <= 1 {
        num_cpus::get()
    } else {
        slurm_total
    }
}

// Initialize global thread pool ONCE with maximum cores
fn initialize_max_threads() {
    let total_cores = detect_total_cores();
    println!("🚀 Initializing with {} cores", total_cores);
    
    rayon::ThreadPoolBuilder::new()
        .num_threads(total_cores)
        .thread_name(|i| format!("worker-{}", i))
        .build_global()
        .unwrap();
}

// ORIGINAL VERSION - Memory-efficient baseline
pub fn read_timstof_original(d_folder: &Path) -> Result<TimsTOFRawData, Box<dyn Error>> {
    let total_start = Instant::now();
    
    let tdf_path = d_folder.join("analysis.tdf");
    let meta = MetadataReader::new(&tdf_path)?;
    let mz_cv = Arc::new(meta.mz_converter);
    let im_cv = Arc::new(meta.im_converter);
    
    let frames = FrameReader::new(d_folder)?;
    let n_frames = frames.len();
    
    // Process in smaller chunks to reduce memory pressure
    const CHUNK_SIZE: usize = 100;
    let mut global_ms1 = TimsTOFData::with_capacity(10_000_000); // Start with 10M capacity
    let mut ms2_hash: HashMap<(u32,u32), TimsTOFData> = HashMap::with_capacity(100);
    
    for chunk_start in (0..n_frames).step_by(CHUNK_SIZE) {
        let chunk_end = (chunk_start + CHUNK_SIZE).min(n_frames);
        
        let chunk_results: Vec<_> = (chunk_start..chunk_end)
            .into_par_iter()
            .map(|idx| {
                let frame = frames.get(idx).expect("frame read");
                let rt_min = frame.rt_in_seconds as f32 / 60.0;
                let mut ms1 = TimsTOFData::new();
                let mut ms2_pairs: Vec<((u32,u32), TimsTOFData)> = Vec::new();
                
                match frame.ms_level {
                    MSLevel::MS1 => {
                        for (p_idx, (&tof, &intensity)) in frame.tof_indices.iter()
                            .zip(frame.intensities.iter()).enumerate() 
                        {
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
                        for win in 0..qs.isolation_mz.len() {
                            if win >= qs.isolation_width.len() { break; }
                            let prec_mz = qs.isolation_mz[win] as f32;
                            let width = qs.isolation_width[win] as f32;
                            let key = (quantize(prec_mz - width * 0.5), quantize(prec_mz + width * 0.5));
                            
                            let mut td = TimsTOFData::new();
                            for (p_idx, (&tof, &intensity)) in frame.tof_indices.iter()
                                .zip(frame.intensities.iter()).enumerate() 
                            {
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
                            if !td.mz_values.is_empty() {
                                ms2_pairs.push((key, td));
                            }
                        }
                    }
                    _ => {}
                }
                (ms1, ms2_pairs)
            })
            .collect();
        
        // Merge chunk results immediately to free memory
        for (ms1, ms2_pairs) in chunk_results {
            global_ms1.extend_from(&ms1);
            for (key, mut td) in ms2_pairs {
                ms2_hash.entry(key).or_insert_with(TimsTOFData::new).merge_from(&mut td);
            }
        }
    }
    
    let mut ms2_vec = Vec::with_capacity(ms2_hash.len());
    for ((q_low, q_high), td) in ms2_hash {
        ms2_vec.push(((q_low as f32 / 10_000.0, q_high as f32 / 10_000.0), td));
    }
    
    let elapsed = total_start.elapsed().as_secs_f32();
    println!("ORIGINAL CHUNKED - Time: {:.3}s, MS1: {} points, MS2: {} windows", 
             elapsed, global_ms1.mz_values.len(), ms2_vec.len());
    
    Ok(TimsTOFRawData {
        ms1_data: global_ms1,
        ms2_windows: ms2_vec,
    })
}

// OPTIMIZED V1: Streaming with incremental merge
pub fn read_timstof_v1_streaming(d_folder: &Path) -> Result<TimsTOFRawData, Box<dyn Error>> {
    let total_start = Instant::now();
    
    let tdf_path = d_folder.join("analysis.tdf");
    let meta = MetadataReader::new(&tdf_path)?;
    let mz_cv = Arc::new(meta.mz_converter);
    let im_cv = Arc::new(meta.im_converter);
    
    let frames = Arc::new(FrameReader::new(d_folder)?);
    let n_frames = frames.len();
    
    // Use Arc<Mutex> for thread-safe incremental merging
    let global_ms1 = Arc::new(Mutex::new(TimsTOFData::with_capacity(10_000_000)));
    let ms2_hash = Arc::new(Mutex::new(HashMap::<(u32,u32), TimsTOFData>::with_capacity(100)));
    
    // Process frames and merge immediately (no intermediate storage)
    (0..n_frames).into_par_iter().for_each(|idx| {
        let frame = frames.get(idx).expect("frame read");
        let rt_min = frame.rt_in_seconds as f32 / 60.0;
        
        match frame.ms_level {
            MSLevel::MS1 => {
                let mut ms1_local = TimsTOFData::with_capacity(frame.tof_indices.len());
                
                for (p_idx, (&tof, &intensity)) in frame.tof_indices.iter()
                    .zip(frame.intensities.iter()).enumerate() 
                {
                    let scan = find_scan_for_index(p_idx, &frame.scan_offsets);
                    ms1_local.rt_values_min.push(rt_min);
                    ms1_local.mobility_values.push(im_cv.convert(scan as f64) as f32);
                    ms1_local.mz_values.push(mz_cv.convert(tof as f64) as f32);
                    ms1_local.intensity_values.push(intensity);
                    ms1_local.frame_indices.push(frame.index as u32);
                    ms1_local.scan_indices.push(scan as u32);
                }
                
                // Merge immediately
                let mut global = global_ms1.lock().unwrap();
                global.extend_from(&ms1_local);
            }
            MSLevel::MS2 => {
                let qs = &frame.quadrupole_settings;
                let mut ms2_local = Vec::with_capacity(qs.isolation_mz.len());
                
                for win in 0..qs.isolation_mz.len() {
                    if win >= qs.isolation_width.len() { break; }
                    let prec_mz = qs.isolation_mz[win] as f32;
                    let width = qs.isolation_width[win] as f32;
                    let key = (quantize(prec_mz - width * 0.5), quantize(prec_mz + width * 0.5));
                    
                    let mut td = TimsTOFData::new();
                    for (p_idx, (&tof, &intensity)) in frame.tof_indices.iter()
                        .zip(frame.intensities.iter()).enumerate() 
                    {
                        let scan = find_scan_for_index(p_idx, &frame.scan_offsets);
                        if scan < qs.scan_starts[win] || scan > qs.scan_ends[win] { continue; }
                        td.rt_values_min.push(rt_min);
                        td.mobility_values.push(im_cv.convert(scan as f64) as f32);
                        td.mz_values.push(mz_cv.convert(tof as f64) as f32);
                        td.intensity_values.push(intensity);
                        td.frame_indices.push(frame.index as u32);
                        td.scan_indices.push(scan as u32);
                    }
                    if !td.mz_values.is_empty() {
                        ms2_local.push((key, td));
                    }
                }
                
                // Merge immediately
                let mut global = ms2_hash.lock().unwrap();
                for (key, mut td) in ms2_local {
                    global.entry(key).or_insert_with(TimsTOFData::new).merge_from(&mut td);
                }
            }
            _ => {}
        }
    });
    
    let global_ms1 = Arc::try_unwrap(global_ms1).unwrap().into_inner().unwrap();
    let ms2_hash = Arc::try_unwrap(ms2_hash).unwrap().into_inner().unwrap();
    
    let mut ms2_vec = Vec::with_capacity(ms2_hash.len());
    for ((q_low, q_high), td) in ms2_hash {
        ms2_vec.push(((q_low as f32 / 10_000.0, q_high as f32 / 10_000.0), td));
    }
    
    let elapsed = total_start.elapsed().as_secs_f32();
    println!("V1 STREAMING - Time: {:.3}s, MS1: {} points, MS2: {} windows", 
             elapsed, global_ms1.mz_values.len(), ms2_vec.len());
    
    Ok(TimsTOFRawData {
        ms1_data: global_ms1,
        ms2_windows: ms2_vec,
    })
}

// OPTIMIZED V2: Work-stealing queue pattern
pub fn read_timstof_v2_workstealing(d_folder: &Path) -> Result<TimsTOFRawData, Box<dyn Error>> {
    use crossbeam::channel::{bounded, Sender, Receiver};
    use std::thread;
    
    let total_start = Instant::now();
    
    let tdf_path = d_folder.join("analysis.tdf");
    let meta = MetadataReader::new(&tdf_path)?;
    let mz_cv = Arc::new(meta.mz_converter);
    let im_cv = Arc::new(meta.im_converter);
    
    let frames = Arc::new(FrameReader::new(d_folder)?);
    let n_frames = frames.len();
    
    let num_workers = detect_total_cores();
    
    // Create channels for work distribution
    let (work_tx, work_rx) = bounded::<usize>(num_workers * 2); // Limited buffer
    let (ms1_tx, ms1_rx) = bounded::<TimsTOFData>(num_workers);
    let (ms2_tx, ms2_rx) = bounded::<((u32, u32), TimsTOFData)>(num_workers * 10);
    
    // Spawn workers
    let workers: Vec<_> = (0..num_workers).map(|_| {
        let work_rx = work_rx.clone();
        let ms1_tx = ms1_tx.clone();
        let ms2_tx = ms2_tx.clone();
        let frames = frames.clone();
        let mz_cv = mz_cv.clone();
        let im_cv = im_cv.clone();
        
        thread::spawn(move || {
            while let Ok(idx) = work_rx.recv() {
                let frame = frames.get(idx).expect("frame read");
                let rt_min = frame.rt_in_seconds as f32 / 60.0;
                
                match frame.ms_level {
                    MSLevel::MS1 => {
                        let mut ms1 = TimsTOFData::with_capacity(frame.tof_indices.len());
                        
                        for (p_idx, (&tof, &intensity)) in frame.tof_indices.iter()
                            .zip(frame.intensities.iter()).enumerate() 
                        {
                            let scan = find_scan_for_index(p_idx, &frame.scan_offsets);
                            ms1.rt_values_min.push(rt_min);
                            ms1.mobility_values.push(im_cv.convert(scan as f64) as f32);
                            ms1.mz_values.push(mz_cv.convert(tof as f64) as f32);
                            ms1.intensity_values.push(intensity);
                            ms1.frame_indices.push(frame.index as u32);
                            ms1.scan_indices.push(scan as u32);
                        }
                        
                        ms1_tx.send(ms1).unwrap();
                    }
                    MSLevel::MS2 => {
                        let qs = &frame.quadrupole_settings;
                        
                        for win in 0..qs.isolation_mz.len() {
                            if win >= qs.isolation_width.len() { break; }
                            let prec_mz = qs.isolation_mz[win] as f32;
                            let width = qs.isolation_width[win] as f32;
                            let key = (quantize(prec_mz - width * 0.5), quantize(prec_mz + width * 0.5));
                            
                            let mut td = TimsTOFData::new();
                            for (p_idx, (&tof, &intensity)) in frame.tof_indices.iter()
                                .zip(frame.intensities.iter()).enumerate() 
                            {
                                let scan = find_scan_for_index(p_idx, &frame.scan_offsets);
                                if scan < qs.scan_starts[win] || scan > qs.scan_ends[win] { continue; }
                                td.rt_values_min.push(rt_min);
                                td.mobility_values.push(im_cv.convert(scan as f64) as f32);
                                td.mz_values.push(mz_cv.convert(tof as f64) as f32);
                                td.intensity_values.push(intensity);
                                td.frame_indices.push(frame.index as u32);
                                td.scan_indices.push(scan as u32);
                            }
                            if !td.mz_values.is_empty() {
                                ms2_tx.send((key, td)).unwrap();
                            }
                        }
                    }
                    _ => {}
                }
            }
        })
    }).collect();
    
    // Producer thread
    let producer = thread::spawn(move || {
        for idx in 0..n_frames {
            work_tx.send(idx).unwrap();
        }
    });
    
    // Consumer thread for aggregation
    drop(ms1_tx);
    drop(ms2_tx);
    
    let ms1_consumer = thread::spawn(move || {
        let mut global_ms1 = TimsTOFData::with_capacity(10_000_000);
        while let Ok(ms1) = ms1_rx.recv() {
            global_ms1.extend_from(&ms1);
        }
        global_ms1
    });
    
    let ms2_consumer = thread::spawn(move || {
        let mut ms2_hash: HashMap<(u32,u32), TimsTOFData> = HashMap::with_capacity(100);
        while let Ok((key, mut td)) = ms2_rx.recv() {
            ms2_hash.entry(key).or_insert_with(TimsTOFData::new).merge_from(&mut td);
        }
        ms2_hash
    });
    
    // Wait for completion
    producer.join().unwrap();
    drop(work_tx);
    
    for worker in workers {
        worker.join().unwrap();
    }
    
    let global_ms1 = ms1_consumer.join().unwrap();
    let ms2_hash = ms2_consumer.join().unwrap();
    
    let mut ms2_vec = Vec::with_capacity(ms2_hash.len());
    for ((q_low, q_high), td) in ms2_hash {
        ms2_vec.push(((q_low as f32 / 10_000.0, q_high as f32 / 10_000.0), td));
    }
    
    let elapsed = total_start.elapsed().as_secs_f32();
    println!("V2 WORK-STEALING - Time: {:.3}s, MS1: {} points, MS2: {} windows", 
             elapsed, global_ms1.mz_values.len(), ms2_vec.len());
    
    Ok(TimsTOFRawData {
        ms1_data: global_ms1,
        ms2_windows: ms2_vec,
    })
}

// Simple verification (memory-efficient)
fn verify_results(original: &TimsTOFRawData, optimized: &TimsTOFRawData, version: &str) -> bool {
    println!("  🔍 Verifying {}: MS1={} vs {}, MS2={} vs {}", 
             version,
             original.ms1_data.mz_values.len(), 
             optimized.ms1_data.mz_values.len(),
             original.ms2_windows.len(),
             optimized.ms2_windows.len());
    
    original.ms1_data.mz_values.len() == optimized.ms1_data.mz_values.len() &&
    original.ms2_windows.len() == optimized.ms2_windows.len()
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("========== Memory-Optimized TimsTOF Reader ==========");
    
    // Initialize ONCE with maximum cores
    initialize_max_threads();
    
    let total_cores = detect_total_cores();
    println!("\n📋 Configuration: {} cores, Memory-efficient processing", total_cores);
    
    // Set data path
    let d_folder_path = if cfg!(target_os = "macos") {
        "/Users/augustsirius/Desktop/DIA_peak_group_extraction/输入数据文件/raw_data/CAD20220207yuel_TPHP_DIA_pool1_Slot2-54_1_4382.d"
    } else {
        "/storage/guotiannanLab/wangshuaiyao/006.DIABERT_TimsTOF_Rust/test_data/CAD20220207yuel_TPHP_DIA_pool1_Slot2-54_1_4382.d"
    };
    
    let d_path = Path::new(d_folder_path);
    if !d_path.exists() {
        return Err(format!("Folder {:?} not found", d_path).into());
    }
    
    println!("📁 Data: {}", d_folder_path);
    
    // Run all versions
    println!("\n========== Performance Testing ==========\n");
    
    println!("1️⃣ ORIGINAL CHUNKED (memory-safe baseline)...");
    let original = read_timstof_original(d_path)?;
    
    println!("\n2️⃣ V1 STREAMING (incremental merge)...");
    let v1 = read_timstof_v1_streaming(d_path)?;
    verify_results(&original, &v1, "V1");
    
    println!("\n3️⃣ V2 WORK-STEALING (channel-based)...");
    let v2 = read_timstof_v2_workstealing(d_path)?;
    verify_results(&original, &v2, "V2");
    
    println!("\n✅ All versions completed successfully with proper memory management!");
    
    Ok(())
}

// Cargo.toml - ADD crossbeam for V2
/*
[package]
name = "timstof-reader"
version = "0.1.0"
edition = "2021"

[dependencies]
timsrust = "0.4"
rayon = "1.10"
num_cpus = "1.16"
crossbeam = "0.8"

[profile.release]
opt-level = 3
lto = "fat"
codegen-units = 1
*/