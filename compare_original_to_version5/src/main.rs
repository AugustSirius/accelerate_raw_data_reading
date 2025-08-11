use std::collections::HashMap;
use std::error::Error;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;
use timsrust::{converters::ConvertableDomain, readers::{FrameReader, MetadataReader}, MSLevel};
use rayon::prelude::*;
use dashmap::DashMap;
use crossbeam_channel::bounded;
use parking_lot::Mutex;

// ============= 共享数据结构 =============
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

// ============= 原始版本实现 =============
mod original_version {
    use super::*;
    
    struct FrameSplit {
        pub ms1: TimsTOFData,
        pub ms2: Vec<((u32, u32), TimsTOFData)>,
    }
    
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
    
    pub fn read_timstof_data_original(d_folder: &Path) -> Result<TimsTOFRawData, Box<dyn Error>> {
        let total_start = Instant::now();
        
        println!("[ORIGINAL] Initializing metadata readers...");
        let tdf_path = d_folder.join("analysis.tdf");
        let meta = MetadataReader::new(&tdf_path)?;
        let mz_cv = Arc::new(meta.mz_converter);
        let im_cv = Arc::new(meta.im_converter);
        
        println!("[ORIGINAL] Initializing frame reader...");
        let frames = FrameReader::new(d_folder)?;
        let n_frames = frames.len();
        println!("[ORIGINAL] Total frames to process: {}", n_frames);
        
        println!("[ORIGINAL] Processing frames in parallel...");
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
        
        println!("[ORIGINAL] Merging data...");
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
        
        println!("[ORIGINAL] MS1 data points: {}", global_ms1.mz_values.len());
        println!("[ORIGINAL] MS2 windows: {}", ms2_vec.len());
        let total_ms2_points: usize = ms2_vec.iter().map(|(_, td)| td.mz_values.len()).sum();
        println!("[ORIGINAL] MS2 data points: {}", total_ms2_points);
        println!("[ORIGINAL] Total processing time: {:.3}s", total_start.elapsed().as_secs_f32());
        
        Ok(TimsTOFRawData {
            ms1_data: global_ms1,
            ms2_windows: ms2_vec,
        })
    }
}

// ============= 优化版本V5（修复后）实现 =============
mod v5_fixed {
    use super::*;
    
    #[derive(Clone)]  // Add this line
    enum ProcessedFrame {
        MS1(usize, TimsTOFData),
        MS2(usize, Vec<((u32, u32), TimsTOFData)>),
    }
    
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
    
    pub fn read_timstof_data_v5_fixed(d_folder: &Path) -> Result<TimsTOFRawData, Box<dyn Error>> {
        let total_start = Instant::now();
        
        println!("[V5_FIXED] Initializing metadata readers...");
        let tdf_path = d_folder.join("analysis.tdf");
        let meta = MetadataReader::new(&tdf_path)?;
        let mz_cv = Arc::new(meta.mz_converter);
        let im_cv = Arc::new(meta.im_converter);
        
        println!("[V5_FIXED] Initializing frame reader...");
        let frames = Arc::new(FrameReader::new(d_folder)?);
        let n_frames = frames.len();
        println!("[V5_FIXED] Total frames to process: {}", n_frames);
        
        println!("[V5_FIXED] Processing frames in parallel with channel...");
        
        let (sender, receiver) = bounded(2000);
        let processed_count = Arc::new(AtomicUsize::new(0));
        let ms1_accumulator = Arc::new(Mutex::new(Vec::with_capacity(n_frames)));
        let ms2_map = Arc::new(DashMap::with_capacity(100));
        
        let ms1_acc_clone = Arc::clone(&ms1_accumulator);
        let ms2_map_clone = Arc::clone(&ms2_map);
        let processed_clone = Arc::clone(&processed_count);
        
        let aggregator_handle = std::thread::spawn(move || {
            let mut frame_buffer: Vec<Option<ProcessedFrame>> = vec![None; n_frames];
            let mut next_frame = 0usize;
            
            while let Ok(frame_data) = receiver.recv() {
                match frame_data {
                    ProcessedFrame::MS1(idx, data) => {
                        frame_buffer[idx] = Some(ProcessedFrame::MS1(idx, data));
                    }
                    ProcessedFrame::MS2(idx, pairs) => {
                        frame_buffer[idx] = Some(ProcessedFrame::MS2(idx, pairs));
                    }
                }
                
                while next_frame < n_frames {
                    if let Some(frame) = frame_buffer[next_frame].take() {
                        match frame {
                            ProcessedFrame::MS1(_, data) => {
                                if !data.mz_values.is_empty() {
                                    ms1_acc_clone.lock().push(data);
                                }
                            }
                            ProcessedFrame::MS2(_, pairs) => {
                                for (key, mut data) in pairs {
                                    if !data.mz_values.is_empty() {
                                        ms2_map_clone.entry(key)
                                            .or_insert_with(|| Arc::new(Mutex::new(TimsTOFData::new())))
                                            .lock()
                                            .merge_from(&mut data);
                                    }
                                }
                            }
                        }
                        next_frame += 1;
                        processed_clone.fetch_add(1, Ordering::Relaxed);
                    } else {
                        break;
                    }
                }
            }
        });
        
        (0..n_frames).into_par_iter().for_each(|idx| {
            let frame = match frames.get(idx) {
                Ok(f) => f,
                Err(_) => return,
            };
            
            let rt_min = frame.rt_in_seconds as f32 / 60.0;
            
            match frame.ms_level {
                MSLevel::MS1 => {
                    let n_peaks = frame.tof_indices.len();
                    let mut ms1 = TimsTOFData::with_capacity(n_peaks);
                    
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
                    
                    if !ms1.mz_values.is_empty() {
                        let _ = sender.send(ProcessedFrame::MS1(idx, ms1));
                    }
                }
                MSLevel::MS2 => {
                    let qs = &frame.quadrupole_settings;
                    let mut ms2_pairs = Vec::with_capacity(qs.isolation_mz.len());
                    
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
                        
                        if !td.mz_values.is_empty() {
                            ms2_pairs.push((key, td));
                        }
                    }
                    
                    if !ms2_pairs.is_empty() {
                        let _ = sender.send(ProcessedFrame::MS2(idx, ms2_pairs));
                    }
                }
                _ => {}
            }
        });
        
        drop(sender);
        aggregator_handle.join().unwrap();
        
        println!("[V5_FIXED] Finalizing data structures...");
        
        let ms1_chunks = ms1_accumulator.lock();
        let mut global_ms1 = TimsTOFData::with_capacity(
            ms1_chunks.iter().map(|c| c.mz_values.len()).sum()
        );
        
        for chunk in ms1_chunks.iter() {
            global_ms1.rt_values_min.extend(&chunk.rt_values_min);
            global_ms1.mobility_values.extend(&chunk.mobility_values);
            global_ms1.mz_values.extend(&chunk.mz_values);
            global_ms1.intensity_values.extend(&chunk.intensity_values);
            global_ms1.frame_indices.extend(&chunk.frame_indices);
            global_ms1.scan_indices.extend(&chunk.scan_indices);
        }
        
        let mut ms2_vec = Vec::with_capacity(ms2_map.len());
        ms2_map.iter().for_each(|entry| {
            let (q_low, q_high) = *entry.key();
            let low = q_low as f32 / 10_000.0;
            let high = q_high as f32 / 10_000.0;
            let data = entry.value().lock().clone();
            ms2_vec.push(((low, high), data));
        });
        
        println!("[V5_FIXED] MS1 data points: {}", global_ms1.mz_values.len());
        println!("[V5_FIXED] MS2 windows: {}", ms2_vec.len());
        let total_ms2_points: usize = ms2_vec.iter().map(|(_, td)| td.mz_values.len()).sum();
        println!("[V5_FIXED] MS2 data points: {}", total_ms2_points);
        println!("[V5_FIXED] Total processing time: {:.3}s", total_start.elapsed().as_secs_f32());
        
        Ok(TimsTOFRawData {
            ms1_data: global_ms1,
            ms2_windows: ms2_vec,
        })
    }
}

// ============= 比较工具 =============
fn compare_tims_data(data1: &TimsTOFData, data2: &TimsTOFData, name: &str) -> bool {
    println!("\n  Comparing {} data...", name);
    let mut all_match = true;
    
    // 比较数据长度
    if data1.mz_values.len() != data2.mz_values.len() {
        println!("    ❌ Length mismatch: {} vs {}", 
                data1.mz_values.len(), data2.mz_values.len());
        all_match = false;
    } else {
        println!("    ✓ Length match: {} points", data1.mz_values.len());
    }
    
    if !all_match {
        return false;
    }
    
    // 创建索引映射以处理顺序差异
    let mut indices1: Vec<usize> = (0..data1.mz_values.len()).collect();
    let mut indices2: Vec<usize> = (0..data2.mz_values.len()).collect();
    
    // 按frame_index和scan_index排序
    indices1.sort_by_key(|&i| (data1.frame_indices[i], data1.scan_indices[i], 
                               (data1.mz_values[i] * 1e6) as i64));
    indices2.sort_by_key(|&i| (data2.frame_indices[i], data2.scan_indices[i],
                               (data2.mz_values[i] * 1e6) as i64));
    
    // 比较排序后的数据
    let mut mismatch_count = 0;
    const MAX_MISMATCHES: usize = 10;
    
    for i in 0..indices1.len() {
        let idx1 = indices1[i];
        let idx2 = indices2[i];
        
        let rt_match = (data1.rt_values_min[idx1] - data2.rt_values_min[idx2]).abs() < 1e-6;
        let mobility_match = (data1.mobility_values[idx1] - data2.mobility_values[idx2]).abs() < 1e-6;
        let mz_match = (data1.mz_values[idx1] - data2.mz_values[idx2]).abs() < 1e-6;
        let intensity_match = data1.intensity_values[idx1] == data2.intensity_values[idx2];
        let frame_match = data1.frame_indices[idx1] == data2.frame_indices[idx2];
        let scan_match = data1.scan_indices[idx1] == data2.scan_indices[idx2];
        
        if !rt_match || !mobility_match || !mz_match || !intensity_match || !frame_match || !scan_match {
            if mismatch_count < MAX_MISMATCHES {
                println!("    ❌ Mismatch at sorted index {}:", i);
                if !rt_match {
                    println!("      RT: {:.6} vs {:.6}", 
                            data1.rt_values_min[idx1], data2.rt_values_min[idx2]);
                }
                if !mz_match {
                    println!("      MZ: {:.6} vs {:.6}", 
                            data1.mz_values[idx1], data2.mz_values[idx2]);
                }
                if !intensity_match {
                    println!("      Intensity: {} vs {}", 
                            data1.intensity_values[idx1], data2.intensity_values[idx2]);
                }
                if !frame_match {
                    println!("      Frame: {} vs {}", 
                            data1.frame_indices[idx1], data2.frame_indices[idx2]);
                }
                if !scan_match {
                    println!("      Scan: {} vs {}", 
                            data1.scan_indices[idx1], data2.scan_indices[idx2]);
                }
            }
            mismatch_count += 1;
            all_match = false;
        }
    }
    
    if mismatch_count > 0 {
        println!("    Total mismatches: {}", mismatch_count);
    } else {
        println!("    ✓ All data points match!");
    }
    
    all_match
}

fn compare_ms2_windows(windows1: &[((f32, f32), TimsTOFData)], 
                       windows2: &[((f32, f32), TimsTOFData)]) -> bool {
    println!("\n  Comparing MS2 windows...");
    let mut all_match = true;
    
    if windows1.len() != windows2.len() {
        println!("    ❌ Window count mismatch: {} vs {}", 
                windows1.len(), windows2.len());
        all_match = false;
    } else {
        println!("    ✓ Window count match: {}", windows1.len());
    }
    
    // 创建HashMap以匹配窗口
    let mut map1: HashMap<(u32, u32), &TimsTOFData> = HashMap::new();
    let mut map2: HashMap<(u32, u32), &TimsTOFData> = HashMap::new();
    
    for ((low, high), data) in windows1 {
        let key = ((low * 10_000.0).round() as u32, (high * 10_000.0).round() as u32);
        map1.insert(key, data);
    }
    
    for ((low, high), data) in windows2 {
        let key = ((low * 10_000.0).round() as u32, (high * 10_000.0).round() as u32);
        map2.insert(key, data);
    }
    
    // 比较每个窗口
    for (key, data1) in &map1 {
        match map2.get(key) {
            Some(data2) => {
                let window_name = format!("MS2 window ({:.2}, {:.2})", 
                                         key.0 as f32 / 10_000.0, 
                                         key.1 as f32 / 10_000.0);
                if !compare_tims_data(data1, data2, &window_name) {
                    all_match = false;
                }
            }
            None => {
                println!("    ❌ Window ({:.2}, {:.2}) missing in second dataset", 
                        key.0 as f32 / 10_000.0, key.1 as f32 / 10_000.0);
                all_match = false;
            }
        }
    }
    
    // 检查第二个数据集中是否有额外的窗口
    for key in map2.keys() {
        if !map1.contains_key(key) {
            println!("    ❌ Window ({:.2}, {:.2}) missing in first dataset", 
                    key.0 as f32 / 10_000.0, key.1 as f32 / 10_000.0);
            all_match = false;
        }
    }
    
    all_match
}

fn main() -> Result<(), Box<dyn Error>> {
    // 配置并行处理
    rayon::ThreadPoolBuilder::new()
        .num_threads(32)
        .build_global()
        .unwrap();
    
    // 设置数据文件路径
    let d_folder_path = "/storage/guotiannanLab/wangshuaiyao/006.DIABERT_TimsTOF_Rust/test_data/CAD20220207yuel_TPHP_DIA_pool1_Slot2-54_1_4382.d";
    
    let d_path = Path::new(d_folder_path);
    if !d_path.exists() {
        return Err(format!("Folder {:?} not found", d_path).into());
    }
    
    println!("========== TimsTOF Version Comparison Tool ==========");
    println!("Data folder: {}", d_folder_path);
    println!();
    
    // 读取原始版本数据
    println!(">>> Reading data with ORIGINAL version...");
    let data_original = original_version::read_timstof_data_original(d_path)?;
    println!();
    
    // 读取V5优化版本数据
    println!(">>> Reading data with V5_FIXED version...");
    let data_v5 = v5_fixed::read_timstof_data_v5_fixed(d_path)?;
    println!();
    
    // 比较结果
    println!("========== COMPARISON RESULTS ==========");
    
    // 比较MS1数据
    let ms1_match = compare_tims_data(&data_original.ms1_data, &data_v5.ms1_data, "MS1");
    
    // 比较MS2数据
    let ms2_match = compare_ms2_windows(&data_original.ms2_windows, &data_v5.ms2_windows);
    
    // 总结
    println!("\n========== SUMMARY ==========");
    if ms1_match && ms2_match {
        println!("✅ SUCCESS: Both versions produce IDENTICAL results!");
    } else {
        println!("❌ FAILURE: The versions produce DIFFERENT results!");
        if !ms1_match {
            println!("  - MS1 data has differences");
        }
        if !ms2_match {
            println!("  - MS2 data has differences");
        }
    }
    
    Ok(())
}

// Cargo.toml 依赖：
/*
[dependencies]
timsrust = "0.4"
rayon = "1.7"
dashmap = "5.5"
crossbeam-channel = "0.5"
parking_lot = "0.12"
mimalloc = { version = "0.1", features = ["secure"] }

[profile.release]
opt-level = 3
lto = true
codegen-units = 1
*/