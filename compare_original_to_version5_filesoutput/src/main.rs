use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use std::io::{Read, Write, BufReader, BufWriter};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;
use timsrust::{converters::ConvertableDomain, readers::{FrameReader, MetadataReader}, MSLevel};
use rayon::prelude::*;
use dashmap::DashMap;
use crossbeam_channel::bounded;
use parking_lot::Mutex;
use serde::{Serialize, Deserialize};
use bincode;
use sha2::{Sha256, Digest};

// ============= 数据结构（支持序列化）=============
#[derive(Debug, Clone, Serialize, Deserialize)]
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
    
    // 计算数据的哈希值
    pub fn calculate_hash(&self) -> String {
        let mut hasher = Sha256::new();
        
        // 先排序数据以确保顺序一致
        let mut indices: Vec<usize> = (0..self.mz_values.len()).collect();
        indices.sort_by_key(|&i| {
            (
                self.frame_indices[i],
                self.scan_indices[i],
                (self.mz_values[i] * 1e6) as i64,
                self.intensity_values[i]
            )
        });
        
        // 按排序后的顺序计算哈希
        for &i in &indices {
            hasher.update(self.rt_values_min[i].to_le_bytes());
            hasher.update(self.mobility_values[i].to_le_bytes());
            hasher.update(self.mz_values[i].to_le_bytes());
            hasher.update(self.intensity_values[i].to_le_bytes());
            hasher.update(self.frame_indices[i].to_le_bytes());
            hasher.update(self.scan_indices[i].to_le_bytes());
        }
        
        format!("{:x}", hasher.finalize())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimsTOFRawData {
    pub ms1_data: TimsTOFData,
    pub ms2_windows: Vec<((f32, f32), TimsTOFData)>,
}

impl TimsTOFRawData {
    // 保存为二进制文件
    pub fn save_binary(&self, filename: &str) -> Result<(), Box<dyn Error>> {
        println!("  Saving to binary file: {}", filename);
        let file = File::create(filename)?;
        let writer = BufWriter::new(file);
        bincode::serialize_into(writer, self)?;
        
        // 计算文件大小
        let metadata = std::fs::metadata(filename)?;
        let file_size = metadata.len();
        println!("    Binary file size: {:.2} MB", file_size as f64 / 1_048_576.0);
        
        Ok(())
    }
    
    // 从二进制文件加载
    pub fn load_binary(filename: &str) -> Result<Self, Box<dyn Error>> {
        println!("  Loading from binary file: {}", filename);
        let file = File::open(filename)?;
        let reader = BufReader::new(file);
        let data = bincode::deserialize_from(reader)?;
        Ok(data)
    }
    
    // 保存为JSON文件（可读性好，但文件较大）
    pub fn save_json(&self, filename: &str) -> Result<(), Box<dyn Error>> {
        println!("  Saving to JSON file: {}", filename);
        let json = serde_json::to_string_pretty(self)?;
        let mut file = File::create(filename)?;
        file.write_all(json.as_bytes())?;
        
        let metadata = std::fs::metadata(filename)?;
        let file_size = metadata.len();
        println!("    JSON file size: {:.2} MB", file_size as f64 / 1_048_576.0);
        
        Ok(())
    }
    
    // 从JSON文件加载
    pub fn load_json(filename: &str) -> Result<Self, Box<dyn Error>> {
        println!("  Loading from JSON file: {}", filename);
        let mut file = File::open(filename)?;
        let mut contents = String::new();
        file.read_to_string(&mut contents)?;
        let data = serde_json::from_str(&contents)?;
        Ok(data)
    }
    
    // 保存数据摘要（用于快速验证）
    pub fn save_summary(&self, filename: &str) -> Result<(), Box<dyn Error>> {
        println!("  Saving summary to: {}", filename);
        let mut file = File::create(filename)?;
        
        writeln!(file, "=== TimsTOF Data Summary ===")?;
        writeln!(file, "MS1 Data Points: {}", self.ms1_data.mz_values.len())?;
        writeln!(file, "MS1 Data Hash: {}", self.ms1_data.calculate_hash())?;
        writeln!(file, "MS2 Windows: {}", self.ms2_windows.len())?;
        
        let total_ms2_points: usize = self.ms2_windows.iter()
            .map(|(_, td)| td.mz_values.len()).sum();
        writeln!(file, "Total MS2 Data Points: {}", total_ms2_points)?;
        
        // 计算每个MS2窗口的哈希
        writeln!(file, "\n=== MS2 Window Hashes ===")?;
        let mut windows_sorted = self.ms2_windows.clone();
        windows_sorted.sort_by(|a, b| {
            a.0.0.partial_cmp(&b.0.0).unwrap()
                .then(a.0.1.partial_cmp(&b.0.1).unwrap())
        });
        
        for ((low, high), data) in &windows_sorted {
            writeln!(file, "Window ({:.4}, {:.4}): {} points, hash: {}", 
                    low, high, data.mz_values.len(), 
                    &data.calculate_hash()[..16])?; // 只显示前16个字符
        }
        
        Ok(())
    }
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

// ============= V5优化版本实现 =============
mod v5_fixed {
    use super::*;
    
    #[derive(Clone)]
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

// ============= 文件比较工具 =============
fn compare_binary_files(file1: &str, file2: &str) -> Result<bool, Box<dyn Error>> {
    println!("\n  Comparing binary files byte-by-byte...");
    println!("    File 1: {}", file1);
    println!("    File 2: {}", file2);
    
    let mut f1 = File::open(file1)?;
    let mut f2 = File::open(file2)?;
    
    let mut buffer1 = Vec::new();
    let mut buffer2 = Vec::new();
    
    f1.read_to_end(&mut buffer1)?;
    f2.read_to_end(&mut buffer2)?;
    
    if buffer1.len() != buffer2.len() {
        println!("    ❌ File sizes differ: {} vs {} bytes", 
                buffer1.len(), buffer2.len());
        return Ok(false);
    }
    
    println!("    File size: {} bytes", buffer1.len());
    
    // 计算文件哈希
    let hash1 = format!("{:x}", Sha256::digest(&buffer1));
    let hash2 = format!("{:x}", Sha256::digest(&buffer2));
    
    println!("    File 1 SHA256: {}", hash1);
    println!("    File 2 SHA256: {}", hash2);
    
    if hash1 != hash2 {
        println!("    ❌ Files have different SHA256 hashes!");
        
        // 找出第一个不同的字节
        for (i, (b1, b2)) in buffer1.iter().zip(buffer2.iter()).enumerate() {
            if b1 != b2 {
                println!("    First difference at byte {}: 0x{:02x} vs 0x{:02x}", 
                        i, b1, b2);
                break;
            }
        }
        return Ok(false);
    }
    
    println!("    ✓ Files are identical!");
    Ok(true)
}

fn compare_data_from_files(file1: &str, file2: &str) -> Result<bool, Box<dyn Error>> {
    println!("\n  Loading and comparing data structures...");
    
    let data1 = TimsTOFRawData::load_binary(file1)?;
    let data2 = TimsTOFRawData::load_binary(file2)?;
    
    let mut all_match = true;
    
    // 比较MS1数据
    println!("\n  Comparing MS1 data...");
    let ms1_hash1 = data1.ms1_data.calculate_hash();
    let ms1_hash2 = data2.ms1_data.calculate_hash();
    
    if ms1_hash1 != ms1_hash2 {
        println!("    ❌ MS1 data hashes differ!");
        println!("      Data1: {}", ms1_hash1);
        println!("      Data2: {}", ms1_hash2);
        all_match = false;
    } else {
        println!("    ✓ MS1 data matches (hash: {}...)", &ms1_hash1[..16]);
    }
    
    // 比较MS2窗口数量
    println!("\n  Comparing MS2 windows...");
    if data1.ms2_windows.len() != data2.ms2_windows.len() {
        println!("    ❌ Different number of MS2 windows: {} vs {}", 
                data1.ms2_windows.len(), data2.ms2_windows.len());
        all_match = false;
    } else {
        println!("    ✓ Same number of MS2 windows: {}", data1.ms2_windows.len());
        
        // 创建HashMap以匹配窗口
        let mut map1: HashMap<(u32, u32), &TimsTOFData> = HashMap::new();
        let mut map2: HashMap<(u32, u32), &TimsTOFData> = HashMap::new();
        
        for ((low, high), data) in &data1.ms2_windows {
            let key = ((low * 10_000.0).round() as u32, (high * 10_000.0).round() as u32);
            map1.insert(key, data);
        }
        
        for ((low, high), data) in &data2.ms2_windows {
            let key = ((low * 10_000.0).round() as u32, (high * 10_000.0).round() as u32);
            map2.insert(key, data);
        }
        
        // 比较每个窗口的哈希
        for (key, data1) in &map1 {
            match map2.get(key) {
                Some(data2) => {
                    let hash1 = data1.calculate_hash();
                    let hash2 = data2.calculate_hash();
                    if hash1 != hash2 {
                        println!("    ❌ MS2 window ({:.2}, {:.2}) data differs", 
                                key.0 as f32 / 10_000.0, key.1 as f32 / 10_000.0);
                        all_match = false;
                    }
                }
                None => {
                    println!("    ❌ MS2 window ({:.2}, {:.2}) missing in second dataset", 
                            key.0 as f32 / 10_000.0, key.1 as f32 / 10_000.0);
                    all_match = false;
                }
            }
        }
    }
    
    Ok(all_match)
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
    
    // 创建输出目录
    std::fs::create_dir_all("./timstof_comparison_output")?;
    
    println!("========== TimsTOF File-Based Version Comparison Tool ==========");
    println!("Data folder: {}", d_folder_path);
    println!("Output directory: ./timstof_comparison_output/");
    println!();
    
    // ===== 步骤1：运行原始版本并保存 =====
    println!(">>> STEP 1: Running ORIGINAL version and saving to files...");
    let data_original = original_version::read_timstof_data_original(d_path)?;
    
    println!("\n[ORIGINAL] Saving data to files...");
    data_original.save_binary("./timstof_comparison_output/original_data.bin")?;
    data_original.save_json("./timstof_comparison_output/original_data.json")?;
    data_original.save_summary("./timstof_comparison_output/original_summary.txt")?;
    println!();
    
    // ===== 步骤2：运行V5版本并保存 =====
    println!(">>> STEP 2: Running V5_FIXED version and saving to files...");
    let data_v5 = v5_fixed::read_timstof_data_v5_fixed(d_path)?;
    
    println!("\n[V5_FIXED] Saving data to files...");
    data_v5.save_binary("./timstof_comparison_output/v5_fixed_data.bin")?;
    data_v5.save_json("./timstof_comparison_output/v5_fixed_data.json")?;
    data_v5.save_summary("./timstof_comparison_output/v5_fixed_summary.txt")?;
    println!();
    
    // ===== 步骤3：比较文件 =====
    println!("========== FILE COMPARISON RESULTS ==========");
    
    // 二进制文件字节级比较
    println!("\n>>> Comparing binary files...");
    let binary_match = compare_binary_files(
        "./timstof_comparison_output/original_data.bin",
        "./timstof_comparison_output/v5_fixed_data.bin"
    )?;
    
    // 如果二进制文件不同，进一步分析数据内容
    let content_match = if !binary_match {
        println!("\n>>> Binary files differ, analyzing data content...");
        compare_data_from_files(
            "./timstof_comparison_output/original_data.bin",
            "./timstof_comparison_output/v5_fixed_data.bin"
        )?
    } else {
        true
    };
    
    // ===== 步骤4：生成比较报告 =====
    println!("\n========== FINAL COMPARISON REPORT ==========");
    
    let mut report = File::create("./timstof_comparison_output/comparison_report.txt")?;
    writeln!(report, "TimsTOF Version Comparison Report")?;
    writeln!(report, "==================================")?;
    writeln!(report, "Generated: {}", chrono::Local::now())?;
    writeln!(report)?;
    
    if binary_match {
        println!("✅ SUCCESS: Binary files are IDENTICAL!");
        println!("   This means both versions produce exactly the same output.");
        writeln!(report, "Result: ✅ SUCCESS")?;
        writeln!(report, "Binary files are identical - both versions produce exactly the same output.")?;
    } else if content_match {
        println!("⚠️  WARNING: Binary files differ but data content matches!");
        println!("   This means the data is the same but stored in different order.");
        writeln!(report, "Result: ⚠️ WARNING")?;
        writeln!(report, "Binary files differ but data content matches.")?;
        writeln!(report, "The data is the same but may be stored in different order.")?;
    } else {
        println!("❌ FAILURE: The versions produce DIFFERENT results!");
        println!("   Check the summary files for details.");
        writeln!(report, "Result: ❌ FAILURE")?;
        writeln!(report, "The versions produce different results.")?;
        writeln!(report, "Check the summary files for detailed differences.")?;
    }
    
    writeln!(report)?;
    writeln!(report, "Files generated:")?;
    writeln!(report, "  - original_data.bin: Binary data from original version")?;
    writeln!(report, "  - v5_fixed_data.bin: Binary data from V5 fixed version")?;
    writeln!(report, "  - original_data.json: JSON data (human-readable)")?;
    writeln!(report, "  - v5_fixed_data.json: JSON data (human-readable)")?;
    writeln!(report, "  - original_summary.txt: Data summary with hashes")?;
    writeln!(report, "  - v5_fixed_summary.txt: Data summary with hashes")?;
    
    println!("\n📁 All output files saved to: ./timstof_comparison_output/");
    println!("   You can manually inspect the JSON and summary files for details.");
    
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
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
bincode = "1.3"
sha2 = "0.10"
chrono = "0.4"

[profile.release]
opt-level = 3
lto = true
codegen-units = 1
*/