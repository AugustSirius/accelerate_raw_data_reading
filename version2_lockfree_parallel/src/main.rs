use std::error::Error;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;
use timsrust::{converters::ConvertableDomain, readers::{FrameReader, MetadataReader}, MSLevel};
use rayon::prelude::*;
use dashmap::DashMap;
use crossbeam_channel::{bounded, Sender, Receiver};
use parking_lot::Mutex;

const NUM_THREADS: usize = 32;
const CHANNEL_BUFFER_SIZE: usize = 1000;

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
    
    fn merge_from(&mut self, other: TimsTOFData) {
        self.rt_values_min.extend(other.rt_values_min);
        self.mobility_values.extend(other.mobility_values);
        self.mz_values.extend(other.mz_values);
        self.intensity_values.extend(other.intensity_values);
        self.frame_indices.extend(other.frame_indices);
        self.scan_indices.extend(other.scan_indices);
    }
}

#[derive(Debug, Clone)]
pub struct TimsTOFRawData {
    pub ms1_data: TimsTOFData,
    pub ms2_windows: Vec<((f32, f32), TimsTOFData)>,
}

enum ProcessedFrame {
    MS1(TimsTOFData),
    MS2(Vec<((u32, u32), TimsTOFData)>),
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

fn process_frame_worker(
    frame_idx: usize,
    frames: &FrameReader,
    mz_cv: Arc<impl ConvertableDomain>,
    im_cv: Arc<impl ConvertableDomain>,
    sender: Sender<ProcessedFrame>,
) {
    let frame = match frames.get(frame_idx) {
        Ok(f) => f,
        Err(_) => return,
    };
    
    let rt_min = frame.rt_in_seconds as f32 / 60.0;
    
    match frame.ms_level {
        MSLevel::MS1 => {
            let n_peaks = frame.tof_indices.len();
            let mut ms1 = TimsTOFData::with_capacity(n_peaks);
            
            for (p_idx, (&tof, &intensity)) in frame.tof_indices.iter()
                .zip(frame.intensities.iter()).enumerate() {
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
            
            let _ = sender.send(ProcessedFrame::MS1(ms1));
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
                for (p_idx, (&tof, &intensity)) in frame.tof_indices.iter()
                    .zip(frame.intensities.iter()).enumerate() {
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
                    ms2_pairs.push((key, td));
                }
            }
            
            if !ms2_pairs.is_empty() {
                let _ = sender.send(ProcessedFrame::MS2(ms2_pairs));
            }
        }
        _ => {}
    }
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
    let frames = Arc::new(FrameReader::new(d_folder)?);
    let n_frames = frames.len();
    println!("  Frame reader initialization: {:.3}s", frame_reader_start.elapsed().as_secs_f32());
    println!("  Total frames to process: {}", n_frames);
    
    println!("Processing frames with lock-free parallel aggregation ({} threads)...", NUM_THREADS);
    let process_start = Instant::now();
    
    let (sender, receiver) = bounded(CHANNEL_BUFFER_SIZE);
    let processed_count = Arc::new(AtomicUsize::new(0));
    
    let ms1_accumulator = Arc::new(Mutex::new(Vec::new()));
    let ms2_map = Arc::new(DashMap::new());
    
    let ms1_acc_clone = Arc::clone(&ms1_accumulator);
    let ms2_map_clone = Arc::clone(&ms2_map);
    let processed_clone = Arc::clone(&processed_count);
    
    let aggregator_handle = std::thread::spawn(move || {
        while let Ok(frame_data) = receiver.recv() {
            match frame_data {
                ProcessedFrame::MS1(data) => {
                    ms1_acc_clone.lock().push(data);
                }
                ProcessedFrame::MS2(pairs) => {
                    for (key, data) in pairs {
                        ms2_map_clone.entry(key)
                            .or_insert_with(|| Arc::new(Mutex::new(TimsTOFData::new())))
                            .lock()
                            .merge_from(data);
                    }
                }
            }
            processed_clone.fetch_add(1, Ordering::Relaxed);
        }
    });
    
    (0..n_frames).into_par_iter().for_each(|idx| {
        let frames_ref = &*frames;
        let mz_cv_clone = Arc::clone(&mz_cv);
        let im_cv_clone = Arc::clone(&im_cv);
        let sender_clone = sender.clone();
        
        process_frame_worker(idx, frames_ref, mz_cv_clone, im_cv_clone, sender_clone);
    });
    
    drop(sender);
    aggregator_handle.join().unwrap();
    
    println!("  Frame processing: {:.3}s", process_start.elapsed().as_secs_f32());
    println!("  Frames processed: {}", processed_count.load(Ordering::Relaxed));
    
    println!("Finalizing data structures...");
    let finalize_start = Instant::now();
    
    let ms1_chunks = ms1_accumulator.lock();
    let total_ms1_size: usize = ms1_chunks.iter().map(|c| c.mz_values.len()).sum();
    let mut global_ms1 = TimsTOFData::with_capacity(total_ms1_size);
    for chunk in ms1_chunks.iter() {
        global_ms1.rt_values_min.extend(&chunk.rt_values_min);
        global_ms1.mobility_values.extend(&chunk.mobility_values);
        global_ms1.mz_values.extend(&chunk.mz_values);
        global_ms1.intensity_values.extend(&chunk.intensity_values);
        global_ms1.frame_indices.extend(&chunk.frame_indices);
        global_ms1.scan_indices.extend(&chunk.scan_indices);
    }
    
    let mut ms2_vec = Vec::with_capacity(ms2_map.len());
    for entry in ms2_map.iter() {
        let (q_low, q_high) = *entry.key();
        let low = q_low as f32 / 10_000.0;
        let high = q_high as f32 / 10_000.0;
        let data = entry.value().lock().clone();
        ms2_vec.push(((low, high), data));
    }
    
    println!("  Data finalization: {:.3}s", finalize_start.elapsed().as_secs_f32());
    
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
    
    println!("========== TimsTOF .d File Reader V2 (Lock-free Parallel Aggregation) ==========");
    println!("Data folder: {}", d_folder_path);
    println!("Parallel threads: {}", NUM_THREADS);
    println!();
    
    let _raw_data = read_timstof_data(d_path)?;
    
    println!("\n========== Processing Complete ==========");
    
    Ok(())
}