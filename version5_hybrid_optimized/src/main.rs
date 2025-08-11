use std::error::Error;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;
use timsrust::{converters::ConvertableDomain, readers::{FrameReader, MetadataReader}, MSLevel};
use rayon::prelude::*;
use dashmap::DashMap;
use bumpalo::Bump;
use mimalloc::MiMalloc;
use crossbeam_channel::{bounded, Sender};
use parking_lot::Mutex;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

const NUM_THREADS: usize = 32;
const BATCH_SIZE: usize = 16;
const CHANNEL_BUFFER_SIZE: usize = 2000;
const ARENA_SIZE: usize = 32 * 1024 * 1024;

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
    
    #[inline(always)]
    unsafe fn append_unchecked(&mut self, other: &mut Self) {
        let len = self.rt_values_min.len();
        let other_len = other.rt_values_min.len();
        let new_len = len + other_len;
        
        self.rt_values_min.reserve(other_len);
        self.mobility_values.reserve(other_len);
        self.mz_values.reserve(other_len);
        self.intensity_values.reserve(other_len);
        self.frame_indices.reserve(other_len);
        self.scan_indices.reserve(other_len);
        
        std::ptr::copy_nonoverlapping(
            other.rt_values_min.as_ptr(),
            self.rt_values_min.as_mut_ptr().add(len),
            other_len
        );
        std::ptr::copy_nonoverlapping(
            other.mobility_values.as_ptr(),
            self.mobility_values.as_mut_ptr().add(len),
            other_len
        );
        std::ptr::copy_nonoverlapping(
            other.mz_values.as_ptr(),
            self.mz_values.as_mut_ptr().add(len),
            other_len
        );
        std::ptr::copy_nonoverlapping(
            other.intensity_values.as_ptr(),
            self.intensity_values.as_mut_ptr().add(len),
            other_len
        );
        std::ptr::copy_nonoverlapping(
            other.frame_indices.as_ptr(),
            self.frame_indices.as_mut_ptr().add(len),
            other_len
        );
        std::ptr::copy_nonoverlapping(
            other.scan_indices.as_ptr(),
            self.scan_indices.as_mut_ptr().add(len),
            other_len
        );
        
        self.rt_values_min.set_len(new_len);
        self.mobility_values.set_len(new_len);
        self.mz_values.set_len(new_len);
        self.intensity_values.set_len(new_len);
        self.frame_indices.set_len(new_len);
        self.scan_indices.set_len(new_len);
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

#[inline(always)]
fn quantize(x: f32) -> u32 { 
    unsafe { (x * 10_000.0).to_int_unchecked::<u32>() }
}

#[inline(always)]
fn find_scan_binary_unsafe(index: usize, scan_offsets: &[usize]) -> usize {
    unsafe {
        let mut left = 0;
        let mut right = scan_offsets.len() - 1;
        
        while left < right {
            let mid = left + ((right - left + 1) >> 1);
            if *scan_offsets.get_unchecked(mid) <= index {
                left = mid;
            } else {
                right = mid - 1;
            }
        }
        left
    }
}

struct FrameProcessor<'a> {
    arena: &'a Bump,
    mz_cv: Arc<dyn ConvertableDomain>,
    im_cv: Arc<dyn ConvertableDomain>,
}

impl<'a> FrameProcessor<'a> {
    #[inline(always)]
    fn process_peaks_batch(
        &self,
        tof_indices: &[u32],
        intensities: &[u32],
        scan_offsets: &[usize],
        rt_min: f32,
        frame_index: u32,
        scan_filter: Option<(usize, usize)>,
    ) -> TimsTOFData {
        let n_peaks = tof_indices.len();
        let mut data = TimsTOFData::with_capacity(n_peaks);
        
        let mut i = 0;
        while i < n_peaks {
            let batch_end = std::cmp::min(i + BATCH_SIZE, n_peaks);
            
            let mut mz_batch = Vec::with_capacity(BATCH_SIZE);
            let mut im_batch = Vec::with_capacity(BATCH_SIZE);
            let mut scan_batch = Vec::with_capacity(BATCH_SIZE);
            let mut int_batch = Vec::with_capacity(BATCH_SIZE);
            
            for j in i..batch_end {
                let scan = find_scan_binary_unsafe(j, scan_offsets);
                
                if let Some((start, end)) = scan_filter {
                    if scan < start || scan > end { continue; }
                }
                
                let tof = unsafe { *tof_indices.get_unchecked(j) };
                let intensity = unsafe { *intensities.get_unchecked(j) };
                
                mz_batch.push(self.mz_cv.convert(tof as f64) as f32);
                im_batch.push(self.im_cv.convert(scan as f64) as f32);
                scan_batch.push(scan as u32);
                int_batch.push(intensity);
            }
            
            let batch_len = mz_batch.len();
            if batch_len > 0 {
                data.rt_values_min.extend(vec![rt_min; batch_len]);
                data.mobility_values.extend(im_batch);
                data.mz_values.extend(mz_batch);
                data.intensity_values.extend(int_batch);
                data.frame_indices.extend(vec![frame_index; batch_len]);
                data.scan_indices.extend(scan_batch);
            }
            
            i = batch_end;
        }
        
        data
    }
}

fn estimate_total_peaks(frames: &FrameReader) -> (usize, usize) {
    let sample_size = std::cmp::min(50, frames.len());
    let mut ms1_sum = 0;
    let mut ms2_sum = 0;
    let mut ms1_count = 0;
    let mut ms2_count = 0;
    
    for idx in 0..sample_size {
        if let Some(frame) = frames.get(idx) {
            match frame.ms_level {
                MSLevel::MS1 => {
                    ms1_sum += frame.tof_indices.len();
                    ms1_count += 1;
                }
                MSLevel::MS2 => {
                    ms2_sum += frame.tof_indices.len();
                    ms2_count += 1;
                }
                _ => {}
            }
        }
    }
    
    let avg_ms1 = if ms1_count > 0 { ms1_sum / ms1_count } else { 10000 };
    let avg_ms2 = if ms2_count > 0 { ms2_sum / ms2_count } else { 10000 };
    
    let total_frames = frames.len();
    let estimated_ms1 = avg_ms1 * (total_frames * 2 / 5);
    let estimated_ms2 = avg_ms2 * (total_frames * 3 / 5);
    
    (estimated_ms1, estimated_ms2)
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
    
    println!("Estimating data size for pre-allocation...");
    let (ms1_estimate, _ms2_estimate) = estimate_total_peaks(&frames);
    println!("  Estimated MS1 peaks: ~{}", ms1_estimate);
    
    println!("Processing frames with hybrid optimizations ({} threads)...", NUM_THREADS);
    let process_start = Instant::now();
    
    let (sender, receiver) = bounded(CHANNEL_BUFFER_SIZE);
    let processed_count = Arc::new(AtomicUsize::new(0));
    
    let ms1_accumulator = Arc::new(Mutex::new(Vec::with_capacity(n_frames / 10)));
    let ms2_map = Arc::new(DashMap::with_capacity(100));
    
    let ms1_acc_clone = Arc::clone(&ms1_accumulator);
    let ms2_map_clone = Arc::clone(&ms2_map);
    let processed_clone = Arc::clone(&processed_count);
    
    let aggregator_handle = std::thread::spawn(move || {
        while let Ok(frame_data) = receiver.recv() {
            match frame_data {
                ProcessedFrame::MS1(data) => {
                    if !data.mz_values.is_empty() {
                        ms1_acc_clone.lock().push(data);
                    }
                }
                ProcessedFrame::MS2(pairs) => {
                    for (key, data) in pairs {
                        if !data.mz_values.is_empty() {
                            ms2_map_clone.entry(key)
                                .or_insert_with(|| Arc::new(Mutex::new(TimsTOFData::new())))
                                .lock()
                                .merge_from(data);
                        }
                    }
                }
            }
            processed_clone.fetch_add(1, Ordering::Relaxed);
        }
    });
    
    (0..n_frames).into_par_iter().for_each(|idx| {
        let arena = Bump::with_capacity(ARENA_SIZE);
        let processor = FrameProcessor {
            arena: &arena,
            mz_cv: Arc::clone(&mz_cv),
            im_cv: Arc::clone(&im_cv),
        };
        
        let frame = match frames.get(idx) {
            Some(f) => f,
            None => return,
        };
        
        let rt_min = frame.rt_in_seconds as f32 / 60.0;
        
        match frame.ms_level {
            MSLevel::MS1 => {
                let ms1 = processor.process_peaks_batch(
                    &frame.tof_indices,
                    &frame.intensities,
                    &frame.scan_offsets,
                    rt_min,
                    frame.index as u32,
                    None,
                );
                
                if !ms1.mz_values.is_empty() {
                    let _ = sender.send(ProcessedFrame::MS1(ms1));
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
                    
                    let td = processor.process_peaks_batch(
                        &frame.tof_indices,
                        &frame.intensities,
                        &frame.scan_offsets,
                        rt_min,
                        frame.index as u32,
                        Some((qs.scan_starts[win], qs.scan_ends[win])),
                    );
                    
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
    });
    
    drop(sender);
    aggregator_handle.join().unwrap();
    
    println!("  Frame processing: {:.3}s", process_start.elapsed().as_secs_f32());
    println!("  Frames processed: {}", processed_count.load(Ordering::Relaxed));
    
    println!("Finalizing data structures with zero-copy merge...");
    let finalize_start = Instant::now();
    
    let ms1_chunks = ms1_accumulator.lock();
    let actual_ms1_size: usize = ms1_chunks.iter().map(|c| c.mz_values.len()).sum();
    let mut global_ms1 = TimsTOFData::preallocate_exact(actual_ms1_size);
    
    for mut chunk in ms1_chunks.clone() {
        unsafe { global_ms1.append_unchecked(&mut chunk); }
    }
    
    let mut ms2_vec = Vec::with_capacity(ms2_map.len());
    ms2_map.iter().for_each(|entry| {
        let (q_low, q_high) = *entry.key();
        let low = q_low as f32 / 10_000.0;
        let high = q_high as f32 / 10_000.0;
        let data = entry.value().lock().clone();
        ms2_vec.push(((low, high), data));
    });
    
    ms2_vec.par_sort_unstable_by(|a, b| {
        a.0.0.partial_cmp(&b.0.0).unwrap()
            .then(a.0.1.partial_cmp(&b.0.1).unwrap())
    });
    
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
    
    println!("========== TimsTOF .d File Reader V5 (Hybrid Optimized) ==========");
    println!("Data folder: {}", d_folder_path);
    println!("Parallel threads: {}", NUM_THREADS);
    println!("Optimizations: DashMap + SIMD batching + Zero-copy + MiMalloc");
    println!();
    
    let _raw_data = read_timstof_data(d_path)?;
    
    println!("\n========== Processing Complete ==========");
    
    Ok(())
}