use std::collections::HashMap;
use std::error::Error;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;
use timsrust::{converters::ConvertableDomain, readers::{FrameReader, MetadataReader}, MSLevel};
use rayon::prelude::*;
use bumpalo::Bump;
use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

const NUM_THREADS: usize = 32;
const ARENA_SIZE: usize = 64 * 1024 * 1024; // 64MB per arena

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
fn quantize(x: f32) -> u32 { 
    unsafe { (x * 10_000.0).to_int_unchecked::<u32>() }
}

#[inline(always)]
fn find_scan_for_index_binary_unsafe(index: usize, scan_offsets: &[usize]) -> usize {
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
    fn process_peaks_unchecked(
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
        
        unsafe {
            data.rt_values_min.set_len(n_peaks);
            data.mobility_values.set_len(n_peaks);
            data.mz_values.set_len(n_peaks);
            data.intensity_values.set_len(n_peaks);
            data.frame_indices.set_len(n_peaks);
            data.scan_indices.set_len(n_peaks);
            
            let rt_ptr = data.rt_values_min.as_mut_ptr();
            let im_ptr = data.mobility_values.as_mut_ptr();
            let mz_ptr = data.mz_values.as_mut_ptr();
            let int_ptr = data.intensity_values.as_mut_ptr();
            let frame_ptr = data.frame_indices.as_mut_ptr();
            let scan_ptr = data.scan_indices.as_mut_ptr();
            
            let mut out_idx = 0;
            for (i, (&tof, &intensity)) in tof_indices.iter().zip(intensities.iter()).enumerate() {
                let scan = find_scan_for_index_binary_unsafe(i, scan_offsets);
                
                if let Some((start, end)) = scan_filter {
                    if scan < start || scan > end { continue; }
                }
                
                let mz = self.mz_cv.convert(tof as f64) as f32;
                let im = self.im_cv.convert(scan as f64) as f32;
                
                *rt_ptr.add(out_idx) = rt_min;
                *im_ptr.add(out_idx) = im;
                *mz_ptr.add(out_idx) = mz;
                *int_ptr.add(out_idx) = intensity;
                *frame_ptr.add(out_idx) = frame_index;
                *scan_ptr.add(out_idx) = scan as u32;
                
                out_idx += 1;
            }
            
            if out_idx < n_peaks {
                data.rt_values_min.set_len(out_idx);
                data.mobility_values.set_len(out_idx);
                data.mz_values.set_len(out_idx);
                data.intensity_values.set_len(out_idx);
                data.frame_indices.set_len(out_idx);
                data.scan_indices.set_len(out_idx);
            }
        }
        
        data
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
    let frames = FrameReader::new(d_folder)?;
    let n_frames = frames.len();
    println!("  Frame reader initialization: {:.3}s", frame_reader_start.elapsed().as_secs_f32());
    println!("  Total frames to process: {}", n_frames);
    
    println!("Processing frames with zero-copy and custom allocator ({} threads)...", NUM_THREADS);
    let process_start = Instant::now();
    
    let splits: Vec<FrameSplit> = (0..n_frames).into_par_iter().map(|idx| {
        let arena = Bump::with_capacity(ARENA_SIZE);
        let processor = FrameProcessor {
            arena: &arena,
            mz_cv: Arc::clone(&mz_cv),
            im_cv: Arc::clone(&im_cv),
        };
        
        let frame = frames.get(idx).expect("frame read");
        let rt_min = frame.rt_in_seconds as f32 / 60.0;
        let mut ms1 = TimsTOFData::new();
        let mut ms2_pairs: Vec<((u32,u32), TimsTOFData)> = Vec::new();
        
        match frame.ms_level {
            MSLevel::MS1 => {
                ms1 = processor.process_peaks_unchecked(
                    &frame.tof_indices,
                    &frame.intensities,
                    &frame.scan_offsets,
                    rt_min,
                    frame.index as u32,
                    None,
                );
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
                    
                    let td = processor.process_peaks_unchecked(
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
            }
            _ => {}
        }
        
        FrameSplit { ms1, ms2: ms2_pairs }
    }).collect();
    
    println!("  Frame processing: {:.3}s", process_start.elapsed().as_secs_f32());
    
    println!("Merging data with zero-copy operations...");
    let merge_start = Instant::now();
    
    let ms1_size_estimate: usize = splits.par_iter().map(|s| s.ms1.mz_values.len()).sum();
    let mut global_ms1 = TimsTOFData::with_capacity(ms1_size_estimate);
    let mut ms2_hash: HashMap<(u32,u32), TimsTOFData> = HashMap::new();
    
    for mut split in splits {
        unsafe { global_ms1.append_unchecked(&mut split.ms1); }
        
        for (key, mut td) in split.ms2 {
            match ms2_hash.get_mut(&key) {
                Some(existing) => unsafe { existing.append_unchecked(&mut td); },
                None => { ms2_hash.insert(key, td); }
            }
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
    
    println!("========== TimsTOF .d File Reader V4 (Zero-copy + Custom Allocator) ==========");
    println!("Data folder: {}", d_folder_path);
    println!("Parallel threads: {}", NUM_THREADS);
    println!("Using mimalloc allocator + bump allocators");
    println!();
    
    let _raw_data = read_timstof_data(d_path)?;
    
    println!("\n========== Processing Complete ==========");
    
    Ok(())
}