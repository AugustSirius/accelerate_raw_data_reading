# TimsTOF Raw Data Loading Optimization Strategy

## Current Implementation Analysis

The original code reads TimsTOF .d folder data using the timsrust library with the following characteristics:

### Bottlenecks Identified:
1. **Memory Allocation**: Multiple small allocations during frame processing
2. **Data Merging**: Sequential HashMap operations for MS2 window aggregation
3. **Scan Finding**: Linear search through scan_offsets for each peak
4. **Data Copying**: Multiple `.extend()` operations during merge phase
5. **Thread Count**: Currently hardcoded to 16 threads (not utilizing full 32 cores)

### Current Performance Characteristics:
- Uses rayon for parallel frame processing
- Pre-allocates capacity for some vectors
- Uses quantization for MS2 window keys
- Single-pass frame reading

## Optimization Strategies

### Version 1: Memory-Mapped I/O + Aggressive Pre-allocation
**Techniques:**
- Memory-mapped file access for raw data
- Pre-calculate total data size from metadata
- Single large allocation per data vector
- Custom memory pool for temporary allocations
- Increase thread count to 32

**Expected Benefits:**
- Reduced system calls for file I/O
- Minimized memory fragmentation
- Better cache locality

### Version 2: Lock-free Parallel Aggregation  
**Techniques:**
- Replace HashMap with DashMap for concurrent MS2 aggregation
- Use crossbeam channels for data streaming
- Implement work-stealing queue for frame distribution
- Atomic counters for progress tracking
- Thread-local buffers with periodic merging

**Expected Benefits:**
- Reduced lock contention
- Better CPU utilization
- Smoother parallel scaling

### Version 3: SIMD + Batch Processing
**Techniques:**
- SIMD operations for mz/mobility conversions
- Batch processing of TOF indices (process 8-16 at once)
- Vectorized quantization operations
- Aligned memory allocations for SIMD
- Loop unrolling for critical paths

**Expected Benefits:**
- Higher throughput for numerical conversions
- Better CPU instruction pipelining
- Reduced branch mispredictions

### Version 4: Zero-copy + Custom Allocator
**Techniques:**
- Arena allocator for frame-local data
- Bump allocator for sequential data
- Zero-copy data structures where possible
- Unsafe code for critical performance paths
- jemalloc or mimalloc for better memory management

**Expected Benefits:**
- Minimal allocation overhead
- Reduced memory copying
- Better memory reuse

### Version 5: Hybrid Optimization (Best of All)
**Techniques:**
- Combine DashMap for MS2 aggregation
- SIMD for numerical conversions
- Arena allocator for temporary data
- Memory-mapped I/O for large files
- Adaptive thread pooling based on data size
- Binary search for scan finding
- Parallel sorting for final data organization

**Expected Benefits:**
- Maximum performance combining all optimizations
- Adaptive to different data characteristics
- Best overall throughput

## Implementation Plan

Each version will:
1. Maintain the same API and output format
2. Use 32 threads as default (configurable)
3. Include timing measurements for each phase
4. Use the same HPC data path
5. Keep the same Rust_run.sh structure

## Performance Metrics to Track

- Total processing time
- Memory usage (peak and average)
- Thread utilization
- I/O throughput
- Cache misses (if measurable)
- Time breakdown by phase:
  - Metadata loading
  - Frame reading
  - Data processing
  - Merging/aggregation

## Build Configuration

All versions will use:
- `opt-level = 3`
- `lto = "fat"`
- `codegen-units = 1`
- CPU-specific optimizations for AMD EPYC 7502
- Release build with maximum optimizations