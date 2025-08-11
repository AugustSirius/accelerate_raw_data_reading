#!/bin/bash
#slurm options
#SBATCH -p amd-ep2,intel-sc3,amd-ep2-short
#SBATCH -q normal
#SBATCH -J rust_v5_hybrid
#SBATCH -c 1
#SBATCH -n 32
#SBATCH --mem 200G
########################## MSConvert run #####################
# module
module load gcc
cd /storage/guotiannanLab/wangshuaiyao/006.DIABERT_TimsTOF_Rust/accelerate_raw_data_reading/version5_hybrid_optimized
export RUSTFLAGS="-C target-cpu=znver2"
cargo run --release