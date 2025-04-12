#![feature(portable_simd)]
#![feature(slice_as_chunks)]

use std::arch::x86_64::{__m256i, _mm256_broadcast_sd, _mm256_cmpeq_epi64, _mm256_or_si256};
use std::hint::black_box;
use std::mem;
use std::mem::transmute;
use std::simd::Simd;

struct Point(i32, i32);

impl Point {
    fn as_u64(&self) -> u64 {
        let mut res = self.0 as u64;
        res = res | ((self.1 as u64) << 32);
        res
    }

    fn as_f64(&self) -> f64 {
        // sometimes bit functions only accept "floats"
        f64::from_bits(self.as_u64())
    }
}

#[target_feature(enable = "avx2")]
unsafe fn contains_simd(input_raw: &[Point], needle_pos: Point) {
    let needle = _mm256_broadcast_sd(&needle_pos.as_f64());

    let input: &[u64] = transmute(input_raw);

    const REGISTER_SIZE_64: usize = 4;
    let (pre, register, post) = input.as_simd::<REGISTER_SIZE_64>();
    let (superchunks, after) = register.as_chunks::<4>();
    for chunk in superchunks {
        let c1 = _mm256_cmpeq_epi64(__m256i::from(chunk[0]), needle);
        let c2 = __m256i::from(chunk[1]);
        let c3 = __m256i::from(chunk[2]);
        let c4 = __m256i::from(chunk[3]);

        let c1 = _mm256_cmpeq_epi64(r1, r2);
        let c2 = _mm256_cmpeq_epi64(r3, r4);

        let chunk_result = _mm256_or_si256(c1, c2);
    }
}

fn main() {
    // let needle = Point(black_box(5), black_box(8));
    // println!("{:064b}", needle.0);
    // println!("{:064b}", needle.1);
    // println!("{:064b}", needle.as_u64());
}
