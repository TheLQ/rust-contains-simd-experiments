#![feature(portable_simd)]
#![feature(slice_as_chunks)]
// too noisy
#![allow(unsafe_op_in_unsafe_fn)]

use std::arch::x86_64::{
    __m256d, __m256i, _mm256_broadcast_sd, _mm256_cmpeq_epi64, _mm256_or_pd, _mm256_or_si256,
    _mm256_testz_pd, _mm256_testz_ps, _mm256_testz_si256,
};
use std::hint::black_box;
use std::mem::transmute;

#[derive(Clone, PartialEq)]
struct Point(i32, i32);

#[target_feature(enable = "avx2")]
#[inline(never)]
fn contains_auto(input_raw: &[Point], needle: Point) -> bool {
    // Make our LANE_COUNT 4x the normal lane count (aiming for 128 bit vectors).
    // The compiler will nicely unroll it.
    const LANE_COUNT: usize = 4 * (128 / (size_of::<Point>() * 8));
    // SIMD
    let mut chunks = input_raw.chunks_exact(LANE_COUNT);
    for chunk in &mut chunks {
        if chunk.iter().fold(false, |acc, x| acc | (*x == needle)) {
            return true;
        }
    }
    // false
    // Scalar remainder
    chunks.remainder().iter().any(|x| *x == needle)
}

#[target_feature(enable = "avx2")]
#[inline(never)]
unsafe fn contains_simd(input_raw: &[Point], needle_pos: Point) -> bool {
    let needle_f64: f64 = transmute(needle_pos);
    let needle: __m256d = _mm256_broadcast_sd(&needle_f64);
    let needle: __m256i = transmute(needle);

    let input: &[u64] = transmute(input_raw);

    const U64_IN_M256: usize = 4;
    let (pre, register, post) = input.as_simd::<U64_IN_M256>();
    let (chunks, chunk_remainder) = register.as_chunks::<4>();
    for chunk in chunks {
        let c1 = _mm256_cmpeq_epi64(__m256i::from(chunk[0]), needle);
        let c2 = _mm256_cmpeq_epi64(__m256i::from(chunk[1]), needle);
        let c3 = _mm256_cmpeq_epi64(__m256i::from(chunk[2]), needle);
        let c4 = _mm256_cmpeq_epi64(__m256i::from(chunk[3]), needle);

        let reduce1 = _mm256_or_si256(c1, c2);
        let reduce2 = _mm256_or_si256(c3, c4);
        let reduce_final = _mm256_or_si256(reduce1, reduce2);

        let zero_flag = _mm256_testz_si256(reduce_final, reduce_final);
        if zero_flag == 0 {
            return true;
        }
    }
    // todo: chunk_remainder, pre, post remainder handling
    false
}

fn main() {
    let value_init = Point(black_box(5), black_box(8));
    let value_not_exists = Point(black_box(5), black_box(253));

    let data = vec![value_init.clone(); 9999];

    let found = unsafe { contains_simd(&data, value_not_exists) };
    // let found = unsafe { contains_auto(&data, value_not_exists) };
    println!("found {found}")

    // let needle = Point(black_box(5), black_box(8));
    // println!("{:064b}", needle.0);
    // println!("{:064b}", needle.1);
    // println!("{:064b}", needle.as_u64());
}
