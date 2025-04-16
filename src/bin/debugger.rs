#![feature(portable_simd)]

use std::arch::x86_64::{
    __m256, __m256i, _mm256_packs_epi32, _mm256_permute4x64_epi64, _mm256_shuffle_ps,
};
use std::simd::Simd;

fn main() {
    unsafe { value() }
}

#[inline(never)]
unsafe fn value() {
    let shuffler = _mm256_shuffle_ps::<136>(
        __m256::from(Simd::from_array([
            201.0f32, 301.0, 202.0, 302.0, 203.0, 303.0, 204.0, 304.0,
        ])),
        __m256::from(Simd::from_array([
            801.0f32, 901.0, 802.0, 902.0, 803.0, 903.0, 804.0, 904.0,
        ])),
    );
    debug_f32(shuffler);
    let shuffler = _mm256_shuffle_ps::<221>(
        __m256::from(Simd::from_array([
            201.0f32, 301.0, 202.0, 302.0, 203.0, 303.0, 204.0, 304.0,
        ])),
        __m256::from(Simd::from_array([
            801.0f32, 901.0, 802.0, 902.0, 803.0, 903.0, 804.0, 904.0,
        ])),
    );
    debug_f32(shuffler);

    println!();

    let packed = _mm256_packs_epi32(
        __m256i::from(Simd::from_array([201i64, 202, 203, 204])),
        __m256i::from(Simd::from_array([801i64, 802, 803, 804])),
    );
    debug_i32(packed);
    let permute = _mm256_permute4x64_epi64::<216>(packed);
    debug_i32(permute)
}

fn debug_f32(val: __m256) {
    let debug: [f32; 8] = Simd::from(val).to_array();
    println!("{}", debug.map(|v| v.to_string()).join(","))
}

fn debug_i32(val: __m256i) {
    let debug: [i32; 8] = Simd::from(val).to_array();
    println!("{}", debug.map(|v| v.to_string()).join(","))
}
