#![feature(portable_simd)]
#![feature(slice_as_chunks)]
// too noisy
#![allow(unsafe_op_in_unsafe_fn)]

use std::arch::x86_64::{
    __m256d, __m256i, _mm256_broadcast_sd, _mm256_cmpeq_epi64, _mm256_or_si256, _mm256_testz_si256,
};
use std::hint::black_box;
use std::intrinsics::transmute;
use std::simd::cmp::SimdPartialEq;
use std::simd::{Mask, Simd};

pub fn inner_main() {
    let data = new_data();

    let test = value_init();
    // let test = value_not_exists();

    let found = unsafe {
        match 1 {
            1 => contains_std(&data, test),
            2 => contains_auto(&data, test),
            3 => contains_portable(&data, test),
            4 => contains_simd(&data, test),
            5 => contains_simd_unrolled(&data, test),
            _ => unimplemented!(),
        }
    };
    println!("found {found}")
}

#[derive(Clone, PartialEq, Debug)]
pub struct Point(pub i32, pub i32);

/// The Standard Library's auto-vectorized contains
#[inline(never)]
pub fn contains_std<T: PartialEq>(input_raw: &[T], needle_pos: T) -> bool {
    input_raw.contains(&needle_pos)
}

/// The Standard Library's auto-vectorized contains but for Structs
#[target_feature(enable = "avx2")]
#[inline(never)]
pub fn contains_auto(input_raw: &[Point], needle_pos: Point) -> bool {
    // Make our LANE_COUNT 4x the normal lane count (aiming for 128 bit vectors).
    // The compiler will nicely unroll it.
    const LANE_COUNT: usize = 8 * (128 / (size_of::<Point>() * 8));
    // SIMD
    let mut chunks = input_raw.chunks_exact(LANE_COUNT);
    for chunk in &mut chunks {
        if chunk.iter().fold(false, |acc, x| acc | (*x == needle_pos)) {
            return true;
        }
    }
    // TODO: remainder handling
    //chunks.remainder().iter().any(|x| *x == needle)
    false
}

/// Manually optimized Portable SIMD
#[target_feature(enable = "avx2")]
#[inline(never)]
pub unsafe fn contains_portable(input_raw: &[Point], needle_pos: Point) -> bool {
    // const LANES: usize = 16;
    const LANES: usize = 8;
    let all_not_equal: Mask<i64, LANES> = Mask::splat(false);

    let needle_64: u64 = transmute(needle_pos);
    let needle: Simd<u64, LANES> = Simd::splat(needle_64);
    let input: &[u64] = transmute(input_raw);

    let (pre, registers, post) = input.as_simd::<LANES>();
    for register in registers {
        let cmp = register.simd_eq(needle);
        if cmp != all_not_equal {
            return true;
        }
    }
    // todo: pre, post remainder handling
    false
}

/// Manually optimized raw intrinsics
#[target_feature(enable = "avx2")]
#[inline(never)]
pub unsafe fn contains_simd(input_raw: &[Point], needle_pos: Point) -> bool {
    let needle_64: f64 = transmute(needle_pos);
    let needle: __m256d = _mm256_broadcast_sd(&needle_64);
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

/// Manually optimized raw intrinsics unrolled 4 times for 16x4=64 wide chunk size
#[target_feature(enable = "avx2")]
#[inline(never)]
pub unsafe fn contains_simd_unrolled(input_raw: &[Point], needle_pos: Point) -> bool {
    let needle_f64: f64 = transmute(needle_pos);
    let needle: __m256d = _mm256_broadcast_sd(&needle_f64);
    let needle: __m256i = transmute(needle);

    let input: &[u64] = transmute(input_raw);

    const U64_IN_M256: usize = 4;
    let (pre, register, post) = input.as_simd::<U64_IN_M256>();
    let (chunks, chunk_remainder) = register.as_chunks::<16>();
    for chunk in chunks {
        let u1 = {
            const OFFSET: usize = 0 * 4;
            let c1 = _mm256_cmpeq_epi64(__m256i::from(chunk[OFFSET + 0]), needle);
            let c2 = _mm256_cmpeq_epi64(__m256i::from(chunk[OFFSET + 1]), needle);
            let c3 = _mm256_cmpeq_epi64(__m256i::from(chunk[OFFSET + 2]), needle);
            let c4 = _mm256_cmpeq_epi64(__m256i::from(chunk[OFFSET + 3]), needle);

            let reduce1 = _mm256_or_si256(c1, c2);
            let reduce2 = _mm256_or_si256(c3, c4);
            let reduce_final = _mm256_or_si256(reduce1, reduce2);
            reduce_final
        };

        let u2 = {
            const OFFSET: usize = 1 * 4;
            let c1 = _mm256_cmpeq_epi64(__m256i::from(chunk[OFFSET + 0]), needle);
            let c2 = _mm256_cmpeq_epi64(__m256i::from(chunk[OFFSET + 1]), needle);
            let c3 = _mm256_cmpeq_epi64(__m256i::from(chunk[OFFSET + 2]), needle);
            let c4 = _mm256_cmpeq_epi64(__m256i::from(chunk[OFFSET + 3]), needle);

            let reduce1 = _mm256_or_si256(c1, c2);
            let reduce2 = _mm256_or_si256(c3, c4);
            let reduce_final = _mm256_or_si256(reduce1, reduce2);
            reduce_final
        };

        // optionally comment u3 and u4 out for 8x wide registers
        let u3 = {
            const OFFSET: usize = 2 * 4;
            let c1 = _mm256_cmpeq_epi64(__m256i::from(chunk[OFFSET + 0]), needle);
            let c2 = _mm256_cmpeq_epi64(__m256i::from(chunk[OFFSET + 1]), needle);
            let c3 = _mm256_cmpeq_epi64(__m256i::from(chunk[OFFSET + 2]), needle);
            let c4 = _mm256_cmpeq_epi64(__m256i::from(chunk[OFFSET + 3]), needle);

            let reduce1 = _mm256_or_si256(c1, c2);
            let reduce2 = _mm256_or_si256(c3, c4);
            let reduce_final = _mm256_or_si256(reduce1, reduce2);
            reduce_final
        };

        let u4 = {
            const OFFSET: usize = 3 * 4;
            let c1 = _mm256_cmpeq_epi64(__m256i::from(chunk[OFFSET + 0]), needle);
            let c2 = _mm256_cmpeq_epi64(__m256i::from(chunk[OFFSET + 1]), needle);
            let c3 = _mm256_cmpeq_epi64(__m256i::from(chunk[OFFSET + 2]), needle);
            let c4 = _mm256_cmpeq_epi64(__m256i::from(chunk[OFFSET + 3]), needle);

            let reduce1 = _mm256_or_si256(c1, c2);
            let reduce2 = _mm256_or_si256(c3, c4);
            let reduce_final = _mm256_or_si256(reduce1, reduce2);
            reduce_final
        };

        let reduce1 = _mm256_or_si256(u1, u2);
        let reduce2 = _mm256_or_si256(u3, u4);
        let reduce_final = _mm256_or_si256(reduce1, reduce2);

        let zero_flag = _mm256_testz_si256(reduce_final, reduce_final);
        // let zero_flag = _mm256_testz_si256(reduce1, reduce1);
        if zero_flag == 0 {
            return true;
        }
    }
    // todo: chunk_remainder, pre, post remainder handling
    false
}

pub fn new_data() -> Vec<Point> {
    const SIZE: usize = 16 * 4 * 125;
    let mut data = Vec::new();
    for i in 0..SIZE {
        let i = i as i32;
        data.push(Point(i, i * 2));
    }
    data
}

/// black_box to avoid optimizing away to const tests
pub fn value_init() -> Point {
    Point(black_box(5), black_box(10))
}

/// black_box to avoid optimizing away to const tests
pub fn value_not_exists() -> Point {
    Point(black_box(5), black_box(253))
}

#[cfg(test)]
mod test {
    use crate::{
        Point, contains_auto, contains_portable, contains_simd, contains_simd_unrolled,
        contains_std, new_data, value_init, value_not_exists,
    };
    use std::mem::transmute;
    use std::simd::Simd;

    #[test]
    fn test_found() {
        let data = new_data();
        confirm_sanity(&data, value_init(), true);
    }

    #[test]
    fn test_not_found() {
        let data = new_data();
        confirm_sanity(&data, value_not_exists(), false);
    }

    #[test]
    fn test_found_all() {
        let data = new_data();
        for entry in testable_range(&data) {
            confirm_sanity(&data, entry.clone(), true);
        }
    }

    fn confirm_sanity(input: &[Point], value: Point, expected: bool) {
        unsafe {
            assert_eq!(
                contains_std(input, value.clone()),
                expected,
                "contains_std {:?}",
                value
            );
            assert_eq!(
                contains_auto(input, value.clone()),
                expected,
                "contains_auto {:?}",
                value
            );
            assert_eq!(
                contains_portable(input, value.clone()),
                expected,
                "contains_portable {:?}",
                value
            );
            assert_eq!(
                contains_simd(input, value.clone()),
                expected,
                "contains_simd {:?}",
                value
            );
            assert_eq!(
                contains_simd_unrolled(input, value.clone()),
                expected,
                "contains_simd_unrolled {:?}",
                value
            );
        }
    }

    fn testable_range(data_points: &[Point]) -> &[Point] {
        const U64_IN_M256: usize = 4;
        let data: &[u64] = unsafe { transmute(data_points) };
        let (pre, simd, post) = data.as_simd::<U64_IN_M256>();

        // assert_eq!(pre.len(), 2);
        // assert_eq!(post.len(), 2);
        let testable_range = &data_points[pre.len()..(data_points.len() - post.len() - 100)];
        testable_range
    }
}
