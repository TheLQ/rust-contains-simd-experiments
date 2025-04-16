use criterion::{Criterion, black_box, criterion_group, criterion_main};
use rust_contains_simd_experiments::{
    Point, contains_auto, contains_simd, contains_simd_unrolled, contains_std,
};
use std::mem;

criterion_group!(benches, micro_bench);
criterion_main!(benches);

fn micro_bench(c: &mut Criterion) {
    let data = new_data();

    let mut group = c.benchmark_group("micro");
    group.bench_with_input("auto", &data, |b, my_data| {
        b.iter(|| test_auto(black_box(my_data)))
    });
    group.bench_with_input("std", &data, |b, my_data| {
        b.iter(|| test_std(black_box(my_data)))
    });
    group.bench_with_input("simd", &data, |b, my_data| {
        b.iter(|| test_simd(black_box(my_data)))
    });
    group.bench_with_input("simd-unrolled", &data, |b, my_data| {
        b.iter(|| test_simd_unrolled(black_box(my_data)))
    });
    group.finish()
}

fn test_auto(input: &[Point]) -> u32 {
    let mut result = 0;
    for needle in NEEDLES {
        unsafe {
            if contains_auto(input, needle) {
                result += 1;
            }
        }
    }
    result
}

fn test_std(input: &[Point]) -> u32 {
    let mut result = 0;
    for needle in NEEDLES {
        unsafe {
            if contains_std(input, needle) {
                result += 1;
            }
        }
    }
    result
}

fn test_simd(input: &[Point]) -> u32 {
    let mut result = 0;
    for needle in NEEDLES {
        unsafe {
            if contains_simd(input, needle) {
                result += 1;
            }
        }
    }
    result
}

fn test_simd_unrolled(input: &[Point]) -> u32 {
    let mut result = 0;
    for needle in NEEDLES {
        unsafe {
            if contains_simd_unrolled(input, needle) {
                result += 1;
            }
        }
    }
    result
}

const DATA_SIZE: usize = 4/*Points per m256*/ * 4/*simd chunks*/ * 4/*ultra chunks*/ * 4000; // 512k

const NEEDLES_SIZE: usize = 10_000;
const NEEDLES: [Point; NEEDLES_SIZE] = new_needles();

const GOLDEN_NEEDLE: Point = Point(389_000, 389_000);

const fn new_needles() -> [Point; NEEDLES_SIZE] {
    let mut res: [Point; NEEDLES_SIZE] = unsafe { mem::zeroed() };

    let mut index = 0;
    while index < NEEDLES_SIZE {
        res[index] = Point(index as i32, index as i32);
        index += 1;
    }

    res
}

fn new_data() -> Vec<Point> {
    vec![black_box(GOLDEN_NEEDLE); DATA_SIZE]
}
