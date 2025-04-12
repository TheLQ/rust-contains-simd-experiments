use criterion::{Criterion, black_box, criterion_group, criterion_main};
use docrust::{BPoint, b_contains_auto, b_contains_auto_forced, b_contains_simd};
use std::mem;

criterion_group!(benches, micro_bench);
criterion_main!(benches);

fn micro_bench(c: &mut Criterion) {
    let data = new_data();

    let mut group = c.benchmark_group("micro");
    group.bench_with_input("auto", &data, |b, my_data| {
        b.iter(|| test_auto(black_box(my_data)))
    });
    group.bench_with_input("forced", &data, |b, my_data| {
        b.iter(|| test_auto_forced(black_box(my_data)))
    });
    group.bench_with_input("simd", &data, |b, my_data| {
        b.iter(|| test_simd(black_box(my_data)))
    });
    group.finish()
}

fn test_auto(input: &[BPoint]) -> u32 {
    let mut result = 0;
    for needle in NEEDLES {
        unsafe {
            if b_contains_auto(input, needle) {
                result += 1;
            }
        }
    }
    result
}

fn test_auto_forced(input: &[BPoint]) -> u32 {
    let mut result = 0;
    for needle in NEEDLES {
        unsafe {
            if b_contains_auto_forced(input, needle) {
                result += 1;
            }
        }
    }
    result
}

fn test_simd(input: &[BPoint]) -> u32 {
    let mut result = 0;
    for needle in NEEDLES {
        unsafe {
            if b_contains_simd(input, needle) {
                result += 1;
            }
        }
    }
    result
}

const DATA_SIZE: usize = 500_000;
const NEEDLES_SIZE: usize = 10_000;
const NEEDLES: [BPoint; NEEDLES_SIZE] = new_needles();

const fn new_needles() -> [BPoint; NEEDLES_SIZE] {
    let mut res: [BPoint; NEEDLES_SIZE] = unsafe { mem::zeroed() };

    let mut index = 0;
    while index < NEEDLES_SIZE {
        res[index] = BPoint(index as i32, index as i32);
        index += 1;
    }

    res
}

fn new_data() -> Vec<BPoint> {
    vec![BPoint(black_box(99), black_box(98)); DATA_SIZE]
}
