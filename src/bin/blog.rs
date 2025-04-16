//! Extra assembly output used in blog post
//! `cargo asm --bin blog blog::unoptimized_struct_loop_test`
//! `cargo asm --bin blog blog::optimized_u64_unrolled_test`

use rust_contains_simd_experiments::{Point, contains_std, new_data, value_not_exists};
use std::hint::black_box;

fn main() {
    let result = unoptimized_struct_loop();
    println!("unoptimized_struct_loop {result}");

    let result = optimized_u64_unrolled();
    println!("optimized_u64_unrolled {result}");

    process();
}

fn unoptimized_struct_loop() -> bool {
    let data = new_data();
    unoptimized_struct_loop_test(&data, value_not_exists())
}

#[inline(never)]
fn unoptimized_struct_loop_test(input: &[Point], value: Point) -> bool {
    input.contains(&value)
}

fn optimized_u64_unrolled() -> bool {
    let data = vec![black_box(88u64); 9999];
    optimized_u64_unrolled_test(&data, black_box(55))
}

#[inline(never)]
fn optimized_u64_unrolled_test(input: &[u64], value: u64) -> bool {
    input.contains(&value)
}

#[derive(Clone, PartialEq)]
struct HugeStruct(u32, u32, Vec<u32>, String);

/// The Standard Library's auto-vectorized contains but for Structs
#[inline(never)]
pub fn contains_auto_huge(input_raw: &[HugeStruct], needle_pos: HugeStruct) -> bool {
    // Make our LANE_COUNT 4x the normal lane count (aiming for 128 bit vectors).
    // The compiler will nicely unroll it.
    assert_ne!(size_of::<HugeStruct>(), 0, "size_of");
    const LANE_COUNT: usize = 8 * (128 / (size_of::<HugeStruct>() * 8));
    println!("{}", 8 * (128 / (size_of::<HugeStruct>() * 8)));
    println!("{}", (128 / (size_of::<HugeStruct>() * 8)));
    println!("{}", (size_of::<HugeStruct>() * 8));
    assert_ne!(LANE_COUNT, 0, "lane");
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

fn process() {
    let vec = vec![HugeStruct(
        black_box(45),
        black_box(44),
        vec![black_box(1)],
        "asdasd".into(),
    )];
    let res = contains_auto_huge(
        &vec,
        black_box(HugeStruct(
            black_box(45),
            black_box(44),
            vec![black_box(1)],
            "asdasd".into(),
        )),
    );
    println!("huge {res}")
}
