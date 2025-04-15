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
