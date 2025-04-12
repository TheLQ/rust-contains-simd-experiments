set -x
cargo clean # deps has many .s files
RUSTFLAGS="--emit asm" cargo build --release
