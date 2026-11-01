//! Quantization schemes and kernels.
//!
//! The Rust standard library is required, even with `--no-default-features`.
//! The pinned CubeCL 0.10 runtime depends on `async-channel` with its `std`
//! feature enabled, which pulls in `parking` through `event-listener-strategy`
//! and `event-listener`. Disabling this crate's default features disables the
//! kernels, but does not make its dependencies compatible with `no_std` targets.

extern crate alloc;

#[cfg(feature = "kernels")]
pub mod dequantize;

#[cfg(feature = "kernels")]
pub mod quantize;

#[cfg(feature = "kernels")]
pub mod layout;

pub use cubecl_common::quant::scheme;

#[cfg(feature = "kernels")]
pub(crate) mod utils {
    use crate::scheme::{QuantLevel, QuantScheme};

    pub(crate) fn check_block_size_compat(scheme: &QuantScheme, div: usize) {
        // Validate block size compatibility
        if let QuantScheme {
            level: QuantLevel::Block(block_size),
            ..
        } = scheme
        {
            let block_size = *block_size.as_slice().last().unwrap() as usize;
            assert!(
                block_size.is_multiple_of(div),
                "Block size must be divisible by {div}, got block_size={block_size}"
            );
        }
    }
}
