use cubecl::prelude::{LaunchError, StorageType};

#[derive(Debug, thiserror::Error)]
pub enum FftError {
    #[error("unsupported FFT storage dtype {actual:?}; expected F32")]
    UnsupportedDtype { actual: StorageType },
    #[error("shape rank {shape_rank} differs from stride rank {stride_rank}")]
    RankMismatch {
        shape_rank: usize,
        stride_rank: usize,
    },
    #[error("FFT axis {dim} is out of bounds for rank {rank}")]
    AxisOutOfBounds { dim: usize, rank: usize },
    #[error("FFT length must be a power of two and at least 2, got {n_fft}")]
    InvalidFftLength { n_fft: usize },
    #[error("{name}={value} is outside {min}..={max}")]
    InvalidLength {
        name: &'static str,
        value: usize,
        min: usize,
        max: usize,
    },
    #[error("complex buffer needs {required} scalar elements but only {available} are available")]
    InsufficientBuffer { required: usize, available: usize },
    #[error("complex buffer byte offset {offset} is not aligned to scalar size {scalar_size}")]
    MisalignedBuffer { offset: u64, scalar_size: usize },
    #[error("complex scalar stride at axis {axis} overflowed")]
    StrideOverflow { axis: usize },
    #[error("complex buffer extent overflowed")]
    SizeOverflow,
    #[error("{name} shape {actual:?} does not match expected shape {expected:?}")]
    ShapeMismatch {
        name: &'static str,
        actual: Vec<usize>,
        expected: Vec<usize>,
    },
    #[error("input and output allocations overlap")]
    OverlappingBindings,
    #[error(transparent)]
    Launch(#[from] LaunchError),
}
