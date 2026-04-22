use cubecl::zspace::Strides;

#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
pub enum VectorizationMode {
    Parallel,
    Perpendicular,
}

pub(crate) fn output_vectorization_axis(
    input_strides: &Strides,
    vectorization_mode: VectorizationMode,
) -> usize {
    if input_strides.len() < 2 {
        // The axis of vectorization for input and output are both 0
        return 0;
    }

    let mut min1 = (usize::MAX, 0); // (value, index)
    let mut min2 = (usize::MAX, 0);

    for (i, &s) in input_strides.iter().enumerate() {
        if s < min1.0 {
            min2 = min1;
            min1 = (s, i);
        } else if s < min2.0 {
            min2 = (s, i);
        }
    }

    match vectorization_mode {
        // Parallel: reduce axis is the contiguous (smallest-stride) axis.
        // The output's vectorization axis is the next-smallest-stride non-reduce axis.
        VectorizationMode::Parallel => min2.1,
        // Perpendicular: reduce axis is a non-contiguous axis. The contiguous
        // (smallest-stride) axis is the SIMD vectorization axis in the output.
        VectorizationMode::Perpendicular => min1.1,
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
/// How bound checks is handled for inner reductions.
pub enum BoundChecks {
    /// No bound check is necessary.
    None,
    /// Using a mask is enough for bound checks.
    /// This will still read the memory in an out-of-bound location,
    /// but will replace the value by the null value.
    Mask,
    /// Branching is necessary for bound checks.
    ///
    /// Probably the right setting when performing fuse on read.
    Branch,
}

impl BoundChecks {
    pub fn idle(self) -> Self {
        Self::Mask
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
pub enum IdleMode {
    None,
    Mask,
    Terminate,
}

impl IdleMode {
    /// Whether idle is activated.
    pub fn is_enabled(&self) -> bool {
        !matches!(self, Self::None)
    }
}
