use cubek_test_utils::CatalogEntry;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FftStrategy {
    Split,
    Interleaved,
}

impl FftStrategy {
    pub fn id(self) -> &'static str {
        match self {
            Self::Split => "default",
            Self::Interleaved => "interleaved",
        }
    }
}

pub fn strategies() -> Vec<CatalogEntry<FftStrategy>> {
    vec![
        CatalogEntry::new("default", "Default (split)", FftStrategy::Split),
        CatalogEntry::new("interleaved", "Interleaved C32", FftStrategy::Interleaved),
    ]
}
