use crate::matmul::test_matmul_strategy;
use cubecl::{Runtime, frontend::CubePrimitive, ir::AddressType, zspace::shape};
use cubek_matmul::{launch::Strategy, routines::BlueprintStrategy};

use cubek_matmul::{
    definition::MatmulGlobalElems,
    definition::{MatmulElems, MatmulProblem},
    routines::mosaic::MosaicStrategy,
};
use cubek_std::MatrixLayout;

type TestRuntime = cubecl::TestRuntime;

/// Sandbox tests for the Mosaic family. CPU-only Row-Col matmul. Kept
/// minimal — a handful of shapes against the f32 path so we can iterate
/// on the tile API without churning a giant test surface.
struct MosaicTestCase {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub lhs_batch: usize,
    pub rhs_batch: usize,
    pub elems: MatmulGlobalElems,
    pub strategy: Strategy,
}

impl MosaicTestCase {
    fn to_problem(&self) -> MatmulProblem {
        MatmulProblem::from_parameters(
            self.m,
            self.n,
            self.k,
            shape![self.lhs_batch],
            shape![self.rhs_batch],
            MatrixLayout::RowMajor,
            MatrixLayout::ColMajor,
            MatrixLayout::RowMajor,
            None,
            None,
            self.elems.clone(),
            AddressType::U32,
        )
    }

    pub(crate) fn test(self) {
        let client = TestRuntime::client(&Default::default());
        let problem = self.to_problem();
        test_matmul_strategy(client, problem, self.strategy);
    }
}

fn mosaic() -> Strategy {
    Strategy::Mosaic(BlueprintStrategy::Inferred(MosaicStrategy {
        target_num_planes: None,
    }))
}

fn elems_f32() -> MatmulGlobalElems {
    MatmulElems::from_single_dtype(f32::as_type_native_unchecked()).as_global_elems()
}

#[test]
pub fn very_small_square() {
    MosaicTestCase {
        m: 8,
        n: 8,
        k: 128,
        lhs_batch: 1,
        rhs_batch: 1,
        elems: elems_f32(),
        strategy: mosaic(),
    }
    .test();
}

#[test]
pub fn small_square() {
    MosaicTestCase {
        m: 32,
        n: 32,
        k: 256,
        lhs_batch: 1,
        rhs_batch: 1,
        elems: elems_f32(),
        strategy: mosaic(),
    }
    .test();
}

#[test]
pub fn rectangular() {
    MosaicTestCase {
        m: 32,
        n: 64,
        k: 128,
        lhs_batch: 1,
        rhs_batch: 1,
        elems: elems_f32(),
        strategy: mosaic(),
    }
    .test();
}

#[test]
pub fn batched() {
    MosaicTestCase {
        m: 16,
        n: 16,
        k: 64,
        lhs_batch: 3,
        rhs_batch: 3,
        elems: elems_f32(),
        strategy: mosaic(),
    }
    .test();
}
