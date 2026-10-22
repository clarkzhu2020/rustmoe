use rustmoe_io_engine::{StoreMode, WeightStore};
use rustmoe_model::ModelDescriptor;
use std::{path::PathBuf, sync::Arc};

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/tiny-ds41/tiny-ds41-00001-of-00001.gguf")
}

fn model() -> Arc<ModelDescriptor> {
    Arc::new(ModelDescriptor::load(fixture()).expect("load fixture"))
}

#[test]
fn reads_expert_bytes_at_computed_offsets() {
    let model = model();
    let store = WeightStore::open(model, StoreMode::Pread).expect("open store");
    let payload = store.read_expert_alloc(0, 0).expect("read expert");
    assert_eq!(payload.gate.len(), 64 * 32 * 4);
    assert_eq!(payload.down.len(), 32 * 64 * 4);
    assert_eq!(payload.up.len(), 64 * 32 * 4);
    assert!(payload.gate.iter().all(|b| *b == 0));
    assert_eq!(store.stats().reads, 3);
    assert_eq!(
        store.stats().bytes_read as usize,
        payload.gate.len() + payload.down.len() + payload.up.len()
    );
}

#[test]
fn mmap_mode_exposes_the_same_expert() {
    let model = model();
    let store = WeightStore::open(model, StoreMode::Mmap).expect("open mmap store");
    let payload = store.read_expert_alloc(0, 1).expect("read expert");
    assert_eq!(payload.expert, 1);
    assert_eq!(payload.gate.len(), 8192);
}

#[tokio::test]
async fn async_read_runs_on_blocking_pool() {
    let model = model();
    let store = Arc::new(WeightStore::open(model, StoreMode::Pread).expect("open store"));
    let payload = store.read_expert_alloc_async(0, 2).await.expect("async read");
    assert_eq!(payload.expert, 2);
    assert_eq!(payload.down.len(), 8192);
}
