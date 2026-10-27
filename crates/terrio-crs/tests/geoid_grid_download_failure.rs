//! A geoid grid that counts as available (the network is on) but cannot be downloaded: heights
//! are kept unchanged and the warning is still raised. Its own test binary: the grid transport
//! and network setting are process-wide.
#![cfg(all(feature = "grid-transport", not(target_family = "wasm")))]

use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use terrio_crs::{
    SpatialRef, TransformOptions, Transformer,
    network::{self, GridStream, GridTransport},
};

struct Unreachable(Arc<AtomicUsize>);
impl GridTransport for Unreachable {
    fn open(&self, _: &str) -> std::io::Result<Box<dyn GridStream>> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Err(std::io::ErrorKind::NotFound.into())
    }
}

#[test]
fn a_geoid_grid_that_fails_to_download_is_reported() {
    terrio_crs::set_grid_search_paths(Some(Vec::new()));
    terrio_crs::set_network_enabled(Some(true));
    let attempts = Arc::new(AtomicUsize::new(0));
    network::set_grid_transport(Arc::new(Unreachable(attempts.clone())));
    let _ = terrio_crs::take_warnings();
    let t = Transformer::with_options(
        &SpatialRef::parse("EPSG:7843").unwrap(),
        &SpatialRef::parse("EPSG:7844+5711").unwrap(),
        &TransformOptions::default(),
    )
    .unwrap();
    assert_eq!(t.transform_3d(144.9691, -37.8177, 40.0).unwrap().2, 40.0);
    // The download was attempted and failed, rather than skipped before reaching the transport
    assert!(attempts.load(Ordering::SeqCst) > 0);
    t.flush_warnings();
    assert!(terrio_crs::take_warnings().iter().any(|w| w.contains("geoid model did not apply")));
}
