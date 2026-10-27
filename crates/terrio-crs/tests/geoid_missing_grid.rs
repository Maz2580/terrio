//! Heights when a geoid grid is missing: kept unchanged by default, as PROJ keeps them, but with a
//! warning, as a missing horizontal grid has; refused under ONLY_BEST=YES. Its own test binary:
//! grid search paths and the warning sink are process-wide.
use terrio_crs::{SpatialRef, TransformOptions, Transformer};

fn no_grids() {
    terrio_crs::set_grid_search_paths(Some(Vec::new()));
    terrio_crs::set_network_enabled(Some(false));
}

fn transformer(src: &str, dst: &str, opts: &TransformOptions) -> Transformer {
    Transformer::with_options(&SpatialRef::parse(src).unwrap(), &SpatialRef::parse(dst).unwrap(), opts).unwrap()
}

fn warned(t: &Transformer) -> bool {
    t.flush_warnings();
    terrio_crs::take_warnings().iter().any(|w| w.contains("geoid model did not apply"))
}

#[test]
fn heights_kept_for_a_missing_geoid_grid_are_reported() {
    no_grids();
    let _ = terrio_crs::take_warnings();
    // GDA2020 ellipsoidal heights to AHD (AUSGeoid2020) in Melbourne: 35.26 m with the grid
    let to_ahd = transformer("EPSG:7843", "EPSG:7844+5711", &TransformOptions::default());
    assert_eq!(to_ahd.transform_3d(144.9691, -37.8177, 40.0).unwrap().2, 40.0);
    assert!(warned(&to_ahd));
    // And from AHD heights back to ellipsoidal ones
    let from_ahd = transformer("EPSG:7844+5711", "EPSG:7843", &TransformOptions::default());
    assert_eq!(from_ahd.transform_3d(144.9691, -37.8177, 35.26).unwrap().2, 35.26);
    assert!(warned(&from_ahd));
    // ONLY_BEST=YES still refuses the point rather than warn
    let strict = transformer("EPSG:7843", "EPSG:7844+5711", &TransformOptions { only_best: true, ..Default::default() });
    assert!(strict.transform_3d(144.9691, -37.8177, 40.0).is_err());
}

#[test]
fn heights_outside_every_geoid_model_are_not_reported() {
    no_grids();
    let _ = terrio_crs::take_warnings();
    // OSGM15 covers Great Britain: a point in the mid-Atlantic has no model whose grid could be missing
    let t = transformer("EPSG:4937", "EPSG:4258+5701", &TransformOptions::default());
    let _ = t.transform_3d(-30.0, 40.0, 100.0);
    assert!(!warned(&t));
}

#[test]
fn transformations_without_heights_are_not_reported() {
    no_grids();
    let _ = terrio_crs::take_warnings();
    let t = transformer("EPSG:7844", "EPSG:7855", &TransformOptions::default());
    assert!(t.transform(144.9691, -37.8177).is_ok());
    assert!(!warned(&t));
}
