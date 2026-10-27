//! Raster-management CLI integration, including preservation on invalid requests.
use std::path::Path;
use std::process::{Command, Output};

fn run(args: &[&str], directory: &Path) -> Output {
    Command::new(env!("CARGO_BIN_EXE_terrio")).args(args).current_dir(directory).output().unwrap()
}

fn ok(args: &[&str], directory: &Path) {
    let output = run(args, directory);
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
}

#[test]
fn invalid_overwrite_preserves_existing_raster() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("existing.tif");
    ok(&["create-raster", "existing.tif", "--size", "8,8", "--burn", "7"], directory.path());
    let original = std::fs::read(&path).unwrap();
    for args in [
        vec!["create-raster", "existing.tif", "--overwrite", "--like", "missing.tif"],
        vec!["create-raster", "existing.tif", "--overwrite", "--size", "0,0"],
        vec!["create-raster", "existing.tif", "--overwrite", "--size", "8,8", "--burn", "1", "--burn", "2"],
    ] {
        assert_eq!(run(&args, directory.path()).status.code(), Some(1));
        assert_eq!(std::fs::read(&path).unwrap(), original);
    }
    ok(&["create-raster", "existing.tif", "--overwrite", "--like", "existing.tif", "--burn", "9"], directory.path());
    let result = terrio::open_raster(path.to_str().unwrap()).unwrap();
    assert!(result.raster.data.to_f64_vec().iter().all(|v| *v == 9.));
}

#[test]
fn invalid_overview_and_move_options_are_rejected() {
    let directory = tempfile::tempdir().unwrap();
    ok(&["create-raster", "input.tif", "--size", "32,32", "--burn", "7", "--crs", "EPSG:3857"], directory.path());
    ok(&["overviews", "input.tif", "2", "4"], directory.path());
    for level in ["0", "1"] {
        assert_eq!(run(&["overviews", "input.tif", "--refresh", "--levels", level], directory.path()).status.code(), Some(1));
    }
    for option in ["--stats", "--hist", "--unset-nodata", "--unset-all-metadata"] {
        assert_eq!(run(&["edit-raster", "input.tif", "--t-srs", "EPSG:4326", option], directory.path()).status.code(), Some(2));
    }
    assert_eq!(run(&["create-raster", "bad.png", "--like", "input.tif", "--copy-overviews"], directory.path()).status.code(), Some(1));
    assert!(!directory.path().join("bad.png").exists());
}
