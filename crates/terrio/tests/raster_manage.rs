//! Managing rasters in place: overviews (`gdaladdo`), with the safety of the updates (a failure
//! leaves the file as it was; read-only, remote and COG inputs are refused).

use terrio::ops::Resampling;
use terrio::ops::overview::{
    DeleteOptions, OverviewOptions, RefreshOptions, add_overviews, add_overviews_to_bytes, delete_overviews, refresh_overviews,
};
use terrio::{DataType, RasterDataset, RasterWriteOptions, Window, WindowOptions};

/// A UInt16 raster of `cols` x `rows` with nodata 0 at every seventh pixel, georeferenced.
fn sample(cols: usize, rows: usize) -> RasterDataset {
    let cfg = terrio::RasterConfig { cols, rows, bands: 1, data_type: DataType::U16, nodata: 0.0, ..Default::default() };
    let values: Vec<f64> = (0..cols * rows).map(|i| if i % 7 == 3 { 0.0 } else { (i * 37 % 1000 + 1) as f64 }).collect();
    let mut ds = RasterDataset::new(terrio::Raster::from_data(cfg, values).unwrap(), true);
    ds.raster.crs.epsg = Some(32613);
    ds.set_geo_transform([500000.0, 30.0, 0.0, 4100000.0, 0.0, -30.0]);
    ds
}

fn write(dir: &std::path::Path, name: &str, ds: &RasterDataset, options: &RasterWriteOptions) -> String {
    let path = dir.join(name).to_string_lossy().into_owned();
    terrio::write_raster_dataset(ds, &path, options).unwrap();
    path
}

fn tiled() -> RasterWriteOptions {
    RasterWriteOptions { compression: Some("DEFLATE".into()), tile_size: Some(64), ..Default::default() }
}

fn level(path: &str, overview: Option<usize>) -> Vec<f64> {
    let mut r = terrio::open_raster_lazy(path).unwrap();
    let (w, h) = overview.map_or((r.width(), r.height()), |i| r.overviews()[i]);
    r.read_window(Window::new(0, 0, w, h), &WindowOptions { bands: vec![], overview }).unwrap().raster.data.to_f64_vec()
}

#[test]
fn internal_overviews_average_valid_pixels_and_keep_the_image() {
    let dir = tempfile::tempdir().unwrap();
    let ds = sample(256, 192);
    let path = write(dir.path(), "a.tif", &ds, &tiled());
    let before = level(&path, None);
    let o = OverviewOptions { levels: vec![2, 4], resampling: Resampling::Average, ..Default::default() };
    assert_eq!(add_overviews(&path, &o).unwrap(), vec![(128, 96), (64, 48)]);
    let mut r = terrio::open_raster_lazy(&path).unwrap();
    assert_eq!(r.overviews(), vec![(128, 96), (64, 48)]);
    assert_eq!(r.nodata(), Some(0.0));
    assert_eq!(r.crs().epsg, Some(32613));
    assert_eq!(level(&path, None), before);
    // Factor 2: the rounded average of the valid pixels of each 2x2 block, as GDAL computes it.
    let first = level(&path, Some(0));
    for (y, x) in [(0, 0), (10, 17), (95, 127)] {
        let block: Vec<f64> =
            [(0, 0), (0, 1), (1, 0), (1, 1)].iter().map(|(dy, dx)| before[(2 * y + dy) * 256 + 2 * x + dx]).filter(|v| *v != 0.0).collect();
        let expected = (block.iter().sum::<f64>() / block.len() as f64 + 0.5).floor();
        assert_eq!(first[y * 128 + x], expected, "({x}, {y})");
    }
    // The resampling is recorded for refreshes, as GDAL records it.
    let ovr = r.read_window(Window::new(0, 0, 4, 4), &WindowOptions { bands: vec![], overview: Some(1) }).unwrap();
    assert_eq!(ovr.raster.cols, 4);
    // Refreshing whole overviews reproduces them.
    let second = level(&path, Some(1));
    assert_eq!(refresh_overviews(&path, &RefreshOptions::default()).unwrap(), vec![(128, 96), (64, 48)]);
    assert_eq!(level(&path, Some(0)), first);
    assert_eq!(level(&path, Some(1)), second);
    // Deleting them leaves the image.
    delete_overviews(&path, &DeleteOptions::default()).unwrap();
    assert!(terrio::open_raster_lazy(&path).unwrap().overviews().is_empty());
    assert_eq!(level(&path, None), before);
}

#[test]
fn external_overviews_are_read_from_the_ovr_sidecar() {
    let dir = tempfile::tempdir().unwrap();
    let ds = sample(200, 150);
    let a = write(dir.path(), "a.tif", &ds, &tiled());
    let b = write(dir.path(), "b.tif", &ds, &tiled());
    let o = OverviewOptions { levels: vec![2, 4, 8], resampling: Resampling::Mode, ..Default::default() };
    add_overviews(&a, &o).unwrap();
    add_overviews(&b, &OverviewOptions { external: true, ..o.clone() }).unwrap();
    assert!(std::path::Path::new(&format!("{b}.ovr")).is_file());
    let r = terrio::open_raster_lazy(&b).unwrap();
    assert_eq!(r.overviews(), vec![(100, 75), (50, 38), (25, 19)]);
    for i in 0..3 {
        assert_eq!(level(&a, Some(i)), level(&b, Some(i)), "overview {i}");
    }
    // External overviews cannot be added next to internal ones.
    assert!(add_overviews(&a, &OverviewOptions { external: true, ..o.clone() }).is_err());
    delete_overviews(&b, &DeleteOptions { external: true, ..Default::default() }).unwrap();
    assert!(!std::path::Path::new(&format!("{b}.ovr")).exists());
    assert!(terrio::open_raster_lazy(&b).unwrap().overviews().is_empty());
}

#[test]
fn automatic_levels_and_masks_follow_gdal() {
    assert_eq!(terrio::ops::overview::automatic_levels(2880, 1773, 256), vec![2, 4, 8, 16]);
    assert_eq!(terrio::ops::overview::automatic_levels(200, 100, 256), Vec::<usize>::new());
    let dir = tempfile::tempdir().unwrap();
    let mut ds = sample(128, 128);
    ds.has_nodata = false;
    ds.band_nodata = vec![None];
    // An internal mask hiding the left half of the image.
    let mask: Vec<u8> = (0..128 * 128).map(|i| if i % 128 < 64 { 0 } else { 255 }).collect();
    let options = RasterWriteOptions { mask: Some(mask), ..tiled() };
    let path = write(dir.path(), "m.tif", &ds, &options);
    let o = OverviewOptions { min_size: 32, resampling: Resampling::Average, ..Default::default() };
    assert_eq!(add_overviews(&path, &o).unwrap(), vec![(64, 64), (32, 32)]);
    let mut r = terrio::open_raster_lazy(&path).unwrap();
    // The mask's overviews: valid where any source pixel is.
    let m = r.read_mask(Window::new(0, 0, 64, 64), Some(0)).unwrap().unwrap();
    assert_eq!(m[0], 0);
    assert_eq!(m[32], 255);
    // Masked pixels stay out of the averages.
    let full = level(&path, None);
    let ovr = level(&path, Some(0));
    let block = [full[64], full[65], full[128 + 64], full[128 + 65]];
    assert_eq!(ovr[32], (block.iter().sum::<f64>() / 4.0 + 0.5).floor());
}

#[test]
fn failed_or_refused_updates_leave_the_file_unchanged() {
    let dir = tempfile::tempdir().unwrap();
    let ds = sample(64, 64);
    let path = write(dir.path(), "a.tif", &ds, &tiled());
    let bytes = std::fs::read(&path).unwrap();
    // Methods GDAL's overviews do not offer here are refused before anything is written.
    let o = OverviewOptions { levels: vec![2], resampling: Resampling::Cubic, ..Default::default() };
    assert!(add_overviews(&path, &o).is_err());
    assert!(add_overviews(&path, &OverviewOptions { levels: vec![1], ..Default::default() }).is_err());
    assert_eq!(std::fs::read(&path).unwrap(), bytes);
    // Remote objects and archive members are not updated in place.
    assert!(add_overviews("https://example.com/a.tif", &OverviewOptions::default()).is_err());
    assert!(add_overviews("/vsizip/a.zip/a.tif", &OverviewOptions::default()).is_err());
    // A COG keeps its layout unless breaking it is allowed, as in GDAL.
    let cog = write(
        dir.path(),
        "c.tif",
        &ds,
        &RasterWriteOptions { driver: Some(terrio::formats::RasterDriver::Cog), tile_size: Some(16), ..Default::default() },
    );
    let cog_bytes = std::fs::read(&cog).unwrap();
    assert!(delete_overviews(&cog, &DeleteOptions::default()).is_err());
    assert_eq!(std::fs::read(&cog).unwrap(), cog_bytes);
    delete_overviews(&cog, &DeleteOptions { ignore_cog_layout_break: true, ..Default::default() }).unwrap();
    let edited = std::fs::read(&cog).unwrap();
    assert!(String::from_utf8_lossy(&edited[..400]).contains("KNOWN_INCOMPATIBLE_EDITION=YES"));
    assert!(terrio::open_raster_lazy(&cog).unwrap().overviews().is_empty());
    assert_eq!(level(&cog, None), level(&path, None));
    // Nothing is left behind next to the files.
    let names: Vec<String> =
        std::fs::read_dir(dir.path()).unwrap().map(|e| e.unwrap().file_name().to_string_lossy().into_owned()).collect();
    assert!(names.iter().all(|n| !n.contains(".terrio-")), "{names:?}");
}

#[cfg(unix)]
#[test]
fn read_only_files_are_refused() {
    let dir = tempfile::tempdir().unwrap();
    let path = write(dir.path(), "a.tif", &sample(64, 64), &tiled());
    let mut perms = std::fs::metadata(&path).unwrap().permissions();
    perms.set_readonly(true);
    std::fs::set_permissions(&path, perms).unwrap();
    let err = add_overviews(&path, &OverviewOptions { levels: vec![2], ..Default::default() }).unwrap_err();
    assert!(err.to_string().contains("read-only"), "{err}");
}

#[test]
fn overviews_of_tiffs_in_memory() {
    let dir = tempfile::tempdir().unwrap();
    let path = write(dir.path(), "a.tif", &sample(100, 80), &RasterWriteOptions::default());
    let bytes = std::fs::read(&path).unwrap();
    let o = OverviewOptions { levels: vec![2, 4], resampling: Resampling::Nearest, ..Default::default() };
    let out = add_overviews_to_bytes(&bytes, &o).unwrap();
    let mut r = terrio::LazyRaster::from_tiff_bytes(out.clone()).unwrap().unwrap();
    // A striped image gets 128-pixel tiled overviews, as GDAL makes them.
    assert_eq!(r.overviews(), vec![(50, 40), (25, 20)]);
    let o0 =
        r.read_window(Window::new(0, 0, 50, 40), &WindowOptions { bands: vec![], overview: Some(0) }).unwrap().raster.data.to_f64_vec();
    let full = level(&path, None);
    // Nearest: the pixel at 0.5 + 2i.
    assert_eq!(o0[3 * 50 + 7], full[(3 * 2) * 100 + 7 * 2]);
    let cleaned = terrio::ops::overview::delete_overviews_from_bytes(&out, &DeleteOptions::default()).unwrap();
    assert!(terrio::LazyRaster::from_tiff_bytes(cleaned).unwrap().unwrap().overviews().is_empty());
}

#[test]
fn edits_change_georeferencing_and_metadata_but_not_pixels() {
    use terrio::ops::edit::{EditOptions, edit_raster};
    let dir = tempfile::tempdir().unwrap();
    let path = write(dir.path(), "e.tif", &sample(120, 90), &tiled());
    let before = level(&path, None);
    let o = EditOptions {
        crs: Some(terrio::crs::SpatialRef::parse("EPSG:4326").unwrap()),
        bbox: Some([-106.0, 39.0, -105.0, 40.0]),
        nodata: Some(7.0),
        metadata: vec![("SOURCE".into(), "a & <b>".into())],
        stats: true,
        histogram: true,
        ..Default::default()
    };
    edit_raster(&path, &o).unwrap();
    let r = terrio::open_raster_lazy(&path).unwrap();
    assert_eq!(r.crs().epsg, Some(4326));
    assert_eq!(r.geo_transform(), [-106.0, 1.0 / 120.0, 0.0, 40.0, 0.0, -1.0 / 90.0]);
    assert_eq!(r.nodata(), Some(7.0));
    assert!(r.metadata().contains(&("SOURCE".into(), "a & <b>".into())));
    assert_eq!(level(&path, None), before);
    assert!(std::fs::read_to_string(format!("{path}.aux.xml")).unwrap().contains("<HistCounts>"));
    // Unsetting.
    let o = EditOptions { unset_crs: true, unset_nodata: true, unset_metadata: vec!["SOURCE".into()], ..Default::default() };
    edit_raster(&path, &o).unwrap();
    let r = terrio::open_raster_lazy(&path).unwrap();
    assert!(r.crs().is_unknown());
    assert_eq!(r.nodata(), None);
    assert!(!r.metadata().iter().any(|(k, _)| k == "SOURCE"));
    // Contradictory edits are refused before the file is touched.
    let bytes = std::fs::read(&path).unwrap();
    assert!(edit_raster(&path, &EditOptions { nodata: Some(1.0), unset_nodata: true, ..Default::default() }).is_err());
    assert_eq!(std::fs::read(&path).unwrap(), bytes);
}

#[test]
fn moves_follow_gdalmove() {
    use terrio::ops::edit::{MoveOptions, move_raster};
    let dir = tempfile::tempdir().unwrap();
    let path = write(dir.path(), "m.tif", &sample(100, 100), &tiled());
    let t_srs = terrio::crs::SpatialRef::parse("EPSG:26913").unwrap();
    // Without a threshold, only a report.
    let r = move_raster(&path, &MoveOptions { t_srs: t_srs.clone(), s_srs: None, error_threshold: None, ignore_cog_layout_break: false })
        .unwrap();
    assert!(!r.updated && r.max_error < 0.01, "{r:?}");
    assert_eq!(terrio::open_raster_lazy(&path).unwrap().crs().epsg, Some(32613));
    let r = move_raster(&path, &MoveOptions { t_srs, s_srs: None, error_threshold: Some(0.5), ignore_cog_layout_break: false }).unwrap();
    assert!(r.updated);
    let lazy = terrio::open_raster_lazy(&path).unwrap();
    assert_eq!(lazy.crs().epsg, Some(26913));
    assert_eq!(lazy.geo_transform(), r.geo_transform);
}
