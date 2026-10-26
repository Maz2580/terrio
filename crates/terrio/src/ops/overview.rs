//! Overviews: their selection for warping, following GDAL's `-ovr` choices, and their management
//! in GeoTIFFs as `gdaladdo` and `gdal raster overview add/delete/refresh` manage them
//! ([`add_overviews`], [`delete_overviews`], [`refresh_overviews`]).
//!
//! Overviews are added to a GeoTIFF in place (internal overviews) or written to its `.ovr`
//! sidecar (external overviews, which [`crate::LazyRaster`] reads as the raster's overviews when
//! it has no internal ones). As GDAL does, each overview is computed from the previous (larger)
//! one being built, the valid pixels (not nodata, not masked out by a per-dataset mask) only,
//! internal overviews take the full-resolution image's compression, predictor and tile size (128
//! pixel tiles for a striped image), and a raster with an internal mask gets mask overviews. The
//! file is changed through a copy renamed over it, so a failure leaves it as it was.
//!
//! ```no_run
//! use terrio::ops::Resampling;
//! use terrio::ops::overview::{OverviewOptions, RefreshOptions, add_overviews, refresh_overviews};
//!
//! // gdaladdo -r average dem.tif 2 4 8
//! let o = OverviewOptions { levels: vec![2, 4, 8], resampling: Resampling::Average, ..Default::default() };
//! add_overviews("dem.tif", &o)?;
//! // gdal raster overview refresh --bbox 500000,4100000,510000,4110000 dem.tif
//! let r = RefreshOptions { bbox: vec![[500000.0, 4100000.0, 510000.0, 4110000.0]], ..Default::default() };
//! refresh_overviews("dem.tif", &r)?;
//! # Ok::<(), terrio::Error>(())
//! ```

use crate::drivers::gtiff_update::{self, Appender, Directory, Entry, Tiff};
use crate::drivers::gtiff_write::{self as gw, CodecParams, Compression, Layout, OverviewResampling, Value};
use crate::ops::Resampling;
use crate::ops::overview_kernel as kernel;
use crate::{Error, LazyRaster, Result, Window, WindowOptions};
use std::io::{Read, Seek, Write};
use wbraster::DataType;
use wbraster::raster::RasterData;

/// Which stored source resolution a warp reads. Overview levels start at zero.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum OverviewSelection {
    /// Choose the finest stored resolution close to the output's sampling ratio.
    #[default]
    Auto,
    /// Choose `n` levels finer than the automatic choice (`AUTO-n`).
    AutoOffset(usize),
    /// Always read full resolution (`NONE`).
    None,
    /// Read this overview, or the last available one if the level is too large.
    Level(usize),
}

impl OverviewSelection {
    /// Parses `AUTO`, `AUTO-n`, `NONE`, or an overview number, case-insensitively.
    pub fn parse(value: &str) -> Result<Self> {
        let word = value.to_ascii_uppercase();
        match word.as_str() {
            "AUTO" => Ok(Self::Auto),
            "NONE" => Ok(Self::None),
            _ => {
                if let Some(offset) = word.strip_prefix("AUTO-") {
                    return offset.parse().map(Self::AutoOffset).map_err(|_| invalid(value));
                }
                word.parse().map(Self::Level).map_err(|_| invalid(value))
            }
        }
    }

    pub(crate) fn select(self, width: usize, overviews: &[(usize, usize)], target_ratio: f64) -> Option<usize> {
        match self {
            Self::None => None,
            Self::Level(index) => {
                if index >= overviews.len() {
                    crate::warnings::warn(format!("cannot get overview level {index}; using the coarsest available source resolution"));
                }
                (!overviews.is_empty()).then(|| index.min(overviews.len() - 1))
            }
            Self::Auto | Self::AutoOffset(_) => {
                if !target_ratio.is_finite() || target_ratio <= 1.0 {
                    return None;
                }
                let mut best = None;
                let mut best_ratio = 0.0;
                // GDAL includes full resolution in the search and uses 0.1 as
                // an absolute ratio tolerance, rather than rounding powers of two.
                for i in 0..=overviews.len() {
                    let ratio = if i == 0 { 1.0 } else { width as f64 / overviews[i - 1].0 as f64 };
                    if ratio >= target_ratio + 0.1 || ratio <= best_ratio {
                        continue;
                    }
                    best = i.checked_sub(1);
                    best_ratio = ratio;
                    if (target_ratio - ratio).abs() < 0.1 {
                        break;
                    }
                }
                let offset = match self {
                    Self::AutoOffset(n) => n,
                    _ => 0,
                };
                best.and_then(|index| index.checked_sub(offset))
            }
        }
    }
}

/// How [`add_overviews`] builds overviews (`gdaladdo`, `gdal raster overview add`).
#[derive(Debug, Clone, PartialEq)]
pub struct OverviewOptions {
    /// The decimation factors (`2 4 8`). Empty for GDAL's automatic levels: factors of 2, 4, 8...
    /// until the overview fits in `min_size` pixels.
    pub levels: Vec<usize>,
    /// For automatic levels, the largest width or height of the smallest overview (256).
    pub min_size: usize,
    /// The resampling: nearest (the default), average, min, max or mode.
    pub resampling: Resampling,
    /// Write the overviews to the `.ovr` sidecar instead of into the file.
    pub external: bool,
    /// The overviews' compression (GDAL's `COMPRESS_OVERVIEW`): by default the full-resolution
    /// image's for internal overviews, `NONE` for external ones, as in GDAL.
    pub compression: Option<String>,
    /// Update a Cloud Optimized GeoTIFF, breaking its layout (GDAL's `IGNORE_COG_LAYOUT_BREAK`
    /// open option); refused otherwise.
    pub ignore_cog_layout_break: bool,
}

impl Default for OverviewOptions {
    fn default() -> Self {
        Self {
            levels: Vec::new(),
            min_size: 256,
            resampling: Resampling::Nearest,
            external: false,
            compression: None,
            ignore_cog_layout_break: false,
        }
    }
}

/// How [`delete_overviews`] deletes overviews (`gdaladdo -clean`, `gdal raster overview delete`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DeleteOptions {
    /// Delete the `.ovr` sidecar instead of internal overviews.
    pub external: bool,
    /// Update a Cloud Optimized GeoTIFF, breaking its layout; refused otherwise.
    pub ignore_cog_layout_break: bool,
}

/// How [`refresh_overviews`] recomputes existing overviews (`gdal raster overview refresh`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RefreshOptions {
    /// The resampling. By default, the method each overview was built with (its `RESAMPLING`
    /// metadata item), else nearest.
    pub resampling: Option<Resampling>,
    /// Refresh only the overviews of these decimation factors; empty for all.
    pub levels: Vec<usize>,
    /// Refresh only the overview blocks over these extents `[min_x, min_y, max_x, max_y]`, in
    /// the raster's CRS (`--bbox`, or the extents of `--like` datasets: see [`dataset_bounds`]);
    /// empty for whole overviews.
    pub bbox: Vec<[f64; 4]>,
    /// Refresh the `.ovr` sidecar's overviews instead of internal ones.
    pub external: bool,
    /// Update a Cloud Optimized GeoTIFF, breaking its layout; refused otherwise.
    pub ignore_cog_layout_break: bool,
}

/// GDAL's automatic overview factors (`gdaladdo` without levels): 2, 4, 8... until an overview
/// fits in `min_size` x `min_size` pixels.
pub fn automatic_levels(width: usize, height: usize, min_size: usize) -> Vec<usize> {
    let min_size = min_size.max(1);
    let mut factor = 1usize;
    let mut out = Vec::new();
    while (width.div_ceil(factor) > min_size || height.div_ceil(factor) > min_size) && factor < width.max(height) {
        factor *= 2;
        out.push(factor);
    }
    out
}

/// GDAL's decimation factor of an overview (`GDALComputeOvFactor`), preferring powers of two.
pub fn overview_factor(overview: (usize, usize), full: (usize, usize)) -> usize {
    let pick = |o: usize, n: usize| {
        let v = (0.5 + n as f64 / o.max(1) as f64) as usize;
        let p2 = if v == 0 { 0 } else { 1usize << (usize::BITS - 1 - v.leading_zeros()) };
        for fact in 1..=2 {
            if p2 > 0 && n.div_ceil(fact * p2) == o {
                return fact * p2;
            }
        }
        v
    };
    if full.0 != 1 && full.0 >= full.1 / 2 { pick(overview.0, full.0) } else { pick(overview.1, full.1) }
}

/// Whether an existing overview is the one of decimation factor `factor`, as GDAL matches them
/// (`GDALComputeOvFactor` and `GDALOvLevelAdjust2`).
fn matches_factor(overview: (usize, usize), full: (usize, usize), factor: usize) -> bool {
    let (w, h) = full;
    let adjusted = if w >= h / 2 && !(w < h && w < factor) {
        (0.5 + w as f64 / w.div_ceil(factor) as f64) as usize
    } else {
        (0.5 + h as f64 / h.div_ceil(factor) as f64) as usize
    };
    let f = overview_factor(overview, full);
    f == factor || f == adjusted
}

/// Adds (or rebuilds) overviews of a local GeoTIFF, in place or in its `.ovr` sidecar, as
/// `gdaladdo` does. Existing overviews of the sizes asked for are recomputed; the others are
/// kept. Returns the sizes of the raster's overviews afterwards, largest first.
pub fn add_overviews(path: &str, options: &OverviewOptions) -> Result<Vec<(usize, usize)>> {
    let file = gtiff_update::writable_local_file(path)?;
    let mut src = open_source(path)?;
    let method = overview_method(options.resampling)?;
    let factors = factors(&src, &options.levels, options.min_size)?;
    if options.external {
        let ovr = ovr_path(&file);
        let main = parse_file(&file)?;
        if main.dirs.iter().any(Directory::is_overview) {
            return Err(Error::InvalidArgument(format!("'{path}' has internal overviews: external ones cannot be added (as in GDAL)")));
        }
        let mut sizes: Vec<(usize, usize)> = factors.iter().map(|f| (src.width().div_ceil(*f), src.height().div_ceil(*f))).collect();
        if ovr.is_file() {
            let existing = parse_file(&ovr)?;
            sizes.extend(existing.dirs.iter().filter(|d| d.is_overview()).map(Directory::size));
        }
        let compression = options.compression.as_deref().unwrap_or("NONE");
        let main_dir = &main.dirs[images(&main).main.unwrap_or(0)];
        let levels = build_new_levels(&mut src, main_dir, &sizes, method, compression)?;
        drop(src);
        gtiff_update::replace_file(&ovr, |f| write_ovr(f, levels, main_dir))?;
        return Ok(sorted_sizes(sizes));
    }
    let compression = options.compression.clone();
    let cog = options.ignore_cog_layout_break;
    gtiff_update::update_file(&file, move |f| {
        // The source is closed before the copy replaces the file.
        let out = add_internal(f, &mut src, &factors, method, compression.as_deref(), cog);
        drop(src);
        out
    })
}

/// [`add_overviews`] on a GeoTIFF held in memory (internal overviews only): returns the updated
/// file.
pub fn add_overviews_to_bytes(tiff: &[u8], options: &OverviewOptions) -> Result<Vec<u8>> {
    if options.external {
        return Err(Error::Unsupported("external overviews need a file: a GeoTIFF in memory takes internal overviews".into()));
    }
    let mut src = LazyRaster::from_tiff_bytes(tiff.to_vec())?.ok_or_else(|| Error::InvalidData("not a TIFF file".into()))?;
    let method = overview_method(options.resampling)?;
    let factors = factors(&src, &options.levels, options.min_size)?;
    let mut out = std::io::Cursor::new(tiff.to_vec());
    add_internal(&mut out, &mut src, &factors, method, options.compression.as_deref(), options.ignore_cog_layout_break)?;
    Ok(out.into_inner())
}

/// Deletes the overviews of a local GeoTIFF (`gdaladdo -clean`): its internal ones (and their
/// masks), or its `.ovr` sidecar with `external`. As in GDAL, the space internal overviews took
/// is not reclaimed.
pub fn delete_overviews(path: &str, options: &DeleteOptions) -> Result<()> {
    let file = gtiff_update::writable_local_file(path)?;
    if options.external {
        let ovr = ovr_path(&file);
        if ovr.is_file() {
            std::fs::remove_file(&ovr)?;
        }
        return Ok(());
    }
    gtiff_update::update_file(&file, |f| delete_internal(f, options.ignore_cog_layout_break))
}

/// [`delete_overviews`] on a GeoTIFF held in memory: returns the updated file.
pub fn delete_overviews_from_bytes(tiff: &[u8], options: &DeleteOptions) -> Result<Vec<u8>> {
    if options.external {
        return Err(Error::Unsupported("a GeoTIFF in memory has no external overviews".into()));
    }
    let mut out = std::io::Cursor::new(tiff.to_vec());
    delete_internal(&mut out, options.ignore_cog_layout_break)?;
    Ok(out.into_inner())
}

/// Recomputes the existing overviews of a local GeoTIFF from its full-resolution image, whole or
/// over some extents only (`gdal raster overview refresh`, `gdaladdo
/// -partial_refresh_from_projwin`). Returns the sizes of the overviews refreshed.
pub fn refresh_overviews(path: &str, options: &RefreshOptions) -> Result<Vec<(usize, usize)>> {
    let file = gtiff_update::writable_local_file(path)?;
    let mut src = open_source(path)?;
    let target = if options.external { ovr_path(&file) } else { file };
    if options.external && !target.is_file() {
        return Err(Error::InvalidArgument(format!("'{}' has no external overviews ({} is missing)", path, target.display())));
    }
    gtiff_update::update_file(&target, move |f| {
        let out = refresh(f, &mut src, options);
        drop(src);
        out
    })
}

/// [`refresh_overviews`] on a GeoTIFF held in memory (internal overviews): returns the updated
/// file.
pub fn refresh_overviews_in_bytes(tiff: &[u8], options: &RefreshOptions) -> Result<Vec<u8>> {
    if options.external {
        return Err(Error::Unsupported("a GeoTIFF in memory has no external overviews".into()));
    }
    let mut src = LazyRaster::from_tiff_bytes(tiff.to_vec())?.ok_or_else(|| Error::InvalidData("not a TIFF file".into()))?;
    let mut out = std::io::Cursor::new(tiff.to_vec());
    refresh(&mut out, &mut src, options)?;
    Ok(out.into_inner())
}

/// The extent `[min_x, min_y, max_x, max_y]` of a raster, in `crs` (the extent's corners and
/// edges transformed when the raster has another CRS): a `--like` dataset's extent.
pub fn dataset_bounds(path: &str, crs: &crate::crs::SpatialRef) -> Result<[f64; 4]> {
    let r = crate::open_raster_lazy(path)?;
    let gt = r.geo_transform();
    let (w, h) = (r.width() as f64, r.height() as f64);
    let corners = [(0.0, 0.0), (w, 0.0), (0.0, h), (w, h)].map(|(c, l)| super::apply_gt(&gt, c, l));
    let xs = corners.map(|c| c.0);
    let ys = corners.map(|c| c.1);
    let b = (xs.iter().copied().fold(f64::INFINITY, f64::min), ys.iter().copied().fold(f64::INFINITY, f64::min), xs.iter().copied().fold(f64::NEG_INFINITY, f64::max), ys.iter().copied().fold(f64::NEG_INFINITY, f64::max));
    let own = r.crs();
    if own.is_unknown() || crs.is_unknown() || own.same_as(crs) {
        return Ok([b.0, b.1, b.2, b.3]);
    }
    let t = super::Ct::new(&own, crs)?;
    let b = super::transform_bounds(&t, b, 21)?;
    Ok([b.0, b.1, b.2, b.3])
}

/// The `.ovr` sidecar of a file.
fn ovr_path(file: &std::path::Path) -> std::path::PathBuf {
    let mut p = file.as_os_str().to_owned();
    p.push(".ovr");
    p.into()
}

fn parse_file(path: &std::path::Path) -> Result<Tiff> {
    gtiff_update::parse(&mut std::io::BufReader::new(std::fs::File::open(path)?))
}

/// Opens a GeoTIFF to compute overviews from (with its `.msk` and `.ovr` sidecars).
fn open_source(path: &str) -> Result<LazyRaster> {
    let r = LazyRaster::open(path)?;
    if r.driver() != crate::formats::RasterDriver::GTiff || !r.is_lazy() {
        return Err(Error::Unsupported(format!("overviews are managed for GeoTIFFs only, not '{path}'")));
    }
    Ok(r)
}

/// The overview method of a resampling.
fn overview_method(r: Resampling) -> Result<OverviewResampling> {
    Ok(match r {
        Resampling::Nearest => OverviewResampling::Nearest,
        Resampling::Average => OverviewResampling::Average,
        Resampling::Min => OverviewResampling::Min,
        Resampling::Max => OverviewResampling::Max,
        Resampling::Mode => OverviewResampling::Mode,
        other => {
            return Err(Error::Unsupported(format!(
                "'{}' overviews are not supported (nearest, average, min, max, mode)",
                other.name()
            )));
        }
    })
}

fn factors(src: &LazyRaster, levels: &[usize], min_size: usize) -> Result<Vec<usize>> {
    if let Some(f) = levels.iter().find(|f| **f < 2) {
        return Err(Error::InvalidArgument(format!("an overview level is a decimation factor of 2 or more, not {f}")));
    }
    let mut f = if levels.is_empty() { automatic_levels(src.width(), src.height(), min_size) } else { levels.to_vec() };
    f.sort_unstable();
    f.dedup();
    Ok(f)
}

fn sorted_sizes(mut sizes: Vec<(usize, usize)>) -> Vec<(usize, usize)> {
    sizes.sort_by_key(|s| std::cmp::Reverse(*s));
    sizes.dedup();
    sizes
}

/// What an image of the file is, for the chain of directories.
struct Images {
    /// The images that are not overviews or their masks, in order (the full-resolution image,
    /// its mask), by index in the file.
    base: Vec<usize>,
    /// The index of the full-resolution image, when the file has it (not an `.ovr`).
    main: Option<usize>,
    /// Whether the full-resolution image has an internal mask.
    main_mask: bool,
    /// The overviews: size, index, and the index of its mask.
    overviews: Vec<((usize, usize), usize, Option<usize>)>,
}

fn images(tiff: &Tiff) -> Images {
    let mut base = Vec::new();
    let mut main = None;
    let mut masks = Vec::new();
    let mut overviews = Vec::new();
    for (i, d) in tiff.dirs.iter().enumerate() {
        if d.is_overview() {
            overviews.push((d.size(), i, None));
        } else if d.is_mask() && d.subfile() & gtiff_update::REDUCED != 0 {
            masks.push((d.size(), i));
        } else {
            if main.is_none() && !d.is_mask() {
                main = Some(i);
            }
            base.push(i);
        }
    }
    let main_size = main.map(|m| tiff.dirs[m].size());
    let main_mask = base.iter().any(|&i| tiff.dirs[i].is_mask() && Some(tiff.dirs[i].size()) == main_size);
    for o in &mut overviews {
        o.2 = masks.iter().find(|(s, _)| *s == o.0).map(|(_, i)| *i);
    }
    Images { base, main, main_mask, overviews }
}

/// The layout, compression and codec parameters of new images like `dir` (the full-resolution
/// image or an existing overview), in another compression when one is given.
fn layout_like(dir: &Directory, compression: Option<&str>, new_image: bool, data_type: DataType) -> Result<(Layout, Compression, CodecParams)> {
    if matches!(dir.uint(339), Some(5 | 6)) {
        return Err(Error::Unsupported("overviews of complex GeoTIFF samples are not supported".into()));
    }
    let bits = dir.uint(258).unwrap_or(1) as usize;
    if bits != data_type.size_bytes() * 8 {
        return Err(Error::Unsupported(format!("overviews of {bits}-bit samples are not supported")));
    }
    let own = Compression::from_tiff(dir.uint(259).unwrap_or(1) as u16, dir.get(50674).and_then(|e| e.uints().get(1).copied()));
    let compression = match compression {
        Some(name) => Compression::parse(name).ok_or_else(|| Error::Unsupported(format!("'{name}' overview compression is not supported")))?,
        None => own.unwrap_or_else(|| {
            crate::warnings::warn("the image's compression is not one terrio encodes: overviews are compressed with DEFLATE".into());
            Compression::Deflate
        }),
    };
    let samples = dir.uint(277).unwrap_or(1) as usize;
    let mut photometric = dir.uint(262).unwrap_or(1) as u16;
    if photometric == 6 && compression != Compression::Jpeg {
        photometric = 2;
    }
    let tile = if dir.tiled() {
        Some((dir.uint(322).unwrap_or(256) as u32, dir.uint(323).unwrap_or(256) as u32))
    } else if new_image {
        // GDAL tiles the overviews of striped images (GDAL_TIFF_OVR_BLOCKSIZE).
        Some((128, 128))
    } else {
        None
    };
    let predictor = if compression.takes_predictor() { dir.uint(317).unwrap_or(1) as u16 } else { 1 };
    let layout = Layout {
        tile,
        rows_per_strip: if tile.is_none() { dir.uint(278).map(|r| r as u32) } else { None },
        planar: dir.uint(284) == Some(2) && samples > 1,
        photometric,
        extra: dir.get(338).map(|e| e.uints().into_iter().map(|v| v as u16).collect()).unwrap_or_default(),
        predictor,
        parts: 1,
    };
    let codec = CodecParams { jpeg_ycbcr: photometric == 6, ..Default::default() };
    Ok((layout, compression, codec))
}

/// The chunk width and height of a layout for an image of `cols` columns (strips: the whole
/// width, and the strip rows the writer uses).
fn chunk_size(layout: &Layout, cols: usize) -> (usize, usize) {
    match layout.tile {
        Some((w, h)) => (w as usize, h as usize),
        None => (cols, layout.rows_per_strip.map_or(usize::MAX, |r| r.max(1) as usize)),
    }
}

/// One overview level being computed: its pixels over rows `rows` (all columns), and its mask
/// (255 valid) over the same rows when the raster has a per-dataset mask.
struct Computed {
    size: (usize, usize),
    rows: std::ops::Range<usize>,
    data: RasterData,
    mask: Option<Vec<u8>>,
}

/// How GDAL schedules the computation of a level.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Schedule {
    /// From the previous level computed (when it is larger), rather than from full resolution.
    from_prev: bool,
    /// The source rows of `GDALRegenerateOverviewsEx`'s chunks, which clamp the windows; `None`
    /// for `GDALRegenerateOverviewsMultiBand`'s whole windows.
    chunk_rows: Option<usize>,
}

/// The pixels overviews are computed from: the raster (its full-resolution image read once when
/// whole levels are computed) and its stored overviews.
struct Sources<'a> {
    src: &'a mut LazyRaster,
    base: Option<(RasterData, Option<Vec<u8>>)>,
    /// The alpha band (from 0), whose zeros GDAL leaves out of the other methods than nearest.
    alpha: Option<usize>,
}

impl Sources<'_> {
    /// Rows `rows` (all columns) of the full-resolution image (`level` `None`) or of a stored
    /// overview, with the level's mask (255 valid) when `want_mask`, and the rows of `computed`
    /// (the level's new pixels) over them.
    fn rows(&mut self, level: Option<usize>, size: (usize, usize), rows: std::ops::Range<usize>, computed: Option<&Computed>, want_mask: bool) -> Result<(RasterData, Option<Vec<u8>>)> {
        let (w, n) = (size.0, rows.len());
        let bands = self.src.band_count();
        if let Some(c) = computed.filter(|c| c.rows.start <= rows.start && c.rows.end >= rows.end) {
            let data = slice_rows(&c.data, c.rows.len(), w, bands, rows.start - c.rows.start..rows.end - c.rows.start);
            let mask = c.mask.as_ref().map(|m| m[(rows.start - c.rows.start) * w..(rows.end - c.rows.start) * w].to_vec());
            return Ok((data, mask));
        }
        if level.is_none() && rows == (0..size.1) {
            if self.base.is_none() {
                let data = self.src.read()?.raster.data;
                let mask = self.src.read_mask(Window::new(0, 0, size.0, size.1), None)?;
                self.base = Some((data, mask));
            }
            let (data, mask) = self.base.as_ref().unwrap();
            return Ok((data.clone(), mask.clone().filter(|_| want_mask)));
        }
        let window = Window::new(0, rows.start, w, n);
        let mut data = self.src.read_window(window, &WindowOptions { bands: Vec::new(), overview: level })?.raster.data;
        let mut mask = if want_mask { self.src.read_mask(window, level)? } else { None };
        if let Some(c) = computed {
            let (a, b) = (rows.start.max(c.rows.start), rows.end.min(c.rows.end));
            if a < b {
                overlay_rows(&mut data, n, rows.start, &c.data, c.rows.len(), c.rows.start, a..b, w, bands);
                if let (Some(m), Some(cm)) = (mask.as_mut(), c.mask.as_ref()) {
                    m[(a - rows.start) * w..(b - rows.start) * w].copy_from_slice(&cm[(a - c.rows.start) * w..(b - c.rows.start) * w]);
                }
            }
        }
        Ok((data, mask))
    }
}

/// Rows `range` of a band-major image of `rows` rows of `cols` columns.
fn slice_rows(data: &RasterData, rows: usize, cols: usize, bands: usize, range: std::ops::Range<usize>) -> RasterData {
    crate::pixel::map_data!(data, v => {
        let mut out = Vec::with_capacity(range.len() * cols * bands);
        for b in 0..bands {
            out.extend_from_slice(&v[(b * rows + range.start) * cols..(b * rows + range.end) * cols]);
        }
        out
    })
}

/// Copies rows `range` (image rows) from `src` (rows from `src_start`) into `dst` (rows from
/// `dst_start`), band-major images of the same type.
#[allow(clippy::too_many_arguments)]
fn overlay_rows(dst: &mut RasterData, dst_rows: usize, dst_start: usize, src: &RasterData, src_rows: usize, src_start: usize, range: std::ops::Range<usize>, cols: usize, bands: usize) {
    macro_rules! copy {
        ($($t:ident),*) => {
            match (dst, src) {
                $((RasterData::$t(d), RasterData::$t(s)) => {
                    for b in 0..bands {
                        let di = (b * dst_rows + range.start - dst_start) * cols;
                        let si = (b * src_rows + range.start - src_start) * cols;
                        let n = range.len() * cols;
                        d[di..di + n].copy_from_slice(&s[si..si + n]);
                    }
                })*
                _ => {}
            }
        };
    }
    copy!(U8, I8, U16, I16, U32, I32, U64, I64, F32, F64);
}

/// The source rows the overview rows `rows` of a level of `out_rows` rows are computed from, in
/// an image of `src_rows` rows (with a row of margin for the windows' rounding).
fn source_rows(rows: &std::ops::Range<usize>, out_rows: usize, src_rows: usize) -> std::ops::Range<usize> {
    let ratio = src_rows as f64 / out_rows as f64;
    let start = ((rows.start as f64 * ratio) as usize).saturating_sub(1);
    let end = (((rows.end as f64 * ratio).ceil() as usize) + 1).min(src_rows);
    start.min(end.saturating_sub(1))..end
}

/// Resamples one band (`values`, rows `src_rows` of a level of `src_size`) to rows `rows` of an
/// overview, as GDAL schedules it.
#[allow(clippy::too_many_arguments)]
fn resample_band(values: &[f64], valid: Option<&[bool]>, src_size: (usize, usize), src_rows: std::ops::Range<usize>, t: &kernel::Target, rows: std::ops::Range<usize>, chunk_rows: Option<usize>, group_origin: usize) -> Vec<f64> {
    let band = kernel::Band { values, valid, width: src_size.0, height: src_size.1, rows: src_rows.clone() };
    match chunk_rows {
        Some(chunk) if rows == (0..t.height) && src_rows == (0..src_size.1) => {
            let mut out = Vec::with_capacity(t.width * t.height);
            for (s, o) in kernel::ex_chunks(src_size.1, t.height, chunk) {
                out.extend(kernel::resample(&band, t, o, s, 0));
            }
            out
        }
        _ => kernel::resample(&band, t, rows, src_rows, group_origin),
    }
}

/// The values of band `b` of a band-major image of `n` pixels per band, as `f64`.
fn band_values(data: &RasterData, b: usize, n: usize) -> Vec<f64> {
    crate::pixel::with_data!(data, v => v[b * n..(b + 1) * n].iter().map(|x| crate::pixel::Pixel::to_f64(*x)).collect())
}

/// A band-major image of `data_type` from values of that type held as `f64`.
fn from_values(data_type: DataType, values: Vec<f64>) -> RasterData {
    match data_type {
        DataType::U8 => RasterData::U8(values.into_iter().map(|v| v as u8).collect()),
        DataType::I8 => RasterData::I8(values.into_iter().map(|v| v as i8).collect()),
        DataType::U16 => RasterData::U16(values.into_iter().map(|v| v as u16).collect()),
        DataType::I16 => RasterData::I16(values.into_iter().map(|v| v as i16).collect()),
        DataType::U32 => RasterData::U32(values.into_iter().map(|v| v as u32).collect()),
        DataType::I32 => RasterData::I32(values.into_iter().map(|v| v as i32).collect()),
        DataType::U64 => RasterData::U64(values.into_iter().map(|v| v as u64).collect()),
        DataType::I64 => RasterData::I64(values.into_iter().map(|v| v as i64).collect()),
        DataType::F32 => RasterData::F32(values.into_iter().map(|v| v as f32).collect()),
        DataType::F64 => RasterData::F64(values),
    }
}

/// What a level is computed from: the previous level (computed, or stored at `level`) or full
/// resolution.
fn source_of<'c>(sched: Schedule, prev: Option<&'c Computed>, size: (usize, usize), full: (usize, usize), overviews: &[(usize, usize)]) -> ((usize, usize), Option<usize>, Option<&'c Computed>) {
    match prev.filter(|p| sched.from_prev && p.size.0 > size.0) {
        Some(p) => (p.size, overviews.iter().position(|s| *s == p.size), Some(p)),
        None => (full, None, None),
    }
}

/// Whether GDAL computes the overviews of a TIFF image like `dir` with
/// `GDALRegenerateOverviewsMultiBand` (block by block, every level from the previous one)
/// rather than band by band with `GDALRegenerateOverviewsEx`: for internal overviews of
/// pixel-interleaved (or single-band, or alpha) images, and for compressed pixel-interleaved
/// external ones.
fn gdal_multiband(dir: &Directory, external: bool, compressed: bool, bands: usize, alpha: bool) -> bool {
    let contig = dir.uint(284) != Some(2) || bands == 1;
    if external { (compressed && contig) || alpha } else { contig || alpha }
}

/// The band GDAL takes as an alpha band (its color interpretation, which `GDAL_METADATA` may
/// override), numbered from 0.
fn alpha_band(src: &mut LazyRaster) -> Option<usize> {
    let header = src.header();
    let explicit = header.band_metadata.iter().position(|m| m.color_interp.as_deref().is_some_and(|c| c.eq_ignore_ascii_case("alpha")));
    if explicit.is_some() {
        return explicit;
    }
    let last = src.alpha_band()?.checked_sub(1)?;
    header.band_metadata.get(last).is_none_or(|m| m.color_interp.is_none()).then_some(last)
}

/// The schedules GDAL computes the levels of `sizes` (largest first, with their blocks) with:
/// for the data, and for the mask when the raster has one.
#[allow(clippy::too_many_arguments)]
fn schedules(full: (usize, usize), base_block: (usize, usize), sizes: &[(usize, usize)], blocks: &[(usize, usize)], method: OverviewResampling, data_type: DataType, multiband: bool, has_mask: bool) -> (Vec<Schedule>, Vec<Option<Schedule>>) {
    let factor = |src: (usize, usize), dst: (usize, usize)| {
        ((src.0 as f64 / dst.0 as f64 + 0.5) as usize).max((src.1 as f64 / dst.1 as f64 + 0.5) as usize).max(1)
    };
    // GDALRegenerateOverviewsEx on a band whose values are `value_size` bytes; `cascadable` for
    // averages and modes whose mask is the nodata value (or none).
    let ex = |value_size: usize, cascadable: bool| -> Vec<Schedule> {
        let cascade = method != OverviewResampling::Nearest && sizes.len() > 1 && cascadable;
        if cascade {
            return (0..sizes.len())
                .map(|i| {
                    let (src, block) = if i == 0 { (full, base_block) } else { (sizes[i - 1], blocks[i - 1]) };
                    let rows = kernel::ex_chunk_rows(src.0, src.1, block, factor(src, sizes[i]), value_size);
                    Schedule { from_prev: true, chunk_rows: rows }
                })
                .collect();
        }
        let max = sizes.iter().map(|s| factor(full, *s)).max().unwrap_or(1);
        match kernel::ex_chunk_rows(full.0, full.1, base_block, max, value_size) {
            Some(rows) => vec![Schedule { from_prev: false, chunk_rows: Some(rows) }; sizes.len()],
            None => vec![Schedule { from_prev: true, chunk_rows: None }; sizes.len()],
        }
    };
    let data = if multiband {
        vec![Schedule { from_prev: true, chunk_rows: None }; sizes.len()]
    } else {
        ex(kernel::work_size(method, data_type), !has_mask)
    };
    let mask = if has_mask { ex(1, true).into_iter().map(Some).collect() } else { vec![None; sizes.len()] };
    (data, mask)
}

/// Computes overview rows `rows` of a level of `size`, as GDAL does: the data with
/// `data` and, for a raster with a per-dataset mask, the mask's overview with `mask` (GDAL
/// computes masks band by band, then the data from them). `group_origin` is the first column of
/// the region GDAL computes (for the grouping of its factor-of-two averages).
#[allow(clippy::too_many_arguments)]
fn compute(srcs: &mut Sources, size: (usize, usize), rows: std::ops::Range<usize>, prev: Option<&Computed>, method: OverviewResampling, data: Schedule, mask: Option<Schedule>, group_origin: usize) -> Result<Computed> {
    let full = (srcs.src.width(), srcs.src.height());
    let overviews = srcs.src.overviews();
    let (bands, data_type, nodata) = (srcs.src.band_count(), srcs.src.data_type(), srcs.src.nodata());
    // The mask's overview.
    let new_mask = match mask {
        Some(ms) => {
            let (msize, level, computed) = source_of(ms, prev.filter(|p| p.mask.is_some()), size, full, &overviews);
            let mrows = if rows == (0..size.1) && ms.chunk_rows.is_some() { 0..msize.1 } else { source_rows(&rows, size.1, msize.1) };
            let (_, m) = srcs.rows(level, msize, mrows.clone(), computed, true)?;
            let m = m.ok_or_else(|| Error::InvalidData("the raster's mask could not be read".into()))?;
            let values: Vec<f64> = m.iter().map(|v| *v as f64).collect();
            let valid: Option<Vec<bool>> = (method != OverviewResampling::Nearest).then(|| m.iter().map(|v| *v != 0).collect());
            let t = kernel::Target { width: size.0, height: size.1, data_type: DataType::U8, nodata: None, method };
            let out = resample_band(&values, valid.as_deref(), msize, mrows, &t, rows.clone(), ms.chunk_rows, group_origin);
            Some(out.into_iter().map(|v| v as u8).collect::<Vec<u8>>())
        }
        None => None,
    };
    let (dsize, level, computed) = source_of(data, prev, size, full, &overviews);
    let drows = if rows == (0..size.1) && data.chunk_rows.is_some() { 0..dsize.1 } else { source_rows(&rows, size.1, dsize.1) };
    let masked = method != OverviewResampling::Nearest;
    let want_mask = masked && (mask.is_some() || (srcs.src.has_mask() && computed.is_none()));
    let (src_data, src_mask) = srcs.rows(level, dsize, drows.clone(), computed, want_mask)?;
    let n = dsize.0 * drows.len();
    let t = kernel::Target { width: size.0, height: size.1, data_type, nodata, method };
    let alpha: Option<Vec<bool>> = srcs.alpha.filter(|_| masked && src_mask.is_none()).map(|a| band_values(&src_data, a, n).iter().map(|v| *v != 0.0).collect());
    let mut out = Vec::with_capacity(size.0 * rows.len() * bands);
    for b in 0..bands {
        let values = band_values(&src_data, b, n);
        // GDAL's mask band: the per-dataset mask, else the alpha band, else the nodata value.
        let valid: Option<Vec<bool>> = match (&src_mask, &alpha, nodata) {
            _ if !masked => None,
            (Some(m), ..) => Some(m.iter().map(|v| *v != 0).collect()),
            (None, Some(a), _) => Some(a.clone()),
            (None, None, Some(nd)) => Some(values.iter().map(|v| !(*v == nd || (v.is_nan() && nd.is_nan()))).collect()),
            (None, None, None) => None,
        };
        out.extend(resample_band(&values, valid.as_deref(), dsize, drows.clone(), &t, rows.clone(), data.chunk_rows, group_origin));
    }
    Ok(Computed { size, rows, data: from_values(data_type, out), mask: new_mask })
}

/// The `GDAL_METADATA` of an overview: the method it was computed with, as GDAL records it.
fn resampling_item(method: OverviewResampling) -> String {
    format!("<GDALMetadata><Item name=\"RESAMPLING\" sample=\"0\">{}</Item></GDALMetadata>", method.gdal_name())
}

/// The method an existing overview was computed with, from its `RESAMPLING` item.
fn stored_method(dir: &Directory) -> Option<Result<OverviewResampling>> {
    let xml = dir.get(42112)?.ascii()?;
    let item = gw::gdal_metadata_items(&xml).into_iter().find(|i| i.attr("name") == Some("RESAMPLING"))?;
    let name = item.text.trim().to_ascii_lowercase();
    Some(OverviewResampling::parse(&name).map_err(|_| {
        Error::Unsupported(format!("the overviews were built with '{name}' resampling, which terrio does not compute: pick another one"))
    }))
}

/// Encodes a computed level as a new overview image like `like` (the full-resolution image).
fn new_level_image(c: &Computed, like: &Directory, compression: Option<&str>, method: OverviewResampling, data_type: DataType, bands: usize) -> Result<(gw::Image, Option<gw::Image>)> {
    let (layout, compression, codec) = layout_like(like, compression, true, data_type)?;
    let level = gw::Level { cols: c.size.0, rows: c.size.1, bands, data: std::borrow::Cow::Borrowed(&c.data) };
    let mut img = level.image(data_type, compression, &codec, &layout)?;
    img.tags.push((254, Value::Longs(vec![1])));
    img.tags.push((42112, Value::Ascii(resampling_item(method))));
    let mask = match &c.mask {
        Some(m) => {
            let (cw, ch) = chunk_size(&layout, c.size.0);
            let mut mi = gw::mask_image(m, c.size.0, c.size.1, (cw, ch.min(c.size.1), layout.tile.is_some()))?;
            for (t, v) in &mut mi.tags {
                if *t == 254 {
                    *v = Value::Longs(vec![5]);
                }
            }
            Some(mi)
        }
        None => None,
    };
    Ok((img, mask))
}

/// Copies the tags of the full-resolution image overviews share: nodata, the color table and
/// the JPEG reference values.
fn shared_tags(main: &Directory, dir: &mut Directory) {
    for tag in [320, 532, 42113] {
        if let Some(e) = main.get(tag) {
            dir.set(e.clone());
        }
    }
}

/// Computes whole new levels of `sizes` (in cascade, largest first) as images like `like`
/// (without mask images: for an `.ovr` sidecar).
fn build_new_levels(src: &mut LazyRaster, like: &Directory, sizes: &[(usize, usize)], method: OverviewResampling, compression: &str) -> Result<Vec<gw::Image>> {
    let sizes = sorted_sizes(sizes.to_vec());
    let (layout, new_compression, ..) = layout_like(like, Some(compression), true, src.data_type())?;
    let blocks: Vec<(usize, usize)> = sizes.iter().map(|s| chunk_size(&layout, s.0)).collect();
    let alpha = alpha_band(src);
    let multiband = gdal_multiband(like, true, new_compression != Compression::None, src.band_count(), alpha.is_some());
    let base_block = base_block(src);
    let (data, _) = schedules((src.width(), src.height()), base_block, &sizes, &blocks, method, src.data_type(), multiband, src.has_mask());
    let mut srcs = Sources { src, base: None, alpha };
    let mut prev: Option<Computed> = None;
    let mut out = Vec::new();
    for (i, &size) in sizes.iter().enumerate() {
        let c = compute(&mut srcs, size, 0..size.1, prev.as_ref(), method, data[i], None, 0)?;
        out.push(new_level_image(&c, like, Some(compression), method, srcs.src.data_type(), srcs.src.band_count())?.0);
        prev = Some(c);
    }
    Ok(out)
}

/// The block size of the full-resolution image (strips: the whole width).
fn base_block(src: &LazyRaster) -> (usize, usize) {
    src.block_size().unwrap_or((src.width(), src.height()))
}

/// Writes a new `.ovr` file: the overview images, largest first, with the tags of the
/// full-resolution image `main` they share.
fn write_ovr(f: &mut std::fs::File, images: Vec<gw::Image>, main: &Directory) -> Result<()> {
    let bytes: u64 = images.iter().flat_map(|i| i.chunks.iter()).map(|c| c.len() as u64 + 2).sum();
    let big = bytes > u32::MAX as u64 - (1 << 24);
    f.write_all(&gtiff_update::header(big))?;
    let mut a = Appender::new(f, big)?;
    let mut dirs = Vec::new();
    for img in images {
        let mut dir = a.image(img)?;
        shared_tags(main, &mut dir);
        dirs.push(dir);
    }
    a.finish(&dirs)
}

/// Adds overviews of these factors to a TIFF (the file, or a copy of it).
fn add_internal<F: Read + Write + Seek>(
    file: &mut F,
    src: &mut LazyRaster,
    factors: &[usize],
    method: OverviewResampling,
    compression: Option<&str>,
    ignore_cog_layout_break: bool,
) -> Result<Vec<(usize, usize)>> {
    let tiff = gtiff_update::parse(file)?;
    gtiff_update::break_cog_layout(file, tiff.big, ignore_cog_layout_break)?;
    let im = images(&tiff);
    let main = im.main.ok_or_else(|| Error::InvalidData("the TIFF has no full-resolution image".into()))?;
    let (w, h) = (src.width(), src.height());
    // Each factor's existing overview, or the size of a new one.
    let mut used = vec![false; im.overviews.len()];
    let mut sizes = Vec::new();
    for &f in factors {
        match im.overviews.iter().enumerate().find(|(j, o)| !used[*j] && matches_factor(o.0, (w, h), f)) {
            Some((j, o)) => {
                used[j] = true;
                sizes.push(o.0);
            }
            None => sizes.push((w.div_ceil(f), h.div_ceil(f))),
        }
    }
    let sizes = sorted_sizes(sizes);
    let main_dir = &tiff.dirs[main];
    if method == OverviewResampling::Average && main_dir.uint(262) == Some(3) {
        return Err(Error::Unsupported("average overviews of palette images are not supported (GDAL averages their colors): use nearest or mode".into()));
    }
    // The block of each level: an existing level's own, else the one new images take.
    let (new_layout, ..) = layout_like(main_dir, compression, true, src.data_type())?;
    let blocks: Vec<(usize, usize)> = sizes
        .iter()
        .map(|s| match im.overviews.iter().find(|o| o.0 == *s) {
            Some(&(_, i, _)) => layout_like(&tiff.dirs[i], None, false, src.data_type()).map(|(l, ..)| chunk_size(&l, s.0)),
            None => Ok(chunk_size(&new_layout, s.0)),
        })
        .collect::<Result<_>>()?;
    let alpha = alpha_band(src);
    let multiband = gdal_multiband(main_dir, false, true, src.band_count(), alpha.is_some());
    let (mut data, masks) = schedules((w, h), base_block(src), &sizes, &blocks, method, src.data_type(), multiband, im.main_mask);
    if !im.main_mask && src.has_mask() {
        // A `.msk` sidecar: its pixels count, and GDAL cannot cascade over it.
        data = schedules((w, h), base_block(src), &sizes, &blocks, method, src.data_type(), multiband, true).0;
    }
    let mut srcs = Sources { src, base: None, alpha };
    let mut a = Appender::new(file, tiff.big)?;
    let mut prev: Option<Computed> = None;
    // Every overview of the result: (size, directory, mask directory).
    let mut levels: Vec<((usize, usize), Directory, Option<Directory>)> = Vec::new();
    for (k, &size) in sizes.iter().enumerate() {
        let existing = im.overviews.iter().find(|o| o.0 == size);
        let c = compute(&mut srcs, size, 0..size.1, prev.as_ref(), method, data[k], masks[k], 0)?;
        let src = &mut *srcs.src;
        let (dir, mask) = match existing {
            // An existing level is recomputed in its own layout.
            Some(&(_, i, mask)) => {
                let mut dir = replace_chunks(&mut a, &tiff.dirs[i], &c, 0..c.size.0, src.data_type(), src.band_count(), None)?;
                dir.set(Entry::new(42112, &Value::Ascii(resampling_item(method)), tiff.big));
                let mask = match (mask, &c.mask) {
                    (Some(m), Some(_)) => Some(replace_chunks(&mut a, &tiff.dirs[m], &c, 0..c.size.0, src.data_type(), src.band_count(), Some(&tiff.dirs[i]))?),
                    (None, Some(_)) if im.main_mask => new_level_image(&c, main_dir, compression, method, src.data_type(), src.band_count())?.1.map(|m| a.image(m)).transpose()?,
                    _ => None,
                };
                (dir, mask)
            }
            None => {
                let (img, mask) = new_level_image(&c, main_dir, compression, method, src.data_type(), src.band_count())?;
                let mut dir = a.image(img)?;
                shared_tags(main_dir, &mut dir);
                let mask = if im.main_mask { mask.map(|m| a.image(m)).transpose()? } else { None };
                (dir, mask)
            }
        };
        levels.push((size, dir, mask));
        prev = Some(c);
    }
    for &(size, i, mask) in &im.overviews {
        if !sizes.contains(&size) {
            levels.push((size, tiff.dirs[i].clone(), mask.map(|m| tiff.dirs[m].clone())));
        }
    }
    levels.sort_by_key(|l| std::cmp::Reverse(l.0));
    let mut chain: Vec<Directory> = im.base.iter().map(|&i| tiff.dirs[i].clone()).collect();
    for (_, dir, mask) in &levels {
        chain.push(dir.clone());
        chain.extend(mask.clone());
    }
    a.finish(&chain)?;
    Ok(levels.iter().map(|l| l.0).collect())
}

/// Re-encodes the chunks of `dir` (an overview, or with `mask_of` the mask of that overview)
/// that rows `c.rows` and columns `cols` of the computed level cover, and returns the directory
/// pointing at them (and at the old chunks elsewhere).
fn replace_chunks<F: Write + Seek>(
    a: &mut Appender<'_, F>,
    dir: &Directory,
    c: &Computed,
    cols: std::ops::Range<usize>,
    data_type: DataType,
    bands: usize,
    mask_of: Option<&Directory>,
) -> Result<Directory> {
    let (w, h) = dir.size();
    if (w, h) != c.size {
        return Err(Error::InvalidData("an overview changed size".into()));
    }
    let rows_n = c.rows.len();
    let (img, chunk) = match mask_of {
        None => {
            let (layout, compression, codec) = layout_like(dir, None, false, data_type)?;
            if dir.uint(277).unwrap_or(1) as usize != bands {
                return Err(Error::InvalidData("an overview has another band count than the raster".into()));
            }
            let level = gw::Level { cols: w, rows: rows_n, bands, data: std::borrow::Cow::Borrowed(&c.data) };
            let chunk = chunk_size(&layout, w);
            (level.image(data_type, compression, &codec, &layout)?, chunk)
        }
        Some(_) => {
            let m = c.mask.as_ref().ok_or_else(|| Error::InvalidData("no mask computed".into()))?;
            let tiled = dir.tiled();
            let chunk = if tiled {
                (dir.uint(322).unwrap_or(256) as usize, dir.uint(323).unwrap_or(256) as usize)
            } else {
                (w, dir.uint(278).unwrap_or(h as u64).max(1) as usize)
            };
            (gw::mask_image(m, w, rows_n, (chunk.0, chunk.1.min(rows_n), tiled))?, chunk)
        }
    };
    let (cw, ch) = (chunk.0, chunk.1.min(h.max(1)));
    if c.rows.start % ch != 0 || (c.rows.end != h && c.rows.end % ch != 0) {
        return Err(Error::InvalidData("refreshed overview rows are not aligned to its blocks".into()));
    }
    let across = w.div_ceil(cw);
    let down = h.div_ceil(ch);
    let strip_down = rows_n.div_ceil(ch);
    let (mut offsets, mut counts) = dir.chunks();
    let planes = (offsets.len() / (across * down).max(1)).max(1);
    if offsets.len() != planes * across * down || counts.len() != offsets.len() {
        return Err(Error::InvalidData("an overview's chunk offsets do not match its size".into()));
    }
    let strip_per_plane = across * strip_down;
    if img.chunks.len() != planes * strip_per_plane {
        return Err(Error::InvalidData("an overview's layout differs from the one recomputed".into()));
    }
    let (tx0, tx1) = (cols.start / cw, cols.end.div_ceil(cw).min(across));
    for (k, chunk) in img.chunks.into_iter().enumerate() {
        let (plane, j) = (k / strip_per_plane, k % strip_per_plane);
        let (ty, tx) = (j / across + c.rows.start / ch, j % across);
        if tx < tx0 || tx >= tx1 {
            continue;
        }
        let idx = plane * across * down + ty * across + tx;
        offsets[idx] = a.write(&chunk)?;
        counts[idx] = chunk.len() as u64;
    }
    let mut out = dir.clone();
    let big = a.is_big();
    let (o, cn) = if dir.tiled() { (324, 325) } else { (273, 279) };
    out.set(Entry::new(o, &Value::Offsets(offsets), big));
    out.set(Entry::new(cn, &Value::Offsets(counts), big));
    Ok(out)
}

/// Removes the overviews (and their masks) from the chain of a TIFF.
fn delete_internal<F: Read + Write + Seek>(file: &mut F, ignore_cog_layout_break: bool) -> Result<()> {
    let tiff = gtiff_update::parse(file)?;
    let chain: Vec<Directory> = tiff.dirs.iter().filter(|d| d.subfile() & gtiff_update::REDUCED == 0).cloned().collect();
    if chain.len() == tiff.dirs.len() {
        return Ok(());
    }
    gtiff_update::break_cog_layout(file, tiff.big, ignore_cog_layout_break)?;
    Appender::new(file, tiff.big)?.finish(&chain)
}

/// Refreshes the overviews of a TIFF (a GeoTIFF, or the `.ovr` of `src`).
fn refresh<F: Read + Write + Seek>(file: &mut F, src: &mut LazyRaster, o: &RefreshOptions) -> Result<Vec<(usize, usize)>> {
    let tiff = gtiff_update::parse(file)?;
    gtiff_update::break_cog_layout(file, tiff.big, o.ignore_cog_layout_break)?;
    let im = images(&tiff);
    let (w, h) = (src.width(), src.height());
    let mut targets: Vec<((usize, usize), usize, Option<usize>)> = im
        .overviews
        .iter()
        .copied()
        .filter(|(s, ..)| {
            o.levels.is_empty() || o.levels.iter().any(|f| matches_factor(*s, (w, h), *f))
        })
        .collect();
    if targets.is_empty() {
        return Err(Error::InvalidArgument("no overview to refresh".into()));
    }
    targets.sort_by_key(|t| std::cmp::Reverse(t.0));
    let methods: Vec<OverviewResampling> = targets
        .iter()
        .map(|(_, i, _)| match o.resampling {
            Some(r) => overview_method(r),
            None => stored_method(&tiff.dirs[*i]).unwrap_or(Ok(OverviewResampling::Nearest)),
        })
        .collect::<Result<_>>()?;
    // The full-resolution pixels to refresh from, [x0, y0, x1, y1).
    let region = match o.bbox.as_slice() {
        [] => Some([0, 0, w, h]),
        boxes => {
            let inv = super::invert_gt(&src.geo_transform()).ok_or_else(|| Error::InvalidData("the raster's geotransform cannot be inverted".into()))?;
            let mut r: Option<[usize; 4]> = None;
            for b in boxes {
                let pts = [(b[0], b[1]), (b[2], b[1]), (b[0], b[3]), (b[2], b[3])].map(|(x, y)| super::apply_gt(&inv, x, y));
                let (c0, c1) = (pts.iter().map(|p| p.0).fold(f64::INFINITY, f64::min), pts.iter().map(|p| p.0).fold(f64::NEG_INFINITY, f64::max));
                let (l0, l1) = (pts.iter().map(|p| p.1).fold(f64::INFINITY, f64::min), pts.iter().map(|p| p.1).fold(f64::NEG_INFINITY, f64::max));
                let x0 = c0.floor().clamp(0.0, w as f64) as usize;
                let x1 = c1.ceil().clamp(0.0, w as f64) as usize;
                let y0 = l0.floor().clamp(0.0, h as f64) as usize;
                let y1 = l1.ceil().clamp(0.0, h as f64) as usize;
                if x0 < x1 && y0 < y1 {
                    r = Some(match r {
                        None => [x0, y0, x1, y1],
                        Some(p) => [p[0].min(x0), p[1].min(y0), p[2].max(x1), p[3].max(y1)],
                    });
                }
            }
            r
        }
    };
    let Some(region) = region else {
        crate::warnings::warn("the extents to refresh do not intersect the raster: no overview changed".into());
        return Ok(Vec::new());
    };
    let sizes: Vec<(usize, usize)> = targets.iter().map(|t| t.0).collect();
    let blocks: Vec<(usize, usize)> = targets
        .iter()
        .map(|&(s, i, _)| layout_like(&tiff.dirs[i], None, false, src.data_type()).map(|(l, ..)| chunk_size(&l, s.0)))
        .collect::<Result<_>>()?;
    let alpha = alpha_band(src);
    let multiband = match im.main {
        Some(m) => gdal_multiband(&tiff.dirs[m], false, true, src.band_count(), alpha.is_some()),
        None => {
            let first = &tiff.dirs[targets[0].1];
            gdal_multiband(first, true, first.uint(259).unwrap_or(1) != 1, src.band_count(), alpha.is_some())
        }
    };
    let has_mask = targets.iter().all(|t| t.2.is_some());
    let whole = region == [0, 0, w, h];
    let mut a = Appender::new(file, tiff.big)?;
    let mut dirs = tiff.dirs.clone();
    let mut prev: Option<Computed> = None;
    let mut srcs = Sources { src, base: None, alpha };
    for (k, (&(size, i, mask), method)) in targets.iter().zip(methods).enumerate() {
        let (mut data, mut masks) = schedules((w, h), base_block(srcs.src), &sizes, &blocks, method, srcs.src.data_type(), multiband, has_mask);
        if !whole {
            // Partial refreshes compute whole windows (GDALRegenerateOverviewsMultiBand's region).
            for s in data.iter_mut().chain(masks.iter_mut().flatten()) {
                s.chunk_rows = None;
            }
        }
        // GDAL's region of the overview, and the rows of whole blocks over it.
        const EPS: f64 = 1e-8;
        let dx0 = (region[0] as f64 / w as f64 * size.0 as f64 + EPS) as usize;
        let dx1 = ((region[2] as f64 / w as f64 * size.0 as f64 - EPS).ceil() as usize).min(size.0).max(dx0 + 1);
        let dy0 = (region[1] as f64 / h as f64 * size.1 as f64 + EPS) as usize;
        let dy1 = ((region[3] as f64 / h as f64 * size.1 as f64 - EPS).ceil() as usize).min(size.1).max(dy0 + 1);
        let ch = blocks[k].1.min(size.1).max(1);
        let (r0, r1) = (dy0 / ch * ch, (dy1.div_ceil(ch) * ch).min(size.1));
        let mut c = compute(&mut srcs, size, r0..r1, prev.as_ref(), method, data[k], masks[k].filter(|_| mask.is_some()), dx0)?;
        if (dx0, dy0, dx1, dy1) != (0, 0, size.0, size.1) {
            // Outside the region, the overview keeps its pixels.
            let level = srcs.src.overviews().iter().position(|s| *s == size);
            let (old, old_mask) = srcs.rows(level, size, r0..r1, None, c.mask.is_some())?;
            keep_outside(&mut c, &old, old_mask.as_deref(), dx0..dx1, dy0..dy1, srcs.src.band_count());
        }
        let mut dir = replace_chunks(&mut a, &tiff.dirs[i], &c, dx0..dx1, srcs.src.data_type(), srcs.src.band_count(), None)?;
        if whole {
            dir.set(Entry::new(42112, &Value::Ascii(resampling_item(method)), tiff.big));
        }
        dirs[i] = dir;
        if let (Some(m), Some(_)) = (mask, &c.mask) {
            dirs[m] = replace_chunks(&mut a, &tiff.dirs[m], &c, dx0..dx1, srcs.src.data_type(), srcs.src.band_count(), Some(&tiff.dirs[i]))?;
        }
        prev = Some(c);
    }
    a.finish(&dirs)?;
    Ok(targets.iter().map(|t| t.0).collect())
}

/// Puts back the pixels (and mask values) of `old` (the level's stored rows `c.rows`) outside
/// columns `cols` and rows `rows` of a computed level.
fn keep_outside(c: &mut Computed, old: &RasterData, old_mask: Option<&[u8]>, cols: std::ops::Range<usize>, rows: std::ops::Range<usize>, bands: usize) {
    let (w, n) = (c.size.0, c.rows.len());
    let inside = |y: usize, x: usize| rows.contains(&y) && cols.contains(&x);
    macro_rules! keep {
        ($($t:ident),*) => {
            match (&mut c.data, old) {
                $((RasterData::$t(d), RasterData::$t(o)) => {
                    for b in 0..bands {
                        for (j, y) in c.rows.clone().enumerate() {
                            for x in 0..w {
                                if !inside(y, x) {
                                    let i = (b * n + j) * w + x;
                                    d[i] = o[i];
                                }
                            }
                        }
                    }
                })*
                _ => {}
            }
        };
    }
    keep!(U8, I8, U16, I16, U32, I32, U64, I64, F32, F64);
    if let (Some(m), Some(o)) = (c.mask.as_mut(), old_mask) {
        for (j, y) in c.rows.clone().enumerate() {
            for x in 0..w {
                if !inside(y, x) {
                    m[j * w + x] = o[j * w + x];
                }
            }
        }
    }
}

fn invalid(value: &str) -> Error {
    Error::InvalidArgument(format!("invalid overview selection '{value}' (AUTO, AUTO-n, NONE or a level from 0)"))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn gdal_ratios_and_fallbacks() {
        let levels = [(512, 512), (256, 256), (128, 128)];
        assert_eq!(OverviewSelection::Auto.select(1024, &levels, 1.0), None);
        assert_eq!(OverviewSelection::Auto.select(1024, &levels, 1.95), Some(0));
        assert_eq!(OverviewSelection::Auto.select(1024, &levels, 3.0), Some(0));
        assert_eq!(OverviewSelection::Auto.select(1024, &levels, 4.0), Some(1));
        assert_eq!(OverviewSelection::AutoOffset(1).select(1024, &levels, 4.0), Some(0));
        assert_eq!(OverviewSelection::AutoOffset(3).select(1024, &levels, 4.0), None);
        assert_eq!(OverviewSelection::None.select(1024, &levels, 16.0), None);
        assert_eq!(OverviewSelection::Level(99).select(1024, &levels, 1.0), Some(2));
        assert_eq!(OverviewSelection::Level(0).select(1024, &[], 1.0), None);
        assert_eq!(OverviewSelection::parse("auto-2").unwrap(), OverviewSelection::AutoOffset(2));
        for invalid in ["-1", "AUTO-", "AUTO--1", "bilinear"] {
            assert!(OverviewSelection::parse(invalid).is_err());
        }
    }
}
