library(dplyr)
library(tidyr)
library(Xcertainty)
library(coda)



# 1. CONFIGURATION
CSV_PATH  <- "/users/matteotosato/desktop/dataset_pm/dati_tesi_width3.csv"
OUT_DIR   <- "/users/matteotosato/desktop/dataset_pm/results_5frames_final"   # single final output folder (read by B and C)
RUN_TAG   <- "v9_adaptive_tilt"

MAX_ANIMAL_FRAMES <- 5
MAX_ALTITUDE_SPAN_M  <- 3
MAX_CALIBRATION_FRAMES      <- 5
MAX_CALIBRATION_ALTITUDE_SPAN_M <- 3
REDUCE_CALIBRATION         <- TRUE

POOR_WIDTH_QUALITY <- 3
GOOD_WIDTH_QUALITIES    <- c(1, 2)
HT_MIN <- 35
HT_MAX <- 80
HEAD_MIN <- 5          # exploratory head / spermaceti region
HEAD_MAX <- 35

MAX_TILT_DEG <- 10
FALLBACK_TILT_DEG <- 20
KEEP_MISSING_TILT <- TRUE
H_MIN <- 0.1; H_MAX <- 130

N_ITER   <- 4e6
N_CHAINS <- 3
REUSE_CHAINS <- TRUE    # reuse Xcertainty_chains.rds if present, instead of re-running the MCMC
N_THIN   <- 100

set.seed(123)
if (!dir.exists(OUT_DIR)) dir.create(OUT_DIR, recursive = TRUE)
cat(sprintf("\n### RUN %s | maximum %d frames per individual-year-flight-altitude group; span <= %g m ###\n",
            RUN_TAG, MAX_ANIMAL_FRAMES, MAX_ALTITUDE_SPAN_M))

# 2. IMPORT DATA AND CORRECT BAROMETRIC ALTITUDE
dat <- read.csv(CSV_PATH, stringsAsFactors = FALSE)
dat$date <- as.Date(dat$date)
dat$year <- as.integer(format(dat$date, "%Y"))

dat$h_laser <- ifelse(!is.na(dat$h_laser) & (dat$h_laser < H_MIN | dat$h_laser > H_MAX),
                      NA_real_, dat$h_laser)

# The barometric correction is done ONCE, in build_data_v2.py: take-off height = mean (laser - raw barometer) over all log seconds with a valid laser reading;
# manual z_baro only for flights without laser (column z_source). Here the corrected barometric altitude of the export (h_baro = h_baro_raw + z_baro) is used as it is, with no second estimate.
stopifnot(all(c("h_baro", "z_baro", "z_source") %in% names(dat)))
dat <- dat %>% mutate(z_used = z_baro, h_baro_corr = h_baro)
stopifnot(isTRUE(all.equal(dat$h_baro_corr, dat$h_baro_raw + dat$z_used)))

flights_without_laser <- unique(dat$flight[dat$z_source != "laser_mean"])
if (length(flights_without_laser) > 0)
  cat(sprintf("Flights without laser data (using the predefined barometric offset): %s\n",
              paste(flights_without_laser, collapse = ", ")))

# 3. IMAGE IDENTIFIERS AND DUPLICATE CHECKS
dat <- dat %>%
  mutate(image_id = ifelse(!is.na(frame_pos),
                           paste0(flight, "_s", sec_video, "_f", as.integer(frame_pos)),
                           paste0(flight, "_s", sec_video)),
         subject_key = ifelse(kind == "cal", paste("CAL", drone, L_known), as.character(animal)))

dup <- dat %>% count(kind, subject_key, image_id) %>% filter(n > 1)
if (nrow(dup) > 0) { print(as.data.frame(dup)); stop("Duplicate rows for the same subject and image") }

# 4. FRAME SELECTION AND ADAPTIVE TILT LIMIT
width_cols <- grep("^width_[0-9]+pct_px$", names(dat), value = TRUE)
required_selection <- c("quality_verdict", "quality_width_measurability", "eyes", "tilt_deg")
missing_selection <- setdiff(required_selection, names(dat))
if (length(missing_selection) > 0) {
  stop(paste("Missing columns required for frame selection:",
             paste(missing_selection, collapse = ", ")))
}
stopifnot(length(MAX_TILT_DEG) == 1L, is.finite(MAX_TILT_DEG), MAX_TILT_DEG >= 0,
          length(FALLBACK_TILT_DEG) == 1L, is.finite(FALLBACK_TILT_DEG),
          FALLBACK_TILT_DEG >= MAX_TILT_DEG,
          length(KEEP_MISSING_TILT) == 1L, !is.na(KEEP_MISSING_TILT))

stopifnot(length(MAX_ALTITUDE_SPAN_M) == 1L,
          is.finite(MAX_ALTITUDE_SPAN_M), MAX_ALTITUDE_SPAN_M > 0,
          length(MAX_ANIMAL_FRAMES) == 1L,
          is.finite(MAX_ANIMAL_FRAMES), MAX_ANIMAL_FRAMES >= 1,
          MAX_ANIMAL_FRAMES == as.integer(MAX_ANIMAL_FRAMES))
stopifnot(length(MAX_CALIBRATION_ALTITUDE_SPAN_M) == 1L,
          is.finite(MAX_CALIBRATION_ALTITUDE_SPAN_M), MAX_CALIBRATION_ALTITUDE_SPAN_M > 0,
          length(MAX_CALIBRATION_FRAMES) == 1L,
          is.finite(MAX_CALIBRATION_FRAMES), MAX_CALIBRATION_FRAMES >= 1,
          MAX_CALIBRATION_FRAMES == as.integer(MAX_CALIBRATION_FRAMES))

# Start a new group only when its span would exceed the allowed range.
assign_altitude_groups <- function(altitudes, max_span = MAX_ALTITUDE_SPAN_M) {
  if (!length(altitudes)) return(integer(0))
  stopifnot(all(is.finite(altitudes)), length(max_span) == 1L,
            is.finite(max_span), max_span > 0)
  order_index <- order(altitudes)
  group_ids <- integer(length(altitudes))
  group_id <- 1L
  group_min <- altitudes[order_index[1L]]
  for (i in order_index) {
    if (altitudes[i] - group_min > max_span) {
      group_id <- group_id + 1L
      group_min <- altitudes[i]
    }
    group_ids[i] <- group_id
  }
  group_ids
}

whale_all <- dat %>% filter(kind == "whale") %>%
  mutate(
    selection_verdict = tolower(trimws(as.character(quality_verdict))),
    passed_verdict = as.integer(!is.na(selection_verdict) & selection_verdict == "passed"),
    has_laser = as.integer(is.finite(h_laser) & h_laser >= H_MIN & h_laser <= H_MAX),
    width_quality = ifelse(is.na(quality_width_measurability), 99, quality_width_measurability),
    quality_eligible = passed_verdict == 1L & width_quality %in% GOOD_WIDTH_QUALITIES,
    has_eyes = as.integer(is.finite(eyes) & eyes > 0),
    selection_altitude = ifelse(has_laser == 1L, h_laser, h_baro_corr),
    valid_altitude = is.finite(selection_altitude) & selection_altitude >= H_MIN & selection_altitude <= H_MAX,
    valid_total_length = is.finite(px) & px > 0,
    selection_unit = paste(animal, year, flight),
    tilt_missing = is.na(tilt_deg),
    tilt_pass_default = (tilt_missing & KEEP_MISSING_TILT) |
      (!tilt_missing & is.finite(tilt_deg) & abs(tilt_deg) <= MAX_TILT_DEG),
    tilt_pass_ceiling = (tilt_missing & KEEP_MISSING_TILT) |
      (!tilt_missing & is.finite(tilt_deg) & abs(tilt_deg) <= FALLBACK_TILT_DEG)
  ) %>%
  group_by(animal, year, flight) %>%
  mutate(
    n_quality_eligible_ceiling = sum(quality_eligible & valid_altitude &
                                       valid_total_length & tilt_pass_ceiling),
    quality_pool = quality_eligible | n_quality_eligible_ceiling == 0L,
    n_usable_default_tilt = sum(quality_pool & valid_altitude &
                                  valid_total_length & tilt_pass_default),
    tilt_relaxed = n_usable_default_tilt == 0L,
    tilt_limit_deg = ifelse(tilt_relaxed, FALLBACK_TILT_DEG, MAX_TILT_DEG),
    tilt_allowed = ifelse(tilt_relaxed, tilt_pass_ceiling, tilt_pass_default),
    n_quality_eligible_flight = sum(quality_eligible & valid_altitude &
                                      valid_total_length & tilt_allowed),
    quality_fallback = n_quality_eligible_flight == 0L,
    eligible = valid_altitude & valid_total_length & tilt_allowed &
      (quality_eligible | quality_fallback),
    tilt_rescued = eligible & !tilt_missing & abs(tilt_deg) > MAX_TILT_DEG
  ) %>%
  ungroup()

candidates <- whale_all %>% filter(eligible) %>%
  group_by(animal, year, flight) %>%
  mutate(altitude_group = assign_altitude_groups(selection_altitude)) %>%
  ungroup() %>%
  group_by(animal, year, flight, altitude_group) %>%
  mutate(group_mean_altitude_m = mean(selection_altitude),
         group_min_altitude_m = min(selection_altitude),
         group_max_altitude_m = max(selection_altitude)) %>%
  ungroup()
if (nrow(candidates) == 0) stop("No animal frames have usable altitude, total length and tilt.")

sel <- candidates %>%
  group_by(animal, year, flight, altitude_group) %>%
  arrange(desc(has_laser), width_quality, desc(has_eyes), image_id, .by_group = TRUE) %>%
  mutate(selection_rank = row_number()) %>%
  ungroup()
whale <- sel %>% filter(selection_rank <= MAX_ANIMAL_FRAMES)

stopifnot(all(whale$tilt_missing | abs(whale$tilt_deg) <= whale$tilt_limit_deg),
          all(!whale$tilt_relaxed | whale$n_usable_default_tilt == 0L),
          all(whale$quality_eligible | whale$n_quality_eligible_flight == 0L),
          all(whale$group_max_altitude_m - whale$group_min_altitude_m <= MAX_ALTITUDE_SPAN_M))
group_selection <- sel %>%
  group_by(animal, year, flight, altitude_group) %>%
  summarise(n_available = n(), n_selected = sum(selection_rank <= MAX_ANIMAL_FRAMES),
            group_mean_altitude_m = first(group_mean_altitude_m),
            group_min_altitude_m = first(group_min_altitude_m),
            group_max_altitude_m = first(group_max_altitude_m),
            selected_mean_altitude_m = mean(selection_altitude[selection_rank <= MAX_ANIMAL_FRAMES]),
            quality_fallback = first(quality_fallback),
            tilt_limit_deg = first(tilt_limit_deg),
            tilt_relaxed = first(tilt_relaxed),
            n_selected_above_default_tilt = sum(tilt_rescued & selection_rank <= MAX_ANIMAL_FRAMES),
            n_selected_missing_tilt = sum(tilt_missing & selection_rank <= MAX_ANIMAL_FRAMES),
            relaxed_tilt_image_ids = paste(image_id[tilt_rescued & selection_rank <= MAX_ANIMAL_FRAMES],
                                           collapse = "; "), .groups = "drop")
stopifnot(all(group_selection$n_selected <= MAX_ANIMAL_FRAMES))
cat("\n--- Selected frames by individual, flight and altitude group ---\n")
print(as.data.frame(group_selection), row.names = FALSE)
write.csv(group_selection, file.path(OUT_DIR, "selection_by_altitude_group.csv"), row.names = FALSE)

selection_audit <- whale_all %>%
  left_join(sel %>% select(animal, year, flight, image_id, altitude_group,
                           group_mean_altitude_m, selection_rank),
            by = c("animal", "year", "flight", "image_id")) %>%
  mutate(selected = !is.na(selection_rank) & selection_rank <= MAX_ANIMAL_FRAMES,
         selection_reason = case_when(
           !valid_altitude ~ "Invalid altitude",
           !valid_total_length ~ "Invalid total length",
           !tilt_allowed ~ "Excluded by the applied tilt limit",
           !eligible ~ "Excluded: quality-eligible frames are available in this flight",
           !selected ~ "Beyond the frame limit for this altitude group",
           quality_fallback ~ "Quality fallback: no quality-eligible frame in this flight",
           TRUE ~ "Quality-eligible frame selected"))
write.csv(selection_audit, file.path(OUT_DIR, "frame_selection_audit.csv"), row.names = FALSE)

cat("\n--- Selected frames by individual, year and flight ---\n")
flight_selection <- whale_all %>% group_by(animal, year, flight) %>%
  summarise(n_available = n(), tilt_limit_deg = first(tilt_limit_deg),
            tilt_relaxed = first(tilt_relaxed),
            n_usable_default_tilt = first(n_usable_default_tilt),
            quality_fallback = first(quality_fallback), .groups = "drop") %>%
  left_join(whale %>% group_by(animal, year, flight) %>%
              summarise(n_selected = n(), n_quality_eligible = sum(quality_eligible),
                        n_with_laser = sum(has_laser), n_with_eyes = sum(has_eyes),
                        n_selected_above_default_tilt = sum(tilt_rescued),
                        n_selected_missing_tilt = sum(tilt_missing), .groups = "drop"),
            by = c("animal", "year", "flight")) %>%
  mutate(across(c(n_selected, n_quality_eligible, n_with_laser, n_with_eyes,
                  n_selected_above_default_tilt, n_selected_missing_tilt), ~ coalesce(.x, 0L)))
print(as.data.frame(flight_selection), row.names = FALSE)
write.csv(flight_selection, file.path(OUT_DIR, "selection_by_flight.csv"), row.names = FALSE)

# 4b. CLASS-3 WIDTHS: FALLBACK ONLY
whale <- whale %>%
  mutate(class3_width_fallback = width_quality == POOR_WIDTH_QUALITY &
           quality_fallback & n_quality_eligible_flight == 0L,
         exclude_widths = !(width_quality %in% GOOD_WIDTH_QUALITIES | class3_width_fallback))
stopifnot(all(!whale$class3_width_fallback | whale$n_quality_eligible_flight == 0L))
if (any(whale$exclude_widths) && length(width_cols) > 0) {
  whale[whale$exclude_widths, width_cols] <- NA_real_
}
whale <- whale %>%
  mutate(across(all_of(width_cols), ~ ifelse(is.finite(.x) & .x > 0, .x, NA_real_)),
         has_usable_widths = rowSums(!is.na(across(all_of(width_cols)))) > 0L)

reliability <- whale %>%
  group_by(Subject = animal, Timepoint = as.character(year)) %>%
  summarise(n_frames_used = n(),
            n_width_frames = sum(has_usable_widths),
            n_quality_fallback_frames = sum(quality_fallback),
            n_class3_frames = sum(width_quality == POOR_WIDTH_QUALITY),
            n_class3_width_frames = sum(class3_width_fallback & has_usable_widths),
            width_quality_label = case_when(
              !any(has_usable_widths) ~ "No usable widths",
              all(width_quality[has_usable_widths] == POOR_WIDTH_QUALITY) ~ "Class 3 only (fallback)",
              any(class3_width_fallback & has_usable_widths) ~ "Classes 1-2 and class 3 (fallback)",
              TRUE ~ "Classes 1-2"),
            .groups = "drop") %>%
  mutate(reliability_reason = case_when(n_width_frames == 0 ~ "No usable widths",
                                        n_class3_width_frames > 0 ~ "Class-3 widths used: no quality-eligible frame in the flight",
                                        n_width_frames < 3 ~ "Fewer than 3 frames with usable widths",
                                        TRUE ~ NA_character_),
         reliable = is.na(reliability_reason))
write.csv(reliability, file.path(OUT_DIR, "reliability.csv"), row.names = FALSE)

# 4c. CALIBRATION FRAME SELECTION
# Apply adaptive tilt screening independently to calibration frames per flight.
cal_all <- dat %>% filter(kind == "cal") %>%
  mutate(calibration_ID = paste("CAL", drone, paste0(L_known, "m"), sep = "_"),
         CAL_px = px,
         calibration_altitude = coalesce(h_laser, h_baro_corr)) %>%
  filter(is.finite(CAL_px), CAL_px > 0, is.finite(L_known), L_known > 0,
         is.finite(calibration_altitude)) %>%
  mutate(tilt_missing = is.na(tilt_deg),
         tilt_pass_default = (tilt_missing & KEEP_MISSING_TILT) |
           (!tilt_missing & is.finite(tilt_deg) & abs(tilt_deg) <= MAX_TILT_DEG),
         tilt_pass_ceiling = (tilt_missing & KEEP_MISSING_TILT) |
           (!tilt_missing & is.finite(tilt_deg) & abs(tilt_deg) <= FALLBACK_TILT_DEG)) %>%
  group_by(flight) %>%
  mutate(tilt_relaxed = !any(tilt_pass_default),
         tilt_limit_deg = ifelse(tilt_relaxed, FALLBACK_TILT_DEG, MAX_TILT_DEG),
         tilt_allowed = ifelse(tilt_relaxed, tilt_pass_ceiling, tilt_pass_default)) %>%
  ungroup() %>% filter(tilt_allowed)
write.csv(cal_all %>% select(flight, image_id, tilt_deg, tilt_limit_deg, tilt_relaxed),
          file.path(OUT_DIR, "calibration_tilt_screening.csv"), row.names = FALSE)

# Build data-driven altitude groups after tilt screening, using the same
# minimum-anchored algorithm as for whale frames. Group means are descriptive.
cal_all <- cal_all %>%
  group_by(flight) %>%
  mutate(calibration_altitude_group = assign_altitude_groups(
    calibration_altitude, max_span = MAX_CALIBRATION_ALTITUDE_SPAN_M)) %>%
  ungroup() %>%
  group_by(flight, calibration_altitude_group) %>%
  mutate(group_mean_altitude_m = mean(calibration_altitude),
         group_min_altitude_m = min(calibration_altitude),
         group_max_altitude_m = max(calibration_altitude)) %>%
  arrange(desc(!is.na(h_laser)), image_id, .by_group = TRUE) %>%
  mutate(selection_rank = row_number(),
         selected = !REDUCE_CALIBRATION | selection_rank <= MAX_CALIBRATION_FRAMES) %>%
  ungroup()
cal <- cal_all %>% filter(selected)
stopifnot(all(cal_all$group_max_altitude_m - cal_all$group_min_altitude_m <=
                MAX_CALIBRATION_ALTITUDE_SPAN_M))

calibration_group_selection <- cal_all %>%
  group_by(flight, calibration_altitude_group) %>%
  summarise(n_available = n(), n_selected = sum(selected),
            group_mean_altitude_m = first(group_mean_altitude_m),
            group_min_altitude_m = first(group_min_altitude_m),
            group_max_altitude_m = first(group_max_altitude_m),
            selected_mean_altitude_m = mean(calibration_altitude[selected]),
            tilt_limit_deg = first(tilt_limit_deg),
            tilt_relaxed = first(tilt_relaxed),
            n_selected_above_default_tilt = sum(selected & !tilt_missing &
                                                  abs(tilt_deg) > MAX_TILT_DEG),
            n_selected_missing_tilt = sum(selected & tilt_missing),
            selected_image_ids = paste(image_id[selected], collapse = "; "),
            .groups = "drop")
write.csv(calibration_group_selection,
          file.path(OUT_DIR, "calibration_selection_by_altitude_group.csv"), row.names = FALSE)

if (REDUCE_CALIBRATION) {
  stopifnot(all(calibration_group_selection$n_selected <= MAX_CALIBRATION_FRAMES))
  cat(sprintf("\nCalibration: %d of %d frames, maximum %d per altitude group; span <= %g m\n",
              nrow(cal), nrow(cal_all), MAX_CALIBRATION_FRAMES,
              MAX_CALIBRATION_ALTITUDE_SPAN_M))
} else {
  cat(sprintf("\nCalibration: all %d frames (subsampling disabled)\n", nrow(cal)))
}
print(cal %>% count(drone, flight, calibration_altitude_group, name = "frame"))

whale
# 5. PREPARE XCERTAINTY INPUTS
whale <- whale %>% rename(TL_px = px)
for (w in intersect(width_cols, names(whale))) {
  pct <- as.numeric(gsub("width_|pct_px", "", w))
  names(whale)[names(whale) == w] <- sprintf("W%02d_px", pct)
}
width_names <- intersect(sprintf("W%02d_px", seq(5, 95, by = 5)), names(whale))
custom_cols <- intersect(c("eyes", "pec"), names(whale))
meas_cols   <- c("TL_px", width_names, custom_cols)

whale <- whale %>% mutate(across(all_of(meas_cols), ~ ifelse(!is.na(.x) & .x <= 0, NA_real_, .x)))

drop_na_measurements <- function(pd) {
  pd$pixel_counts <- pd$pixel_counts %>% filter(is.finite(PixelCount))
  po <- pd$pixel_counts %>% select(Subject, Measurement, Timepoint) %>% unique()
  if (!is.null(pd$training_objects))
    po <- setdiff(po, pd$training_objects %>% select(Subject, Measurement, Timepoint))
  pd$prediction_objects <- if (nrow(po) > 0) po else NULL
  pd
}

calibration_data <- parse_observations(
  x = cal, subject_col = "calibration_ID", meas_col = "CAL_px", tlen_col = "L_known",
  image_col = "image_id", barometer_col = "h_baro_corr", laser_col = "h_laser",
  flen_col = "FOCAL_LENGTH_MM", iwidth_col = "IMAGE_WIDTH_PX",
  swidth_col = "SENSOR_WIDTH_MM", uas_col = "drone", timepoint_col = "year"
) %>% drop_na_measurements()

whale_data <- parse_observations(
  x = whale, subject_col = "animal", meas_col = meas_cols,
  image_col = "image_id", barometer_col = "h_baro_corr", laser_col = "h_laser",
  flen_col = "FOCAL_LENGTH_MM", iwidth_col = "IMAGE_WIDTH_PX",
  swidth_col = "SENSOR_WIDTH_MM", uas_col = "drone", timepoint_col = "year"
) %>% drop_na_measurements()

xdata <- combine_observations(calibration_data, whale_data)

# 6. PRIOR DISTRIBUTIONS
altimeter_types <- if (any(!is.na(dat$h_laser))) c("Barometer", "Laser") else "Barometer"
# Bounds and prior parameters are unchanged from the previous analysis.
priors <- list(
  image_altitude     = c(min = H_MIN, max = H_MAX),
  altimeter_bias     = data.frame(altimeter = altimeter_types, mean = 0, sd = 1e2),
  altimeter_variance = data.frame(altimeter = altimeter_types, shape = 0.01, rate = 0.01),
  altimeter_scaling  = data.frame(altimeter = altimeter_types, mean = 1, sd = 10),
  pixel_variance     = c(shape = 0.01, rate = 0.01),
  object_lengths     = c(min = 0.01, max = 20)
)

# 7. MCMC SAMPLING
# If the chains of a previous identical run are already on disk, they can be
# reused: the posterior samples do not change, so there is no reason to spend
# hours re-running the MCMC just to compute a second body region. Set
REUSE_CHAINS <- FALSE #(or delete the file) to force a fresh run. WARNING:
# reuse only if the data and the selection rules above are unchanged, otherwise
# the saved chains no longer match the data object built here.
CHAINS_FILE <- file.path("/Users/matteotosato/Desktop/codice_tesi/Xcertainty_ind_length_tesi_chains.rds")

if (REUSE_CHAINS && file.exists(CHAINS_FILE)) {
  cat("\n=== SAMPLER SKIPPED: reusing", CHAINS_FILE, "===\n")
  chains <- readRDS(CHAINS_FILE)
  stopifnot(length(chains) == 3)
  fit <- chains[[1]]
} else {
  cat("\n=== SAMPLER ===\n")
  sampler <- independent_length_sampler(data = xdata, priors = priors, package_only = FALSE)
  run_chain <- function(seed) { set.seed(seed); sampler(niter = N_ITER, thin = N_THIN, summary.burn = 0.5) }
  
  t0 <- Sys.time()
  chains <- lapply(seq_len(N_CHAINS), function(i) { cat(sprintf("Chain %d/%d\n", i, N_CHAINS)); run_chain(100 + i) })
  cat(sprintf("MCMC completed in %.1f minutes.\n", as.numeric(difftime(Sys.time(), t0, units = "mins"))))
  
  # Summaries retain the original choice of the first chain.
  fit <- chains[[1]]
  saveRDS(chains, CHAINS_FILE)
}
#saveRDS(chains, file.path(OUT_DIR, "Xcertainty_ind_length_chains.rds"))
#saveRDS(chains, file.path(OUT_DIR, "Xcertainty_ind_length_tesi_chains.rds"))

ess <- fit$summaries$objects$ESS
cat(sprintf("\nESS  min %.0f | median %.0f | max %.0f\n", min(ess), median(ess), max(ess)))

if (N_CHAINS > 1) {
  obj_names <- names(fit$objects)[grepl("TL_px", names(fit$objects))]
  rhat <- sapply(obj_names, function(nm) {
    ml <- mcmc.list(lapply(chains, function(ch) {
      s <- ch$objects[[nm]]$samples; mcmc(s[floor(length(s) * 0.5):length(s)])
    }))
    gelman.diag(ml, autoburnin = FALSE)$psrf[1, 1]
  })
  cat(sprintf("Total-length R-hat: max %.3f (threshold %.2f) - %d/%d subjects above threshold\n",
              max(rhat), 1.05, sum(rhat > 1.05), length(rhat)))
  write.csv(data.frame(object = obj_names, rhat = rhat),
            file.path(OUT_DIR, "Xcertainty_rhat.csv"), row.names = FALSE)
}

obj_names <- names(fit$objects)[grepl("TL_px", names(fit$objects))]


cat("\n=== ALTIMETERS: estimated bias and scaling ===\n")
print(as.data.frame(fit$summaries$altimeters %>% mutate(across(where(is.numeric), ~ round(.x, 4)))),
      row.names = FALSE)
write.csv(fit$summaries$altimeters, file.path(OUT_DIR, "altimeters.csv"), row.names = FALSE)
