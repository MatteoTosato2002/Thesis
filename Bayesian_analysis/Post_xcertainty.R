library(Xcertainty)
library(dplyr)
library(tidyr)
library(ggplot2)
library(coda)

theme_set(theme_classic(base_size = 12))
OUT_DIR <- "/Users/matteotosato/Desktop/codice_tesi"
# These two functions avoid repeating the path every time a file is saved.
save_table <- function(table, name) {
  write.csv(table, file.path(OUT_DIR, name), row.names = FALSE)
}
save_plot <- function(plot, name, width = 8, height = 5) {
  ggsave(file.path(OUT_DIR, name), plot,
         width = width, height = height, dpi = 300)
}

# 1. MEASUREMENTS AND UNCERTAINTY
# Thesis: Uncertainty diagnostics; Relative uncertainty by UAS and flight altitude.
# Each row of meta describes one individual-year.
meta <- whale %>%
  mutate(altitude = coalesce(h_laser, h_baro_corr)) %>%
  group_by(Subject = animal, Timepoint = as.character(year)) %>%
  summarise(n_images = n_distinct(image_id), n_flights = n_distinct(flight),
            flight_ids = paste(sort(unique(flight)), collapse = ", "),
            drones = paste(sort(unique(drone)), collapse = ", "),
            mean_flight_altitude = mean(altitude), .groups = "drop")

results <- fit$summaries$objects %>%
  mutate(Timepoint = as.character(Timepoint)) %>%
  left_join(meta, by = c("Subject", "Timepoint")) %>%
  left_join(reliability, by = c("Subject", "Timepoint")) %>%
  mutate(CV_pct = 100 * sd / mean, HPD_width = upper - lower)
save_table(results, "Xcertainty_all_measurements.csv")

# Save the frames actually used: they will be needed for W/TL and W/EW analyses.
# TL_px and Wxx_px in this table are in PIXELS; in results they are in METRES.
whale
selected_frames <- whale %>%
  select(animal, year, image_id, flight, drone, TL_px, eyes,
         all_of(sprintf("W%02d_px", seq(5, 95, 5))))
save_table(selected_frames, "selected_whale_measurements.csv")

lengths <- results %>% filter(Measurement == "TL_px") %>%
  mutate(label = paste0(Subject, " (", Timepoint, ")"))
save_table(lengths, "total_length_by_individual.csv")
length_summary <- lengths %>%
  summarise(n = n(), mean_length = mean(mean), sd_length = sd(mean),
            median_length = median(mean), min_length = min(mean),
            max_length = max(mean), mean_CV = mean(CV_pct),
            median_CV = median(CV_pct), min_CV = min(CV_pct), max_CV = max(CV_pct))
save_table(length_summary, "total_length_summary.csv")

p_length <- ggplot(lengths, aes(mean, reorder(label, mean), colour = drones)) +
  geom_errorbar(aes(xmin = lower, xmax = upper), orientation = "y", width = .15) +
  geom_point(size = 2.5) +
  labs(x = "Total length (m): posterior mean and 95% HPD", y = NULL,
       colour = "Drone configuration")
p_length
save_plot(p_length, "total_length.png", 8, 8)

# Drone combinations remain separate groups: an estimate obtained using
# two drones is not counted twice under the individual drones.

# Altimeter availability for each individual-year
altimeter_groups <- whale %>%
  group_by(Subject = animal, Timepoint = as.character(year)) %>%
  summarise(
    altimeter_group = ifelse(
      any(!is.na(h_laser)),
      "Barometer + laser",
      "Barometer only"
    ),
    .groups = "drop"
  )

cv_plot_data <- lengths %>%
  left_join(altimeter_groups, by = c("Subject", "Timepoint"))

means <- cv_plot_data %>%
  group_by(drones) %>%
  summarise(mean_cv = mean(CV_pct), highest_cv = max(CV_pct), 
            highest_cv_flights = flight_ids[which.max(CV_pct)], .groups = "drop"
  )
means

# CV by drone, with points coloured by altimeter availability
p_cv <- ggplot(cv_plot_data, aes(drones, CV_pct)) +
  geom_boxplot(outlier.shape = NA) +
  geom_jitter(
    aes(colour = altimeter_group),
    width = .08,
    height = 0,
    size = 2.5
  ) +
  scale_colour_manual(
    values = c(
      "Barometer + laser" = "#2b5c8f",
      "Barometer only" = "#d97706"
    )
  ) +
  labs(
    x = "Drone configuration",
    y = "Posterior CV of total length (%)",
    colour = "Altimeter availability"
  )

p_cv
save_plot(p_cv, "CV_by_drone.png")

altitude_by_drone <- lengths %>% group_by(drones) %>%
  summarise(n = n(), mean_altitude = mean(mean_flight_altitude),
            sd_altitude = sd(mean_flight_altitude),
            median_altitude = median(mean_flight_altitude),
            min_altitude = min(mean_flight_altitude),
            max_altitude = max(mean_flight_altitude), .groups = "drop")
altitude_by_drone
save_table(altitude_by_drone, "mean_altitude_by_drone.csv")

# 2. ALTIMETERS
# Thesis: Altimeter performance.
altimeters <- fit$summaries$altimeters %>%
  filter(parameter %in% c("bias", "scaling")) %>%
  mutate(altimeter_id = paste(UAS, altimeter, sep = " - "))

save_table(altimeters, "altimeters.csv")

references <- data.frame(
  parameter = c("bias", "scaling"),
  value = c(0, 1)
)

p_altimeters <- ggplot(
  altimeters,
  aes(mean, altimeter_id, colour = altimeter)
) +
  geom_errorbar(
    aes(xmin = lower, xmax = upper),
    orientation = "y",
    width = .15
  ) +
  geom_point(size = 2.5) +
  geom_vline(
    data = references,
    aes(xintercept = value),
    linetype = 2
  ) +
  facet_wrap(
    ~ parameter,
    scales = "free_x",
    labeller = as_labeller(c(
      bias = "Additive bias (m)",
      scaling = "Scaling factor"
    ))
  ) +
  labs(
    x = "Posterior mean and 95% HPD",
    y = NULL,
    colour = "Altimeter"
  )

altimeters
p_altimeters
save_plot(p_altimeters, "altimeter_parameters.png", 10, 5)

altitude_observations <- cal %>%
  transmute(Image = image_id, UAS = drone,
            Barometer = h_baro_corr, Laser = h_laser) %>%
  distinct() %>%
  pivot_longer(c(Barometer, Laser), names_to = "Altimeter",
               values_to = "observed_altitude") %>%
  filter(!is.na(observed_altitude))
altitude_differences <- fit$summaries$images %>%
  select(Image, posterior_altitude = mean, lower, upper) %>%
  inner_join(altitude_observations, by = "Image") %>%
  mutate(difference = posterior_altitude - observed_altitude)
save_table(altitude_differences, "altitude_differences_by_image.csv")
altitude_summary <- altitude_differences %>% group_by(UAS, Altimeter) %>%
  summarise(n = n(), mean_difference = mean(difference),
            median_difference = median(difference), sd_difference = sd(difference),
            MAE = mean(abs(difference)), RMSE = sqrt(mean(difference^2)),
            .groups = "drop")
save_table(altitude_summary, "altitude_differences_summary.csv")
altitude_differences
altitude_summary
p_altitude <- ggplot(altitude_differences,
                     aes(observed_altitude, difference, colour = Altimeter)) +
  geom_hline(yintercept = 0, linetype = 2) + geom_point(alpha = .5) +
  geom_smooth(method = "loess", se = TRUE) + facet_wrap(~ UAS) +
  labs(x = "Observed altitude (m)", y = "Posterior minus observed altitude (m)")
p_altitude
save_plot(p_altitude, "altitude_differences.png", 10, 5)

single_drone <- cv_plot_data %>%
  filter(!grepl(",", drones))

p_altitude_cv <- ggplot(
  single_drone,
  aes(
    x = mean_flight_altitude,
    y = CV_pct,
    colour = drones
  )
) +
  geom_point(size = 3.5, alpha = .85) +
  scale_colour_manual(
    values = c(
      "Inspire" = "#2b5c8f",
      "Mavic" = "#d97706",
      "Matrice" = "#328653"
    )
  ) +
  labs(
    x = "Mean flight altitude (m)",
    y = "Posterior CV of total length (%)",
    colour = "Drone",
  ) +
  theme_classic(base_size = 13) +
  theme(legend.position = "bottom")

p_altitude_cv
save_plot(p_altitude_cv, "CV_vs_altitude.png")

# 3. TILT
# Thesis: Drone tilt uncertainty. Descriptive comparison, without multiple regression.
# Classes are defined using AVAILABLE tilt; n_missing_tilt also identifies
# individual-years for which tilt was recorded for only some frames.
tilt <- whale %>%
  group_by(Subject = animal, Timepoint = as.character(year)) %>%
  summarise(n_tilt = sum(!is.na(tilt_deg)), n_missing_tilt = sum(is.na(tilt_deg)),
            n_above10 = sum(abs(tilt_deg) > 10, na.rm = TRUE), .groups = "drop") %>%
  mutate(tilt_class = case_when(
    n_tilt == 0 ~ "Tilt unavailable",
    n_above10 == 0 ~ "<= 10 degrees",
    n_above10 == n_tilt ~ "> 10 degrees",
    TRUE ~ "Mixed recorded tilt")) %>%
  left_join(lengths, by = c("Subject", "Timepoint"))
save_table(tilt, "tilt_by_individual.csv")
tilt_summary <- tilt %>% group_by(tilt_class) %>%
  summarise(n = n(), mean_CV = mean(CV_pct), sd_CV = sd(CV_pct),
            median_CV = median(CV_pct), Q1_CV = quantile(CV_pct, .25),
            Q3_CV = quantile(CV_pct, .75), .groups = "drop")
save_table(tilt_summary, "CV_by_tilt.csv")
p_tilt <- ggplot(tilt, aes(tilt_class, CV_pct)) +
  geom_boxplot(outlier.shape = NA) + geom_jitter(width = .08, height = 0) +
  labs(x = "Recorded tilt", y = "Posterior CV of total length (%)")
p_tilt
tilt_summary
save_plot(p_tilt, "CV_by_tilt.png")

# 4. BAI FROM POSTERIOR DISTRIBUTIONS
# Thesis: Body Area Index estimates. BAI analyses are in file 02.
compute_bai <- function(lo, hi, tag) {
  sections <- seq(lo, hi, 5)
  bc <- body_condition(data = whale_data, output = fit, length_name = "TL_px",
                       width_names = sprintf("W%02d_px", sections),
                       width_increments = sections, summary.burn = .5,
                       metric = c("surface_area", "body_area_index", "standardized_widths"))
  saveRDS(bc, file.path(OUT_DIR, paste0("body_condition", tag, ".rds")))
  for (metric in c("surface_area", "body_area_index", "standardized_widths")) {
    table <- bc$summaries[[metric]] %>%
      mutate(Timepoint = as.character(Timepoint)) %>%
      left_join(meta, by = c("Subject", "Timepoint")) %>%
      left_join(reliability, by = c("Subject", "Timepoint"))
    save_table(table, paste0("Xcertainty_", metric, tag, ".csv"))
  }
  bc
}
bc <- compute_bai(35, 80, "")
bc_head <- compute_bai(5, 35, "_head")
fit$objects[["Pm1_23 TL_px 2023"]]$summary
# 5. VOLUME FROM POSTERIOR DISTRIBUTIONS
# Thesis: Body volume. Height-to-width ratios already present in the original code.
# Use samples from the SAME iteration for TL and all widths.
posterior_samples <- function(subject, year, measurement) {
  samples <- fit$objects[[paste(subject, measurement, year)]]$samples
  first <- floor(length(samples) / 2) + 1
  samples[first:length(samples)]
}
height_width <- c(1.72, 1.48, 1.38, 1.27, 1.21, 1.19, 1.15, 1.14, 1.19,
                  1.29, 1.42, 1.55, 1.64, 1.72, 2.09, 2.41, 2.50)
sections <- seq(5, 85, 5)
volume_rows <- list()
meta
for (i in seq_len(nrow(meta))) {
  subject <- meta$Subject[i]
  year <- meta$Timepoint[i]
  TL <- posterior_samples(subject, year, "TL_px")
  widths <- matrix(0, nrow = length(TL), ncol = 21)
  heights <- widths
  for (j in seq_along(sections)) {
    widths[, j + 1] <- posterior_samples(subject, year, sprintf("W%02d_px", sections[j]))
    heights[, j + 1] <- widths[, j + 1] * height_width[j]
  }
  # Columns: 1 = 0%, 18 = 85%, 19 = 90%, 20 = 95%, 21 = 100%.
  widths[, 19] <- widths[, 18] * 2 / 3
  widths[, 20] <- widths[, 18] / 3
  heights[, 19] <- heights[, 18] * 2 / 3
  heights[, 20] <- heights[, 18] / 3
  volume <- rep(0, length(TL))
  for (j in 1:20) {
    WA <- widths[, j]
    WP <- widths[, j + 1]
    HA <- heights[, j]
    HP <- heights[, j + 1]
    volume <- volume + pi / 4 * .05 * TL *
      ((WA * HA + WP * HP) / 3 + (WA * HP + WP * HA) / 6)
  }
  interval <- HPDinterval(mcmc(volume), prob = .95)
  volume_rows[[i]] <- data.frame(Subject = subject, Timepoint = year,
                                 mean = mean(volume), sd = sd(volume), lower = interval[1, "lower"],
                                 upper = interval[1, "upper"], TL_mean = mean(TL), n_draws = length(volume))
}

body_volume <- bind_rows(volume_rows) %>%
  left_join(meta, by = c("Subject", "Timepoint")) %>%
  left_join(reliability, by = c("Subject", "Timepoint"))
save_table(body_volume, "Xcertainty_body_volume_glarou.csv")

# 6. APPROXIMATE LENGTH-DERIVED AGE
# Thesis: Length-age relationship and length-derived age estimates.
# These intervals propagate only TL uncertainty, not uncertainty in the curve.

age_results <- lengths %>%
  mutate(age_estimate = (mean - 7.3129) / .1759,
         age_lower = (lower - 7.3129) / .1759,
         age_upper = (upper - 7.3129) / .1759,
         maturity = case_when(
           mean >= 10.5 & lower >= 10.5 & age_estimate >= 10 ~ "Likely sexually mature",
           mean >= 10.5 & lower < 10.5 & age_estimate >= 10 ~ "Possibly mature",
           mean < 10.5 & age_estimate >= 10 & age_lower >= 10 ~ "Possibly immature",
           mean < 10.5 & age_estimate >= 10  & age_lower < 10 ~ "Likely immature",
         ),
         length_threshold = case_when(
           upper < 10.5 ~ "Below 10.5 m",
           lower >= 10.5 ~ "Above 10.5 m",
           TRUE ~ "HPD overlaps 10.5 m"))
save_table(age_results, "age_estimates_from_length.csv")

age_results <- age_results %>%
  mutate(
    maturity = factor(
      maturity,
      levels = c(
        "Likely sexually mature",
        "Possibly mature",
        "Possibly immature",
        "Likely immature"
      )
    )
  )

save_table(count(age_results, maturity), "maturity_summary.csv")
p_age <- ggplot(age_results,
                aes(age_estimate, reorder(label, age_estimate), colour = maturity)) +
  geom_errorbar(aes(xmin = age_lower, xmax = age_upper), orientation = "y", width = .15) +
  geom_point(size = 2.5) + geom_vline(xintercept = 10, linetype = 2) +
  labs(x = "Length-derived age (years)", y = NULL, colour = "Exploratory classification")
p_age
save_plot(p_age, "length_derived_age.png", 9, 8)

print(length_summary)
print(cv_by_drone)
print(altitude_summary)
print(tilt_summary)
cat("\nEstimates and diagnostics saved in:", OUT_DIR, "\n")