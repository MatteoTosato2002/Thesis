library(Xcertainty)
library(dplyr)
library(ggplot2)
library(coda)

GROWTH_DIR <- file.path("/Users/matteotosato/Desktop/codice_tesi/individui_rivisti")
dir.create(GROWTH_DIR, recursive = TRUE, showWarnings = FALSE)

# 1. THE THREE INDIVIDUALS OBSERVED IN DIFFERENT YEARS
repeat_ids <- c("Pm1_23", "Pm2_23", "Pm2_25")
whale_growth <- whale %>% filter(animal %in% repeat_ids)
observations <- whale_growth %>% group_by(animal, year) %>%
  summarise(n_images = n_distinct(image_id), first_date = min(as.Date(date)),
            last_date = max(as.Date(date)), .groups = "drop")
write.csv(observations, file.path(GROWTH_DIR, "observations_by_year.csv"), row.names = FALSE)

# 2. NON-DECREASING MODEL: TOTAL LENGTH ONLY
# Calibration and priors are the same as in the main model.
whale_growth_data <- parse_observations(
  x = whale_growth, subject_col = "animal", meas_col = "TL_px",
  image_col = "image_id", barometer_col = "h_baro_corr", laser_col = "h_laser",
  flen_col = "FOCAL_LENGTH_MM", iwidth_col = "IMAGE_WIDTH_PX",
  swidth_col = "SENSOR_WIDTH_MM", uas_col = "drone", timepoint_col = "year")
xdata_growth <- combine_observations(calibration_data, whale_growth_data)
sampler_nd <- nondecreasing_length_sampler(data = xdata_growth,
                                           priors = priors, package_only = FALSE)
chains_nd <- list()
for (i in seq_len(N_CHAINS)) {
  set.seed(500 + i)
  cat("\nNon-decreasing chain", i, "of", N_CHAINS, "\n")
  chains_nd[[i]] <- sampler_nd(niter = N_ITER, thin = N_THIN, summary.burn = .5)
}
saveRDS(chains_nd, file.path(OUT_DIR, "Xcertainty_nondecreasing_TL_chains.rds"))

# To reuse previously saved chains, replace section 2 with this line:
chains_nd <- readRDS(file.path("/Users/matteotosato/Desktop/codice_tesi/Xcertainty_nondecreasing_TL_chains.rds"))
fit_nd <- chains_nd[[1]]

# 3. DIAGNOSTICS OF THE NEW MODEL
# R-hat compares the chains; ESS is calculated across chains after burn-in.
# We do not recalculate the diagnostics of the main model.
diagnostic_rows <- list()
for (name in names(fit_nd$objects)) {
  chain_samples <- list()
  for (i in seq_along(chains_nd)) {
    samples <- chains_nd[[i]]$objects[[name]]$samples
    first <- floor(length(samples) / 2) + 1
    chain_samples[[i]] <- mcmc(samples[first:length(samples)])
  }
  mcmc_chains <- mcmc.list(chain_samples)
  diagnostic_rows[[name]] <- data.frame(object = name,
                                        Rhat = gelman.diag(mcmc_chains, autoburnin = FALSE)$psrf[1, 1],
                                        ESS = unname(effectiveSize(mcmc_chains)[1]))
}
mcmc_chains <- mcmc.list(chain_samples)[[1]]
Rhat = gelman.diag(mcmc_chains, autoburnin = FALSE)$psrf[1,]
ESS = unname(effectiveSize(mcmc_chains)[1])
ESS
diagnostics_nd <- bind_rows(diagnostic_rows)
write.csv(diagnostics_nd, file.path(GROWTH_DIR, "nondecreasing_diagnostics.csv"), row.names = FALSE)
print(diagnostics_nd)

# 4. COMPARISON OF INDEPENDENT AND NON-DECREASING ESTIMATES
# Estimates still use the first chain, as in the original code.
independent <- results %>%
  filter(Measurement == "TL_px", Subject %in% repeat_ids) %>%
  transmute(Subject, Year = as.integer(Timepoint), model = "Independent",
            mean, sd, lower, upper)
nondecreasing <- fit_nd$summaries$objects %>% filter(Measurement == "TL_px") %>%
  transmute(Subject, Year = as.integer(Timepoint), model = "Non-decreasing",
            mean, sd, lower, upper)
length_comparison <- bind_rows(independent, nondecreasing) %>% arrange(Subject, Year, model)
write.csv(length_comparison, file.path(GROWTH_DIR, "lengths_both_models.csv"), row.names = FALSE)
p_lengths <- ggplot(length_comparison, aes(Year, mean, colour = model, group = model)) +
  geom_errorbar(aes(ymin = lower, ymax = upper), width = .06) +
  geom_line() + geom_point(size = 2.5) + facet_wrap(~ Subject, scales = "free_y") +
  scale_x_continuous(breaks = sort(unique(length_comparison$Year))) +
  labs(x = "Year", y = "Total length (m)", colour = "Model") + theme_classic()
p_lengths
ggsave(file.path(GROWTH_DIR, "lengths_both_models.png"), p_lengths, width = 10, height = 4, dpi = 300)

# 5. ANNUAL LENGTH CHANGES FROM POSTERIOR DISTRIBUTIONS
# This calculation was already in the original code, but is not explicitly described
# in the resighting paragraph: see the map before using it in the Results.
# Use the difference between years, as before: this is not the exact time between dates.
# Samples from the two years are paired by iteration, without reshuffling.
models <- list(Independent = fit, Nondecreasing = fit_nd)
growth_rows <- list()
row <- 0
for (model_name in names(models)) {
  output <- models[[model_name]]
  for (subject in repeat_ids) {
    years <- sort(unique(whale_growth$year[whale_growth$animal == subject]))
    year1 <- min(years)
    year2 <- max(years)
    samples1 <- output$objects[[paste(subject, "TL_px", year1)]]$samples
    samples2 <- output$objects[[paste(subject, "TL_px", year2)]]$samples
    first <- floor(length(samples1) / 2) + 1
    retained <- first:length(samples1)
    change <- samples2[retained] - samples1[retained]
    growth <- change / (year2 - year1)
    interval <- HPDinterval(mcmc(growth), prob = .95)
    row <- row + 1
    growth_rows[[row]] <- data.frame(Subject = subject, model = model_name,
                                     first_year = year1, last_year = year2, years_between = year2 - year1,
                                     mean_change_m = mean(change), mean_growth_m_y = mean(growth),
                                     median_growth_m_y = median(growth), lower_95 = interval[1, "lower"],
                                     upper_95 = interval[1, "upper"], P_growth_positive = mean(growth > 0))
  }
}
growth_results <- bind_rows(growth_rows)
write.csv(growth_results, file.path(GROWTH_DIR, "growth_both_models.csv"), row.names = FALSE)
# The table is created BEFORE the plot that uses it.
p_growth <- ggplot(growth_results, aes(mean_growth_m_y, Subject, colour = model)) +
  geom_vline(xintercept = 0, linetype = 2) +
  geom_errorbar(aes(xmin = lower_95, xmax = upper_95), orientation = "y", width = .15,
                position = position_dodge(width = .4)) +
  geom_point(position = position_dodge(width = .4), size = 2.5) +
  labs(x = "Annual length change (m/year): mean and 95% HPD", y = NULL) + theme_classic()
p_growth
ggsave(file.path(GROWTH_DIR, "growth_both_models.png"), p_growth, width = 8, height = 4, dpi = 300)
print(growth_results)
# In the non-decreasing model, a negative change is ruled out by the constraint:
# P_growth_positive is not independent confirmation that the animal has grown.


# Select annual growth estimates from the non-decreasing model
growth_comparison <- growth_results %>%
  filter(model == "Nondecreasing") %>%
  mutate(
    reference_growth = 0.1759,
    label = paste0(Subject, " (", first_year, "–", last_year, ")")
  )

# Build the reference trajectory from the first length estimate
growth_curve <- nondecreasing %>%
  group_by(Subject) %>%
  arrange(Year, .by_group = TRUE) %>%
  mutate(
    first_year = first(Year),
    first_length = first(mean),
    expected_length = first_length + 0.1759 * (Year - first_year)
  ) %>%
  ungroup()

growth_curve
write.csv(growth_curve, file.path(GROWTH_DIR, "growth_theoretical_vs_observed.csv"), row.names = FALSE)


# Compare non-decreasing lengths with the reference trajectory
p_growth_curve <- ggplot(nondecreasing, aes(Year, mean)) +
  geom_errorbar(
    aes(ymin = lower, ymax = upper),
    width = 0.06,
    colour = "steelblue4"
  ) +
  geom_line(
    aes(colour = "Non-decreasing"),
    linewidth = 1
  ) +
  geom_point(
    aes(colour = "Non-decreasing"),
    size = 2.5
  ) +
  geom_line(
    data = growth_curve,
    aes(y = expected_length, colour = "Maio et al. (2022)"),
    linewidth = 1,
    linetype = 2
  ) +
  geom_point(
    data = growth_curve,
    aes(y = expected_length, colour = "Maio et al. (2022)"),
    size = 2.5,
    shape = 17
  ) +
  facet_wrap(~ Subject, scales = "free_y") +
  scale_x_continuous(
    breaks = sort(unique(nondecreasing$Year))
  ) +
  scale_colour_manual(
    values = c(
      "Non-decreasing" = "steelblue4",
      "Maio et al. (2022)" = "darkorange3"
    )
  ) +
  labs(
    x = "Year",
    y = "Total length (m)",
    colour = NULL
  ) +
  theme_classic(base_size = 12) +
  theme(legend.position = "bottom")

p_growth_curve

nondecreasing

ggsave(
  file.path(GROWTH_DIR, "Lengths_vs_Maio_curve.png"),
  p_growth_curve,
  width = 10,
  height = 4,
  dpi = 300
)