library(dplyr)
library(tidyr)
library(ggplot2)
library(smatr)

RES_DIR <- ""
ANALYSIS_DIR <- file.path(RES_DIR, "")
dir.create(ANALYSIS_DIR, recursive = TRUE, showWarnings = FALSE)
CONTROL <- "Pm3_26"
theme_set(theme_classic(base_size = 12))

save_table <- function(table, name) {
  write.csv(table, file.path(ANALYSIS_DIR, name), row.names = FALSE)
}
save_plot <- function(plot, name, width = 8, height = 5) {
  ggsave(file.path(ANALYSIS_DIR, name), plot,
         width = width, height = height, dpi = 300)
}

# 1. DATA: POSTERIOR MEANS IN METRES AND FRAMES IN PIXELS
measurements <- read.csv(file.path(RES_DIR, "Xcertainty_all_measurements.csv")) %>%
  mutate(Timepoint = as.character(Timepoint))
whales <- measurements %>%
  select(Subject, Timepoint, Measurement, mean) %>%
  pivot_wider(names_from = Measurement, values_from = mean) %>%
  rename_with(~ sub("_px$", "", .x)) %>%
  left_join(measurements %>% filter(Measurement == "TL_px") %>%
              select(Subject, Timepoint, n_images, drones, reliable),
            by = c("Subject", "Timepoint")) %>%
  mutate(id = paste0(Subject, " (", Timepoint, ")"),
         case_study = Subject == CONTROL)

trunk <- read.csv(file.path(RES_DIR, "Xcertainty_body_area_index.csv")) %>%
  transmute(Subject, Timepoint = as.character(Timepoint),
            BAI_trunk = mean, trunk_lower = lower, trunk_upper = upper)
head <- read.csv(file.path(RES_DIR, "Xcertainty_body_area_index_head.csv")) %>%
  transmute(Subject, Timepoint = as.character(Timepoint),
            BAI_head = mean, head_lower = lower, head_upper = upper)
volume <- read.csv(file.path(RES_DIR, "Xcertainty_body_volume_glarou.csv")) %>%
  transmute(Subject, Timepoint = as.character(Timepoint),
            Volume = mean, volume_lower = lower, volume_upper = upper)
whales <- whales %>% left_join(trunk, by = c("Subject", "Timepoint")) %>%
  left_join(head, by = c("Subject", "Timepoint")) %>%
  left_join(volume, by = c("Subject", "Timepoint"))

frames <- read.csv(file.path(RES_DIR, "selected_whale_measurements.csv")) %>%
  mutate(Subject = animal, Timepoint = as.character(year),
         id = paste0(Subject, " (", Timepoint, ")"))
width_names <- sprintf("W%02d_px", seq(5, 95, 5))

# DESCRIPTIVE PLOTS OF THE TWO BODY AREA INDICES

# Trunk BAI: 35-80% of total length
p_bai_trunk <- ggplot(
  whales,
  aes(BAI_trunk, reorder(id, BAI_trunk))
) +
  geom_errorbar(
    aes(xmin = trunk_lower, xmax = trunk_upper),
    orientation = "y",
    width = .15
  ) +
  geom_point(size = 2.5) +
  labs(
    x = "Trunk BAI (35-80% TL)",
    y = NULL,
    subtitle = "Posterior means and 95% HPD intervals"
  ) +
  theme_classic(base_size = 12)

p_bai_trunk
save_plot(p_bai_trunk, "BAI_trunk_estimates.png", 8, 8)


# Head BAI: 5-35% of total length
p_bai_head <- ggplot(
  whales,
  aes(BAI_head, reorder(id, BAI_head))
) +
  geom_errorbar(
    aes(xmin = head_lower, xmax = head_upper),
    orientation = "y",
    width = .15
  ) +
  geom_point(size = 2.5) +
  labs(
    x = "Head BAI (5-35% TL)",
    y = NULL,
    subtitle = "Posterior means and 95% HPD intervals"
  ) +
  theme_classic(base_size = 12)

p_bai_head
save_plot(p_bai_head, "BAI_head_estimates.png", 8, 8)


# 2. BAI AND VOLUME: SIZE DEPENDENCE AND CORRECTION
# Thesis: Body Area Index analyses; Body volume.
# Use ln(index) = a + b * ln(TL). exp(fitted()) returns the expected value
# to the original scale. SCI = 100 * (observed / expected - 1).
# The same short function is used for BAI, volume and EI.
size_analysis <- function(values, TL) {
  model <- lm(log(values) ~ log(TL))
  expected <- exp(fitted(model))
  interval <- confint(model)[2, ]
  coefficients <- summary(model)$coefficients
  table <- data.frame(n = length(values), a = unname(coef(model)[1]),
                      b = unname(coef(model)[2]), SE_b = coefficients[2, 2],
                      lower_b = interval[1], upper_b = interval[2],
                      p_b0 = coefficients[2, 4], R2 = summary(model)$r.squared)
  list(model = model, table = table, expected = expected,
       SCI = 100 * (values / expected - 1))
}

size_rows <- list()
for (metric in c("BAI_trunk", "BAI_head", "Volume")) {
  analysis <- size_analysis(whales[[metric]], whales$TL)
  whales[[paste0(metric, "_expected")]] <- analysis$expected
  whales[[paste0(metric, "_SCI")]] <- analysis$SCI
  size_rows[[metric]] <- data.frame(metric = metric, analysis$table)
}
size_table <- bind_rows(size_rows)
# For volume, the isometric hypothesis is b = 3; for BAI, independence
# from size corresponds to b = 0. p_b0 remains the standard lm() test.
size_table <- size_table %>%
  mutate(reference_b = ifelse(metric == "Volume", 3, 0),
         reference_in_CI = lower_b <= reference_b & upper_b >= reference_b)
save_table(size_table, "02_size_dependence_BAI_volume.csv")

# Relationship between body area indices and total length on the ln–ln scale
bai_size <- whales %>%
  select(id, TL, BAI_trunk, BAI_head) %>%
  pivot_longer(
    cols = c(BAI_trunk, BAI_head),
    names_to = "region",
    values_to = "BAI"
  ) %>%
  mutate(
    ln_TL = log(TL),
    ln_BAI = log(BAI),
    region = factor(
      region,
      levels = c("BAI_trunk", "BAI_head"),
      labels = c("Trunk (35–80% TL)", "Head (5–35% TL)")
    )
  )

p_bai_size <- ggplot(bai_size, aes(ln_TL, ln_BAI)) +
  geom_point(size = 2.5) +
  geom_smooth(
    method = "lm",
    formula = y ~ x,
    se = TRUE,
    colour = "black",
    fill = "grey80"
  ) +
  facet_wrap(~ region, scales = "free_y") +
  labs(
    x = "ln(Total length [m])",
    y = "ln(BAI)"
  ) +
  theme_classic(base_size = 12)

p_bai_size
save_plot(p_bai_size, "BAI_vs_length.png", 10, 5)

# Size-corrected body area indices
# Use trunk SCI to set the same individual-year order in both panels
individual_order <- whales %>%
  arrange(BAI_trunk_SCI) %>%
  pull(id)

bai_corrected_trunk <- whales %>%
  select(id, BAI_trunk_SCI) %>%
  pivot_longer(
    cols = c(BAI_trunk_SCI),
    names_to = "region",
    values_to = "SCI"
  ) %>%
  mutate(
    id = factor(id, levels = individual_order),
    region = factor(
      region,
      levels = c("BAI_trunk_SCI"),
      labels = c("Trunk (35–80% TL)")
    )
  )

p_bai_corrected <- ggplot(bai_corrected_trunk, aes(SCI, id)) +
  geom_vline(xintercept = 0, linetype = 2, colour = "grey50") +
  geom_point(size = 2.5) +
  facet_wrap(~ region) +
  labs(
    x = "Size-corrected BAI (% deviation from expected value)",
    y = NULL
  ) +
  theme_classic(base_size = 12)




p_bai_corrected
save_plot(p_bai_corrected, "BAI_size_corrected.png", 10, 8)

individual_order <- whales %>%
  arrange(BAI_head_SCI) %>%
  pull(id)


bai_corrected_head <- whales %>%
  select(id, BAI_head_SCI) %>%
  pivot_longer(
    cols = c(BAI_head_SCI),
    names_to = "region",
    values_to = "SCI"
  ) %>%
  mutate(
    id = factor(id, levels = individual_order),
    region = factor(
      region,
      levels = c("BAI_head_SCI"),
      labels = c("Head (5-35% TL)")
    )
  )

p_bai_corrected <- ggplot(bai_corrected_head, aes(SCI, id)) +
  geom_vline(xintercept = 0, linetype = 2, colour = "grey50") +
  geom_point(size = 2.5) +
  facet_wrap(~ region) +
  labs(
    x = "Size-corrected BAI (% deviation from expected value)",
    y = NULL
  ) +
  theme_classic(base_size = 12)

p_bai_corrected
# One row per comparison; exact = FALSE also handles tied ranks.
correlation_row <- function(x, y, label, method) {
  test <- cor.test(x, y, method = method, exact = FALSE)
  data.frame(comparison = label, method = method, n = length(x),
             coefficient = unname(test$estimate), p = test$p.value)
}
bai_comparisons <- bind_rows(
  #correlation_row(whales$BAI_trunk, whales$BAI_head, "BAI trunk vs head: raw", "spearman"),
  correlation_row(whales$BAI_trunk_SCI, whales$BAI_head_SCI,
                  "BAI trunk vs head: SCI", "spearman"),
  correlation_row(whales$BAI_trunk_SCI, whales$Volume_SCI,
                  "BAI trunk vs volume: SCI", "spearman"),
  correlation_row(whales$BAI_head_SCI, whales$Volume_SCI,
                  "BAI head vs volume: SCI", "spearman"))
save_table(bai_comparisons, "02_BAI_volume_correlations.csv")

# Comparison between size-corrected trunk and head BAI
# Select the case study and the minimum and maximum of each SCI
whales_bi <- whales
whales_bi$b_h_rank <- NA
order.BAI_head_SCI <- order(whales$BAI_head_SCI)
whales_bi <- whales_bi[order.BAI_head_SCI,]
whales_bi$b_h_rank <- rank(whales_bi$BAI_head_SCI)

whales_bi$b_t_rank <- NA
order.BAI_trunk_SCI <- order(whales$BAI_trunk_SCI)
whales_bi <- whales_bi[order.BAI_trunk_SCI,]
whales_bi$b_t_rank <- rank(whales_bi$BAI_trunk_SCI)

whales_bi$tot_index <- whales_bi$b_t_rank + whales_bi$b_h_rank 
labelled_whales <- whales_bi %>%
  filter(
    case_study |
      tot_index == min(tot_index) |
      tot_index == max(tot_index)
  )

# Comparison between size-corrected trunk and head BAI

p_bai_comparison <- ggplot(
  whales,
  aes(BAI_trunk_SCI, BAI_head_SCI)
) +
  geom_hline(
    yintercept = 0,
    linetype = 2,
    colour = "grey70"
  ) +
  geom_vline(
    xintercept = 0,
    linetype = 2,
    colour = "grey70"
  ) +
  geom_point(size = 2.5) +
  geom_point(
    data = labelled_whales,
    aes(colour = case_when(
      case_study ~ "Case study",
      tot_index == min(tot_index) ~ "Lowest rank sum",
      tot_index == max(tot_index) ~ "Highest rank sum"
    )),
    size = 3
  )  + geom_point(
    data = filter(whales, Subject == "Pm3_23"),
    colour = "#7B3294", size = 3
  ) +
  ggrepel::geom_text_repel(
    data = whales_bi,
    aes(
      label = ifelse(id %in% labelled_whales$id, id, ""),
      colour = case_when(
        case_study ~ "Case study",
        tot_index == min(tot_index) ~ "Lowest rank sum",
        tot_index == max(tot_index) ~ "Highest rank sum",
        TRUE ~ "Other individuals"
      )
    ),
    size = 3.5,
    box.padding = 0.6,
    point.padding = 0.4,
    min.segment.length = 0,
    max.overlaps = Inf,
    seed = 124
  ) +
  scale_colour_manual(
    values = c(
      "Case study" = "red3",
      "Lowest rank sum" = "steelblue4",
      "Highest rank sum" = "#7B3294",
      "Other individuals" = "black"
    ),
    guide = "none"
  ) +
  labs(
    x = "BAI trunk SCI (%)",
    y = "BAI head SCI (%)",
    subtitle = "Spearman’s ρ = 0.599; p = 0.0012; n = 26"
  ) +
  theme_classic(base_size = 12)

p_bai_comparison
save_plot(p_bai_comparison, "BAI_trunk_vs_head_SCI.png", 7, 6)

# The plot uses the same log-log regression as the test.
condition_long <- whales %>%
  select(id, TL, case_study, BAI_trunk, BAI_head, Volume) %>%
  pivot_longer(c(BAI_trunk, BAI_head, Volume), names_to = "metric", values_to = "value")
p_size <- ggplot(condition_long, aes(TL, value)) +
  geom_point(aes(colour = case_study), size = 2.5) +
  scale_x_log10() + scale_y_log10() + geom_smooth(method = "lm", formula = y ~ x) +
  facet_wrap(~ metric, scales = "free_y") +
  labs(x = "Total length (m, log scale)", y = "Index (log scale)", colour = "Pm3_26")
p_size
save_plot(p_size, "02_BAI_volume_vs_length.png", 11, 4)

bai_plot <- bind_rows(
  whales %>% transmute(id, metric = "BAI trunk", value = BAI_trunk,
                       lower = trunk_lower, upper = trunk_upper),
  whales %>% transmute(id, metric = "BAI head", value = BAI_head,
                       lower = head_lower, upper = head_upper))
p_bai <- ggplot(bai_plot, aes(value, id)) +
  geom_errorbar(aes(xmin = lower, xmax = upper), orientation = "y", width = .15) +
  geom_point() + facet_wrap(~ metric, scales = "free_x") +
  labs(x = "Posterior mean and 95% HPD", y = NULL)
p_bai
save_plot(p_bai, "02_BAI_estimates.png", 10, 8)


# Fit the volume–length relationship on the ln–ln scale
volume_model <- lm(log(Volume) ~ log(TL), data = whales)

# Display the volume–length model results
summary(volume_model)

# Extract the allometric exponent and its 95% confidence interval
b <- unname(coef(volume_model)[2])
b_interval <- confint(volume_model)[2, ]

volume_model_results <- data.frame(
  n = nobs(volume_model),
  a = unname(coef(volume_model)[1]),
  b = b,
  SE_b = summary(volume_model)$coefficients[2, 2],
  lower_b = b_interval[1],
  upper_b = b_interval[2],
  R2 = summary(volume_model)$r.squared,
  isometry_in_CI = b_interval[1] <= 3 & b_interval[2] >= 3
)

print(volume_model_results, row.names = FALSE)
save_table(volume_model_results, "Volume_length_model.csv")

# Compare volume-based SCI with the two BAI-based SCIs
volume_comparisons <- whales %>%
  select(id, case_study, Volume_SCI, BAI_trunk_SCI, BAI_head_SCI) %>%
  pivot_longer(
    cols = c(BAI_trunk_SCI, BAI_head_SCI),
    names_to = "region",
    values_to = "BAI_SCI"
  ) %>%
  mutate(
    region = factor(
      region,
      levels = c("BAI_trunk_SCI", "BAI_head_SCI"),
      labels = c(
        "Trunk BAI\nSpearman’s ρ = 0.956; p < 0.001",
        "Head BAI\nSpearman’s ρ = 0.790; p < 0.001"
      )
    )
  )

# Calculate ranks separately for each comparison
volume_comparisons <- volume_comparisons %>%
  group_by(region) %>%
  mutate(
    volume_rank = rank(Volume_SCI),
    BAI_rank = rank(BAI_SCI),
    total_rank = volume_rank + BAI_rank,
    point_group = case_when(
      case_study ~ "Case study",
      total_rank == min(total_rank) ~ "Lowest total rank",
      total_rank == max(total_rank) ~ "Highest total rank",
      TRUE ~ "Other individuals"
    ),
    plot_label = ifelse(point_group != "Other individuals", id, "")
  ) %>%
  ungroup()

volume_comparisons

# Compare volume-based SCI with the two BAI-based SCIs
p_volume_comparisons <- ggplot(
  volume_comparisons,
  aes(Volume_SCI, BAI_SCI)
) +
  geom_hline(
    yintercept = 0,
    linetype = 2,
    colour = "grey70"
  ) +
  geom_vline(
    xintercept = 0,
    linetype = 2,
    colour = "grey70"
  ) +
  geom_point(
    aes(colour = point_group),
    size = 2.5
  ) +
  ggrepel::geom_text_repel(
    aes(label = plot_label, colour = point_group),
    size = 3.5,
    box.padding = 0.6,
    point.padding = 0.4,
    min.segment.length = 0,
    max.overlaps = Inf,
    seed = 123
  ) +
  scale_colour_manual(
    values = c(
      "Case study" = "red3",
      "Lowest total rank" = "steelblue4",
      "Highest total rank" = "darkgreen",
      "Other individuals" = "black"
    ),
    guide = "none"
  ) +
  facet_wrap(~ region, nrow = 1) +
  labs(
    x = "Volume SCI (%)",
    y = "BAI SCI (%)"
  ) +
  theme_classic(base_size = 12)

p_volume_comparisons
save_plot(p_volume_comparisons, "Volume_vs_BAI_SCI.png", 11, 5)

# Predict volumes across the observed length range
volume_curves <- data.frame(
  TL = seq(min(whales$TL), max(whales$TL), length.out = 100)
)

volume_curves$study <- exp(
  predict(volume_model, newdata = volume_curves)
)

volume_curves$Glarou <- exp(
  -4.048 + 2.861 * log(volume_curves$TL)
)

# Compare individual volumes and the two fitted relationships
p_volume_length <- ggplot(whales, aes(TL, Volume)) +
  geom_errorbar(
    aes(ymin = volume_lower, ymax = volume_upper),
    width = 0.04,
    colour = "grey60"
  ) +
  geom_point(size = 2.5) +
  geom_point(
    data = whales %>% filter(case_study),
    colour = "red3",
    size = 3
  ) +
  geom_line(
    data = volume_curves,
    aes(y = study, colour = "This study"),
    linewidth = 1
  ) +
  geom_line(
    data = volume_curves,
    aes(y = Glarou, colour = "Glarou et al. (2023)"),
    linewidth = 1,
    linetype = 2
  ) +
  scale_colour_manual(
    values = c(
      "This study" = "steelblue4",
      "Glarou et al. (2023)" = "darkorange3"
    )
  ) +
  labs(
    x = "Total length (m)",
    y = "Body volume (m³)",
    colour = NULL
  ) +
  geom_point(
    data = filter(whales, Subject == "Pm3_23"),
    colour = "#7B3294", size = 3
  ) +
  theme_classic(base_size = 12) +
  theme(legend.position = "bottom")

p_volume_length
save_plot(p_volume_length, "Volume_vs_length_Glarou.png", 8, 6)

# 3. DESCRIPTIVE COMPARISON WITH THE GLAROU RELATIONSHIP
# Thesis: Body volume. Coefficients retained from the original code.
whales <- whales %>%
  mutate(volume_Glarou = exp(-4.048 + 2.861 * log(TL)),
         volume_difference_pct = 100 * (Volume / volume_Glarou - 1))
volume_comparison <- whales %>%
  summarise(n = n(), mean_volume = mean(Volume), min_volume = min(Volume),
            max_volume = max(Volume), mean_difference_pct = mean(volume_difference_pct),
            median_difference_pct = median(volume_difference_pct),
            r = cor(Volume, volume_Glarou))
save_table(volume_comparison, "03_volume_Glarou_summary.csv")


p_volume <- ggplot(whales, aes(volume_Glarou, Volume)) +
  geom_abline(intercept = 0, slope = 1, linetype = 2) +
  geom_point(aes(colour = case_study), size = 2.5) +
  labs(x = "Volume predicted by Glarou (m3)", y = "Estimated volume (m3)", colour = "Pm3_26")
p_volume
save_plot(p_volume, "03_volume_Glarou.png")

# 4. WIDTH VARIATION BETWEEN INDIVIDUAL-YEARS
# Thesis: Variation along body length between individuals.
# The updated thesis uses SD, not CV_between.
# Sections 5-85%, as in the previous C analysis.
# Select body widths from 5% to 85% of total length
variation_widths <- sprintf("W%02d_px", seq(5, 85, 5))

# Calculate relative widths for each frame
relative_widths <- frames %>%
  select(
    Subject, Timepoint, id, image_id, TL_px,
    all_of(variation_widths)
  ) %>%
  pivot_longer(
    cols = all_of(variation_widths),
    names_to = "measurement",
    values_to = "width"
  ) %>%
  mutate(
    section = as.numeric(gsub("W|_px", "", measurement)),
    ratio = width / TL_px
  )

# Calculate mean relative widths for each individual-year
individual_widths <- relative_widths %>%
  group_by(Subject, Timepoint, id, section) %>%
  summarise(
    n_frames = n(),
    mean_ratio = mean(ratio),
    .groups = "drop"
  )

# Calculate variation between individual-years at each section
width_variation <- individual_widths %>%
  group_by(section) %>%
  summarise(
    n = n(),
    SD_between = sd(mean_ratio),
    mean_ratio = mean(mean_ratio),
    .groups = "drop"
  )

# Display results for all sections
print(as.data.frame(width_variation), row.names = FALSE)

# Display the three sections with the highest SD
width_variation %>%
  arrange(desc(SD_between)) %>%
  slice_head(n = 3) %>%
  print()

# Display the section with the highest mean relative width
width_variation %>%
  filter(mean_ratio == max(mean_ratio)) %>%
  print()

# Save individual profiles and section summaries
save_table(individual_widths, "04_relative_widths_by_individual.csv")
save_table(width_variation, "04_width_variation.csv")

# Show variation across all body sections
print(as.data.frame(width_variation), row.names = FALSE)

# Section with the highest between-individual SD
cat("\nSection with the highest between-individual SD:\n")
width_variation %>%
  filter(SD_between == max(SD_between)) %>%
  print()

# Section with the highest mean relative width
cat("\nSection with the highest mean relative width:\n")
width_variation %>%
  filter(mean_ratio == max(mean_ratio)) %>%
  print()

# Rank individual-years within each body section
width_ranks <- individual_widths %>%
  group_by(section) %>%
  mutate(width_rank = rank(mean_ratio, ties.method = "min")) %>%
  ungroup()

# Compare the case study with the sample mean at each section
case_profile <- width_ranks %>%
  filter(Subject == CONTROL) %>%
  select(id, section, individual_ratio = mean_ratio, width_rank) %>%
  left_join(
    width_variation %>%
      select(section, sample_mean = mean_ratio),
    by = "section"
  ) %>%
  mutate(
    difference = individual_ratio - sample_mean,
    below_mean = individual_ratio < sample_mean
  ) %>%
  arrange(section)

cat("\nCase-study relative width profile:\n")
print(as.data.frame(case_profile), row.names = FALSE)

cat("\nSections where the case study has the lowest relative width:\n")
case_profile %>%
  filter(width_rank == 1) %>%
  print()

save_table(case_profile, "04_case_study_width_profile.csv")

p_variation <- ggplot(width_variation, aes(section, SD_between)) +
  geom_line() + geom_point() +
  labs(x = "Section (% TL)", y = "SD of individual-year mean W/TL")
save_plot(p_variation, "04_width_variation.png")

individual_widths2 <- individual_widths[individual_widths$section != "85",]
individual_widths2

p_profiles <- ggplot(individual_widths2, aes(section, mean_ratio, group = id)) +
  geom_line(colour = "grey70") +
  geom_line(data = filter(individual_widths2, Subject == CONTROL), colour = "firebrick", linewidth = 1) +
  geom_line(
    data = filter(individual_widths2, Subject == "Pm3_23"),
    colour = "#7B3294", linewidth = 1
  )+
  labs(x = "Section (% TL)", y = "Mean W/TL")
p_profiles
save_plot(p_profiles, "04_relative_width_profiles.png")


# Descriptive results for eye-to-eye width
EW_estimates <- measurements %>%
  filter(Measurement == "eyes")

EW_summary <- EW_estimates %>%
  summarise(
    n_individual_years = n(),
    n_individuals = n_distinct(Subject),
    mean_EW = mean(mean),
    SD_EW = sd(mean),
    min_EW = min(mean),
    max_EW = max(mean)
  )

# Individual-years with the lowest and highest posterior means
EW_extremes <- EW_estimates %>%
  filter(mean == min(mean) | mean == max(mean)) %>%
  select(Subject, Timepoint, mean, sd, lower, upper) %>%
  arrange(mean)

print(EW_summary)
print(EW_extremes)

save_table(EW_summary, "06_EW_descriptive_summary.csv")
save_table(EW_extremes, "06_EW_extremes.csv")


# 5. ISOMETRY OF EW AND THE OTHER WIDTHS
# Thesis: Isometry of eye-to-eye measurement.
# Comparison using the same units with EW: posterior means in METRES.
# smatr uses log10 with log = "xy". The test here is b = 1, not b = 0.
eye_whales <- whales %>% filter(!is.na(eyes))

sma_rows <- list()
for (measurement in c("eyes", sprintf("W%02d", seq(5, 80, 5)))) {
  sma_data <- data.frame(TL = eye_whales$TL, width = eye_whales[[measurement]])
  model <- smatr::sma(width ~ TL, data = sma_data, log = "xy", slope.test = 1)
  slope_test <- model$slopetest[[1]]
  sma_rows[[measurement]] <- data.frame(measurement = measurement, n = nrow(sma_data),
                                        slope = slope_test$b, lower = slope_test$ci[1], upper = slope_test$ci[2],
                                        R2 = model$r2[[1]], p_b1 = slope_test$p)
}
sma_results <- bind_rows(sma_rows)

save_table(sma_results, "05_SMA_eyes_and_widths.csv")
p_sma <- ggplot(sma_results, aes(slope, reorder(measurement, slope))) +
  geom_vline(xintercept = 1, linetype = 2) +
  geom_errorbar(aes(xmin = lower, xmax = upper), orientation = "y", width = .15) +
  geom_point() + labs(x = "SMA slope and 95% CI", y = NULL)
p_sma
save_plot(p_sma, "05_SMA_slopes.png", 8, 7)

measurement_order <- c("eyes", sprintf("W%02d", seq(5, 80, 5)))
measurement_labels <- c("EW", sprintf("W%02d", seq(5, 80, 5)))

# Stesso campione e stesse misure in metri usati nell'analisi SMA
sma_plot_data <- eye_whales %>%
  select(Subject, TL, all_of(measurement_order)) %>%
  pivot_longer(
    cols = all_of(measurement_order),
    names_to = "measurement",
    values_to = "width"
  ) %>%
  mutate(
    point_group = case_when(
      Subject == "Pm3_26" ~ "Pm3_26",
      Subject == "Pm3_23" ~ "Pm3_23",
      TRUE ~ "Other individuals"
    )
  )

# Intercette delle rette SMA sullo stesso campione
# La retta SMA passa per il centro dei dati log-trasformati
sma_plot_fits <- sma_plot_data %>%
  group_by(measurement) %>%
  summarise(
    mean_log_TL = mean(log10(TL)),
    mean_log_width = mean(log10(width)),
    .groups = "drop"
  ) %>%
  left_join(
    sma_results %>%
      select(measurement, slope, lower, upper, R2),
    by = "measurement"
  ) %>%
  mutate(
    intercept = mean_log_width - slope * mean_log_TL,
    intercept_isometry = mean_log_width - mean_log_TL,
    plot_label = sprintf(
      "b = %.3f [%.3f; %.3f]\nR² = %.3f",
      slope, lower, upper, R2
    )
  )

sma_plot_data$measurement <- factor(
  sma_plot_data$measurement,
  levels = measurement_order,
  labels = measurement_labels
)

sma_plot_fits$measurement <- factor(
  sma_plot_fits$measurement,
  levels = measurement_order,
  labels = measurement_labels
)

p_widths_TL_SMA <- ggplot(
  sma_plot_data,
  aes(log10(TL), log10(width))
) +
  # Riferimento isometrico: b = 1, passante per il centro dei dati
  geom_abline(
    data = sma_plot_fits,
    aes(intercept = intercept_isometry, slope = 1),
    colour = "grey60",
    linetype = 2
  ) +
  # Retta SMA già stimata
  geom_abline(
    data = sma_plot_fits,
    aes(intercept = intercept, slope = slope),
    colour = "steelblue4",
    linewidth = 0.8
  ) +
  geom_point(aes(colour = point_group), size = 2) +
  geom_text(
    data = sma_plot_fits,
    aes(x = -Inf, y = Inf, label = plot_label),
    inherit.aes = FALSE,
    hjust = -0.05,
    vjust = 1.15,
    size = 2.8
  ) +
  scale_colour_manual(
    values = c(
      "Other individuals" = "black",
      "Pm3_26" = "red3",
      "Pm3_23" = "#7B3294"
    )
  ) +
  scale_y_continuous(
    expand = expansion(mult = c(0.08, 0.25))
  ) +
  facet_wrap(~ measurement, ncol = 4, scales = "free_y") +
  labs(
    x = "log10(Total length [m])",
    y = "log10(Transverse measurement [m])",
    colour = "Individual",
    caption = paste0(
      "Blue: fitted SMA relationship. ",
      "Dashed grey: isometric reference (b = 1)."
    )
  ) +
  theme_classic(base_size = 12) +
  theme(
    legend.position = "bottom",
    strip.background = element_rect(
      colour = "black", fill = "white"
    )
  )

p_widths_TL_SMA

save_plot(
  p_widths_TL_SMA,
  "05_EW_and_widths_vs_TL_SMA.png",
  width = 12,
  height = 13
)

# Select individuals with an EW estimate
eye_whales <- whales %>%
  filter(!is.na(eyes))

# Extract the SMA slope and calculate the intercept
eye_sma <- sma_results %>%
  filter(measurement == "eyes")

b <- eye_sma$slope
a <- mean(log10(eye_whales$eyes)) -
  b * mean(log10(eye_whales$TL))

# Plot EW against total length
p_EW_length <- ggplot(
  eye_whales,
  aes(log10(TL), log10(eyes))
) +
  geom_point(size = 2.5) +
  geom_abline(
    intercept = a,
    slope = b,
    colour = "steelblue4",
    linewidth = 1
  ) +
  labs(
    x = "log10(Total length [m])",
    y = "log10(Eye-to-eye width [m])",
    subtitle = sprintf(
      "SMA: b = %.3f; R² = %.3f; n = %d",
      b, eye_sma$R2, nrow(eye_whales)
    )
  ) +
  theme_classic(base_size = 12)

p_EW_length
save_plot(p_EW_length, "EW_vs_length_SMA.png", 7, 6)

# Back-transform the SMA relationship to metres
eye_curve <- data.frame(
  TL = seq(min(eye_whales$TL), max(eye_whales$TL), length.out = 100)
)

eye_curve$EW <- 10^a * eye_curve$TL^b

# Plot EW against total length in metres
p_EW_length <- ggplot(eye_whales, aes(TL, eyes)) +
  geom_point(size = 2.5) +
  geom_line(
    data = eye_curve,
    aes(y = EW),
    colour = "steelblue4",
    linewidth = 1
  ) +
  labs(
    x = "Total length (m)",
    y = "Eye-to-eye width (m)",
    subtitle = sprintf(
      "SMA: b = %.3f; R² (log–log) = %.3f; n = %d",
      b, eye_sma$R2, nrow(eye_whales)
    )
  ) +
  theme_classic(base_size = 12)

p_EW_length
save_plot(p_EW_length, "EW_vs_length_SMA.png", 7, 6)


# Add posterior intervals for EW and total length
eye_plot_data <- eye_whales %>%
  left_join(
    EW_estimates %>%
      select(Subject, Timepoint, EW_lower = lower, EW_upper = upper),
    by = c("Subject", "Timepoint")
  ) %>%
  left_join(
    measurements %>%
      filter(Measurement == "TL_px") %>%
      select(Subject, Timepoint, TL_lower = lower, TL_upper = upper),
    by = c("Subject", "Timepoint")
  )

# Plot posterior means and 95% HPD intervals
p_EW_length <- ggplot(eye_plot_data, aes(TL, eyes)) +
  geom_errorbar(
    aes(ymin = EW_lower, ymax = EW_upper),
    width = 0.04,
    colour = "grey65"
  ) +
  geom_errorbar(
    aes(xmin = TL_lower, xmax = TL_upper),
    orientation = "y",
    width = 0.01,
    colour = "grey65"
  ) +
  geom_line(
    data = eye_curve,
    aes(y = EW),
    colour = "steelblue4",
    linewidth = 1
  ) +
  geom_point(size = 2.5) +
  labs(
    x = "Total length (m)",
    y = "Eye-to-eye width (m)",
    subtitle = sprintf(
      "SMA: b = %.3f; R² = %.3f; n = %d",
      b, eye_sma$R2, nrow(eye_plot_data)
    )
  ) +
  theme_classic(base_size = 12)

p_EW_length
save_plot(p_EW_length, "EW_vs_length_SMA.png", 7, 6)

# 6. FRAME-TO-FRAME STABILITY OF EW
# Thesis: Precision of the eye-to-eye measurement.
# EW is missing in some frames: this filter defines the common sample.
# Select frames in which EW was measured
eye_frames <- frames %>%
  filter(!is.na(eyes))

# Keep individual-year-flight groups with at least two frames
precision_frames <- eye_frames %>%
  group_by(Subject, Timepoint, flight) %>%
  filter(n() >= 2) %>%
  ungroup()

# Calculate CVs within each individual-year-flight group
precision <- precision_frames %>%
  select(
    Subject, Timepoint, flight, id, TL_px, eyes,
    all_of(sprintf("W%02d_px", seq(5, 80, 5)))
  ) %>%
  pivot_longer(
    cols = c(eyes, all_of(sprintf("W%02d_px", seq(5, 80, 5)))),
    names_to = "measurement",
    values_to = "width"
  ) %>%
  mutate(ratio = width / TL_px) %>%
  group_by(Subject, Timepoint, flight, id, measurement) %>%
  summarise(
    n_frames = n(),
    mean_ratio = mean(ratio),
    CV = 100 * sd(ratio) / mean(ratio),
    .groups = "drop"
  ) %>%
  group_by(Subject, Timepoint, flight) %>%
  mutate(rank_CV = rank(CV, ties.method = "min")) %>%
  ungroup()

# Give equal weight to each individual-year-flight group
precision_summary <- precision %>%
  group_by(measurement) %>%
  summarise(
    n_groups = n(),
    mean_CV = mean(CV),
    sd_CV = sd(CV),
    median_CV = median(CV),
    .groups = "drop"
  ) %>%
  mutate(rank_CV = rank(mean_CV, ties.method = "min")) %>%
  arrange(rank_CV)

print(precision_summary)
save_table(precision_summary, "06_CV_by_measurement.csv")

# Count groups in which EW has the lowest CV
eye_rank <- precision %>%
  filter(measurement == "eyes")

precision_count <- eye_rank %>%
  summarise(
    n_groups = n(),
    n_lowest_CV = sum(rank_CV == 1),
    pct_lowest_CV = 100 * mean(rank_CV == 1)
  )

print(precision_summary)
print(precision_count)

save_table(precision, "06_CV_by_individual_flight.csv")
save_table(precision_summary, "06_CV_by_measurement.csv")
save_table(precision_count, "06_EW_lowest_CV_count.csv")

# Plot mean CVs and individual-year-flight values
p_precision <- ggplot(precision, aes(measurement, CV)) +
  stat_summary(fun = mean, geom = "col", fill = "grey85") +
  geom_jitter(width = .1, height = 0, alpha = .5) +
  coord_flip() +
  labs(x = NULL, y = "Within-flight CV of width/TL (%)")

p_precision
save_plot(p_precision, "06_precision.png", 8, 7)


# 7. EI: SUM W/EW PER FRAME, THEN AVERAGE PER INDIVIDUAL-YEAR
# Thesis: The new EW exploratory index.
# Do not replace the mean of the ratios with the ratio of the means.
eye_frames$EI_trunk <- rowSums(eye_frames[, sprintf("W%02d_px", seq(35, 80, 5))]) / eye_frames$eyes
eye_frames$EI_head <- rowSums(eye_frames[, sprintf("W%02d_px", seq(5, 35, 5))]) / eye_frames$eyes
eye_indices <- eye_frames %>%
  select(Subject, Timepoint, image_id, EI_trunk, EI_head) %>%
  pivot_longer(c(EI_trunk, EI_head), names_to = "metric", values_to = "value") %>%
  group_by(Subject, Timepoint, metric) %>%
  summarise(n_frames = n(), mean = mean(value), SD = sd(value),
            SE = SD / sqrt(n_frames), CV = 100 * SD / mean, .groups = "drop")
# With only one frame, SD, SE and CV are NA: variability cannot be estimated.
save_table(eye_indices, "07_EI_by_individual.csv")
EI_wide <- eye_indices %>% select(Subject, Timepoint, metric, mean) %>%
  pivot_wider(names_from = metric, values_from = mean)
eye_whales <- eye_whales %>% left_join(EI_wide, by = c("Subject", "Timepoint"))

# Summarise individual-year EI values
EI_summary <- eye_indices %>%
  group_by(metric) %>%
  summarise(
    n_individual_years = n(),
    n_individuals = n_distinct(Subject),
    n_frames_total = sum(n_frames),
    mean_EI = mean(mean),
    SD_EI = sd(mean),
    min_EI = min(mean),
    max_EI = max(mean),
    .groups = "drop"
  )

# Summarise frame-to-frame variability where estimable
EI_CV_summary <- eye_indices %>%
  filter(n_frames >= 2) %>%
  group_by(metric) %>%
  summarise(
    n_individual_years = n(),
    mean_CV = mean(CV),
    .groups = "drop"
  )

# Identify the lowest and highest individual-year values
EI_extremes <- eye_indices %>%
  group_by(metric) %>%
  filter(mean == min(mean) | mean == max(mean)) %>%
  ungroup()

print(EI_summary)
print(EI_CV_summary)
print(EI_extremes)

save_table(EI_summary, "07_EI_summary.csv")
save_table(EI_CV_summary, "07_EI_CV_summary.csv")
save_table(EI_extremes, "07_EI_extremes.csv")

# 8. EI AND SIZE; COMPARISON BETWEEN INDICES
# Thesis: Dependence of EI on body size; EI comparison with established morphometric indices.
eye_size_rows <- list()
for (metric in c("EI_trunk", "EI_head")) {
  analysis <- size_analysis(eye_whales[[metric]], eye_whales$TL)
  eye_whales[[paste0(metric, "_SCI")]] <- analysis$SCI
  spearman <- cor.test(eye_whales[[metric]], eye_whales$TL, method = "spearman", exact = FALSE)
  eye_size_rows[[metric]] <- data.frame(metric = metric, analysis$table,
                                        rho_TL = unname(spearman$estimate), p_rho = spearman$p.value)
}
save_table(bind_rows(eye_size_rows), "08_EI_size_dependence.csv")

# Collect size-corrected EI values
EI_corrected <- eye_whales %>%
  select(Subject, Timepoint, id, case_study, EI_trunk_SCI, EI_head_SCI) %>%
  pivot_longer(
    cols = c(EI_trunk_SCI, EI_head_SCI),
    names_to = "metric",
    values_to = "SCI"
  )

# Summarise the range of each size-corrected index
EI_SCI_summary <- EI_corrected %>%
  group_by(metric) %>%
  summarise(
    n = n(),
    min_SCI = min(SCI),
    max_SCI = max(SCI),
    .groups = "drop"
  )

# Identify individual-years with the lowest and highest values
EI_SCI_extremes <- EI_corrected %>%
  group_by(metric) %>%
  filter(SCI == min(SCI) | SCI == max(SCI)) %>%
  ungroup()

print(EI_SCI_summary)
print(EI_SCI_extremes)

save_table(EI_corrected, "08_EI_size_corrected.csv")
save_table(EI_SCI_summary, "08_EI_SCI_summary.csv")
save_table(EI_SCI_extremes, "08_EI_SCI_extremes.csv")

# Order individual-years by trunk SCI
EI_order <- eye_whales %>%
  arrange(EI_trunk_SCI) %>%
  pull(id)

EI_corrected <- EI_corrected %>%
  mutate(
    id = factor(id, levels = EI_order),
    metric = factor(
      metric,
      levels = c("EI_trunk_SCI", "EI_head_SCI"),
      labels = c("Trunk (35–80% TL)", "Head (5–35% TL)")
    )
  )

# Plot size-corrected EIs
p_EI_corrected <- ggplot(EI_corrected, aes(SCI, id)) +
  geom_vline(xintercept = 0, linetype = 2, colour = "grey50") +
  geom_point(aes(colour = case_study), size = 2.5) +
  scale_colour_manual(
    values = c("FALSE" = "black", "TRUE" = "red3"),
    guide = "none"
  ) +
  facet_wrap(~ metric) +
  geom_point(
    data = filter(EI_corrected, Subject == "Pm3_23"),
    colour = "#7B3294", size = 2.5
  ) +
  labs(
    x = "Size-corrected EI (% deviation from expected value)",
    y = NULL
  ) +
  theme_classic(base_size = 12)

p_EI_corrected
save_plot(p_EI_corrected, "EI_size_corrected.png", 10, 7)

# BAI and volume corrections remain those from the full sample (section 2).
# Comparisons with EI use the individual-years with EW.
comparisons <- list()
i <- 0
for (eye_metric in c("EI_trunk", "EI_head")) {
  for (bai_metric in c("BAI_trunk", "BAI_head")) {
    for (method in c("pearson", "spearman")) {
      i <- i + 1
      comparisons[[i]] <- correlation_row(eye_whales[[eye_metric]], eye_whales[[bai_metric]],
                                          paste(eye_metric, "vs", bai_metric, "raw"), method)
    }
  }
}
corrected <- c("EI_trunk_SCI", "EI_head_SCI", "BAI_trunk_SCI", "BAI_head_SCI", "Volume_SCI")
pairs <- combn(corrected, 2)
for (j in seq_len(ncol(pairs))) {
  for (method in c("pearson", "spearman")) {
    i <- i + 1
    comparisons[[i]] <- correlation_row(eye_whales[[pairs[1, j]]], eye_whales[[pairs[2, j]]],
                                        paste(pairs[1, j], "vs", pairs[2, j]), method)
  }
}
save_table(bind_rows(comparisons), "08_EI_index_correlations.csv")

# 9. POSITION OF Pm3_26
# Thesis: Pm3_26 case study. Descriptive rank: 1 = lowest value.
# All five indices are compared using the same sample with EW.
rankings <- eye_whales %>%
  select(Subject, Timepoint, id, EI_trunk, EI_head, BAI_trunk, BAI_head, Volume,
         all_of(corrected)) %>%
  pivot_longer(-c(Subject, Timepoint, id), names_to = "metric", values_to = "value") %>%
  group_by(metric) %>% mutate(rank = rank(value), n = n()) %>% ungroup()
save_table(rankings, "09_all_rankings_eye_sample.csv")
save_table(filter(rankings, Subject == CONTROL), "09_Pm3_26_rankings.csv")
save_table(whales, "09_BAI_volume_all_individuals.csv")
save_table(eye_whales, "09_all_indices_eye_sample.csv")
p_ranks <- ggplot(rankings, aes(value, id, colour = Subject == CONTROL)) +
  geom_point() + facet_wrap(~ metric, scales = "free_x", ncol = 5) +
  labs(x = "Index value", y = NULL, colour = "Pm3_26")
p_ranks
save_plot(p_ranks, "09_condition_indices.png", 16, 10)

# Place the rank of each SCI in a separate column
mean_rankings <- rankings %>%
  filter(metric %in% c(
    "BAI_trunk_SCI",  "Volume_SCI","EI_trunk_SCI"
    #"EI_head_SCI", "BAI_head_SCI"
  )) %>%
  select(Subject, Timepoint, id, metric, rank) %>%
  pivot_wider(
    names_from = metric,
    values_from = rank,
    names_prefix = "rank_"
  ) %>%
  rowwise() %>%
  mutate(
    mean_rank = mean(c_across(starts_with("rank_"))),
    min_rank = min(c_across(starts_with("rank_"))),
    max_rank = max(c_across(starts_with("rank_")))
  ) %>%
  ungroup() %>%
  mutate(overall_rank = rank(mean_rank)) %>%
  arrange(overall_rank) %>%
  select(
    Subject, Timepoint, id, overall_rank, mean_rank,
    rank_BAI_trunk_SCI,rank_EI_trunk_SCI, rank_Volume_SCI,
    # rank_EI_head_SCI,rank_BAI_head_SCI
    min_rank, max_rank
  )

print(as.data.frame(mean_rankings), row.names = FALSE)
save_table(mean_rankings, "09_mean_rankings_only_trunk.csv")

# 10. SENSITIVITY TO THE SELECTED SECTIONS
# Thesis: Sensitivity analysis of the eye index.
# For 16 sections there are 2^16 - 1 = 65,535 non-empty combinations.
# The mean of the sum of W/EW equals the sum of the mean W/EW values
# when the same frames are used, as in this dataset.
sections <- seq(5, 80, 5)
eye_ratios <- eye_frames %>%
  select(Subject, Timepoint, eyes, all_of(sprintf("W%02d_px", sections))) %>%
  mutate(across(all_of(sprintf("W%02d_px", sections)), ~ .x / eyes)) %>%
  group_by(Subject, Timepoint) %>%
  summarise(n_frames = n(), across(all_of(sprintf("W%02d_px", sections)), mean), .groups = "drop") %>%
  left_join(whales %>% select(Subject, Timepoint, TL), by = c("Subject", "Timepoint"))

# combn() lists single sections first, then pairs, and so on.
combinations <- list()
for (n_sections in 1:16) {
  combinations <- c(combinations, combn(sections, n_sections, simplify = FALSE))
}
# Full analysis and repetition using the subset with at least 3 frames with EW.
samples <- list(all = eye_ratios, at_least_3_frames = filter(eye_ratios, n_frames >= 3))
for (sample_name in names(samples)) {
  data <- samples[[sample_name]]
  n_individuals <- nrow(data)
  raw_first <- rep(0, n_individuals)
  corrected_first <- rep(0, n_individuals)
  combination_rows <- vector("list", length(combinations))
  control_row <- which(data$Subject == CONTROL)
  cat("\nEI combinations:", sample_name, "-", n_individuals, "individual-years\n")
  for (j in seq_along(combinations)) {
    columns <- sprintf("W%02d_px", combinations[[j]])
    values <- rowSums(data[, columns, drop = FALSE])
    model <- lm(log(values) ~ log(TL), data = data)
    SCI <- 100 * (values / exp(fitted(model)) - 1)
    raw_rank <- rank(values, ties.method = "min")
    corrected_rank <- rank(SCI, ties.method = "min")
    raw_first <- raw_first + (raw_rank == 1)
    corrected_first <- corrected_first + (corrected_rank == 1)
    model_summary <- summary(model)
    combination_rows[[j]] <- data.frame(
      sections = paste(combinations[[j]], collapse = ";"),
      n_sections = length(columns), control_raw_rank = raw_rank[control_row],
      control_SCI_rank = corrected_rank[control_row],
      b = unname(coef(model)[2]), SE_b = model_summary$coefficients[2, 2],
      p_b0 = model_summary$coefficients[2, 4], R2 = model_summary$r.squared)
  }
  combination_results <- bind_rows(combination_rows)
  frequency <- data %>% select(Subject, Timepoint, n_frames) %>%
    mutate(n_combinations = length(combinations),
           n_first_raw = raw_first, pct_first_raw = 100 * raw_first / n_combinations,
           n_first_SCI = corrected_first, pct_first_SCI = 100 * corrected_first / n_combinations)
  save_table(frequency, paste0("10_first_rank_frequency_", sample_name, ".csv"))
  save_table(combination_results, paste0("10_all_combinations_", sample_name, ".csv"))
  save_table(filter(combination_results, control_raw_rank == 1 | control_SCI_rank == 1),
             paste0("10_combinations_Pm3_26_first_", sample_name, ".csv"))
}
# Combinations share the data: frequencies are descriptive, not p-values.
# Ties count as first place for each individual involved.
print(size_table)
print(precision_summary)
print(filter(rankings, Subject == CONTROL))
cat("\nTables and plots saved in:", ANALYSIS_DIR, "\n")

# ==============================================================================
# 11. MOST INFLUENTIAL BODY SECTIONS WHEN Pm3_26 RANKS FIRST
# ==============================================================================

for (sample_name in names(samples)) {
  file_path <- file.path(ANALYSIS_DIR, paste0("10_combinations_Pm3_26_first_", sample_name, ".csv"))
  if (!file.exists(file_path)) next
  
  comb_data <- read.csv(file_path)
  
  # Filter for the size-corrected metric (SCI)
  first_sci <- comb_data %>% filter(control_SCI_rank == 1)
  n_first <- nrow(first_sci)
  
  cat("\n======================================================================\n")
  cat("SAMPLE:", toupper(sample_name), "\n")
  cat(sprintf("Combinations with %s ranked first (SCI): %d of 65,535 (%.2f%%)\n", 
              CONTROL, n_first, 100 * n_first / 65535))
  cat("======================================================================\n")
  
  if (n_first == 0) {
    cat("No combinations found where", CONTROL, "ranks first.\n")
    next
  }
  
  # Calculate inclusion frequency for each section
  section_counts <- first_sci %>%
    select(sections) %>%
    tidyr::separate_rows(sections, sep = ";") %>%
    mutate(section = as.numeric(sections)) %>%
    count(section, name = "n_presence") %>%
    tidyr::complete(section = seq(5, 80, 5), fill = list(n_presence = 0)) %>%
    mutate(
      pct_inclusion = 100 * n_presence / n_first,
      section_label = sprintf("W%02d", section),
      region = ifelse(section <= 35, "Head (5–35% TL)", "Trunk (40–80% TL)"),
      # In the total combinatorial space, every section appears in exactly 50% of subsets
      effect = ifelse(pct_inclusion >= 50, "Favours rank 1 (>50%)", "Unfavourable (<50%)")
    ) %>%
    arrange(desc(pct_inclusion))
  
  # 1. Print sorted ranking to console
  cat("\nBody sections ranked by percentage of inclusion:\n")
  print(as.data.frame(section_counts %>% 
                        select(Section = section_label, 
                               TL_pct = section, 
                               Count = n_presence, 
                               `Inclusion %` = pct_inclusion, 
                               Region = region)), 
        row.names = FALSE)
  
  # Save summary table
  save_table(section_counts, paste0("11_section_frequency_Pm3_26_", sample_name, ".csv"))
  
  # 2. Lollipop plot ordered by importance
  p_sections <- ggplot(section_counts, aes(x = pct_inclusion, y = reorder(section_label, pct_inclusion))) +
    geom_vline(xintercept = 50, linetype = 2, colour = "grey50") +
    geom_segment(aes(x = 50, xend = pct_inclusion, 
                     y = reorder(section_label, pct_inclusion), 
                     yend = reorder(section_label, pct_inclusion)),
                 colour = "grey70", linewidth = 0.8) +
    geom_point(aes(colour = region, size = pct_inclusion)) +
    scale_colour_manual(values = c("Head (5–35% TL)" = "steelblue4", 
                                   "Trunk (40–80% TL)" = "darkorange3")) +
    scale_size_continuous(range = c(2.5, 5), guide = "none") +
    labs(
      title = sprintf("Influential body sections for %s ranking first", CONTROL),
      subtitle = sprintf("Sample: %s | Dashed line = Expected by chance (50%%)", 
                         sample_name),
      x = "% of rank 1 combinations including the section",
      y = "Body section",
      colour = "Body region"
    ) +
    theme_classic(base_size = 12) +
    theme(legend.position = "bottom")
  
  print(p_sections)
  save_plot(p_sections, paste0("11_influential_sections_", sample_name, ".png"), width = 8, height = 6)
}

# ==============================================================================
# 12. TOP 5 COMPOSITE INDICES: ALLOMETRY, INDIVIDUAL PLOTS & CORRELATIONS
# ==============================================================================

# Helper function to extract top 5 most frequent sections when CONTROL ranks 1st
get_top5_sections <- function(sample_name) {
  file_path <- file.path(ANALYSIS_DIR, paste0("10_combinations_Pm3_26_first_", sample_name, ".csv"))
  if (!file.exists(file_path)) stop(paste("File not found:", file_path))
  
  read.csv(file_path) %>%
    filter(control_SCI_rank == 1) %>%
    select(sections) %>%
    tidyr::separate_rows(sections, sep = ";") %>%
    mutate(section = as.numeric(sections)) %>%
    count(section, sort = TRUE) %>%
    slice_head(n = 5) %>%
    pull(section)
}

# 1. Identify the top 5 sections from each sensitivity analysis
top5_all     <- get_top5_sections("all")
top5_3frames <- get_top5_sections("at_least_3_frames")

cat("\n======================================================================\n")
cat("TOP 5 BODY SECTIONS SELECTED FOR COMPOSITE INDICES\n")
cat("Top 5 (All samples):        ", paste(sprintf("W%02d", top5_all), collapse = ", "), "\n")
cat("Top 5 (>= 3 frames sample): ", paste(sprintf("W%02d", top5_3frames), collapse = ", "), "\n")
cat("======================================================================\n")

# 2. Compute frame-level composite ratios and individual-year averages
cols_all     <- sprintf("W%02d_px", top5_all)
cols_3frames <- sprintf("W%02d_px", top5_3frames)

eye_frames$EI_top5_all <- rowSums(eye_frames[, cols_all, drop = FALSE]) / eye_frames$eyes
eye_frames$EI_top5_3f  <- rowSums(eye_frames[, cols_3frames, drop = FALSE]) / eye_frames$eyes

top5_individual <- eye_frames %>%
  select(Subject, Timepoint, id, EI_top5_all, EI_top5_3f) %>%
  pivot_longer(c(EI_top5_all, EI_top5_3f), names_to = "metric", values_to = "value") %>%
  group_by(Subject, Timepoint, id, metric) %>%
  summarise(mean_EI = mean(value), .groups = "drop") %>%
  pivot_wider(names_from = metric, values_from = mean_EI)

# Merge with whale metadata and established size-corrected metrics (SCI)
top5_whales <- whales %>%
  filter(!is.na(eyes)) %>%
  select(Subject, Timepoint, id, TL, case_study, 
         BAI_trunk_SCI, BAI_head_SCI, Volume_SCI) %>%
  left_join(top5_individual, by = c("Subject", "Timepoint", "id"))

# 3. TEST FOR ALLOMETRY (H0: b = 0) AND SIZE CORRECTION (SCI)
allometry_rows <- list()

for (metric in c("EI_top5_all", "EI_top5_3f")) {
  analysis <- size_analysis(top5_whales[[metric]], top5_whales$TL)
  top5_whales[[paste0(metric, "_SCI")]] <- analysis$SCI
  
  spearman_tl <- cor.test(top5_whales[[metric]], top5_whales$TL, method = "spearman", exact = FALSE)
  
  b_val   <- analysis$table$b
  lower_b <- analysis$table$lower_b
  upper_b <- analysis$table$upper_b
  p_val   <- analysis$table$p_b0
  
  allometry_rows[[metric]] <- data.frame(
    metric      = metric,
    n           = analysis$table$n,
    b           = b_val,
    SE_b        = analysis$table$SE_b,
    lower_b     = lower_b,
    upper_b     = upper_b,
    R2          = analysis$table$R2,
    p_b0        = p_val,
    rho_TL      = unname(spearman_tl$estimate),
    p_rho       = spearman_tl$p.value,
    zero_in_CI  = (lower_b <= 0 & upper_b >= 0),
    allometric  = ifelse(p_val < 0.05, "Yes (p < 0.05)", "No (p >= 0.05)")
  )
}

allometry_table <- bind_rows(allometry_rows)

cat("\n======================================================================\n")
cat("TEST OF SIZE-INDEPENDENCE (H0: b = 0) FOR TOP 5 COMPOSITE INDICES\n")
cat("======================================================================\n")
print(as.data.frame(allometry_table), row.names = FALSE)
save_table(allometry_table, "12_top5_allometry_test_b0.csv")

# 4. SPEARMAN CORRELATIONS: BETWEEN TOP 5 INDICES & VS BAI/VOLUME SCIs
corr_definitions <- list(
  list(x = "EI_top5_all_SCI", y = "EI_top5_3f_SCI",  label = "Top 5 All vs Top 5 (>= 3 frames)"),
  list(x = "EI_top5_all_SCI", y = "BAI_trunk_SCI",   label = "Top 5 All vs Trunk BAI"),
  list(x = "EI_top5_all_SCI", y = "BAI_head_SCI",    label = "Top 5 All vs Head BAI"),
  list(x = "EI_top5_all_SCI", y = "Volume_SCI",      label = "Top 5 All vs Volume"),
  list(x = "EI_top5_3f_SCI",  y = "BAI_trunk_SCI",   label = "Top 5 (>= 3 frames) vs Trunk BAI"),
  list(x = "EI_top5_3f_SCI",  y = "BAI_head_SCI",    label = "Top 5 (>= 3 frames) vs Head BAI"),
  list(x = "EI_top5_3f_SCI",  y = "Volume_SCI",      label = "Top 5 (>= 3 frames) vs Volume")
)

corr_rows <- list()
for (i in seq_along(corr_definitions)) {
  item <- corr_definitions[[i]]
  test <- cor.test(top5_whales[[item$x]], top5_whales[[item$y]], method = "spearman", exact = FALSE)
  corr_rows[[i]] <- data.frame(
    comparison = item$label,
    index_1    = item$x,
    index_2    = item$y,
    n          = nrow(top5_whales),
    rho        = unname(test$estimate),
    p_value    = test$p.value
  )
}

top5_correlations <- bind_rows(corr_rows)

cat("\n======================================================================\n")
cat("SPEARMAN CORRELATIONS: COMPOSITE INDICES VS ESTABLISHED SCIs\n")
cat("======================================================================\n")
print(as.data.frame(top5_correlations %>% select(comparison, n, rho, p_value)), row.names = FALSE)
save_table(top5_correlations, "12_top5_spearman_correlations.csv")

top5_scaling_data <- top5_whales %>%
  select(Subject, Timepoint, id, TL, EI_top5_all, EI_top5_3f) %>%
  pivot_longer(
    cols = c(EI_top5_all, EI_top5_3f),
    names_to = "metric",
    values_to = "value"
  ) %>%
  mutate(
    metric = factor(
      metric,
      levels = c("EI_top5_all", "EI_top5_3f"),
      labels = c(
        paste0(
          "Top 5 All Sections (",
          paste(sprintf("W%02d", top5_all), collapse = ", "),
          ")"
        ),
        paste0(
          "Top 5 >= 3 Frames (",
          paste(sprintf("W%02d", top5_3frames), collapse = ", "),
          ")"
        )
      )
    )
  )

p_top5_scaling <- ggplot(top5_scaling_data, aes(TL, value)) +
  geom_point(size = 2.5) +
  scale_x_log10(breaks = c(10, 11, 12, 13)) +
  scale_y_log10(breaks = scales::pretty_breaks(n = 4)) +
  geom_smooth(
    method = "lm",
    formula = y ~ x,
    se = TRUE,
    colour = "black",
    fill = "grey80"
  ) +
  geom_point(
    data = filter(top5_scaling_data, Subject == "Pm3_26"),
    colour = "red3",
    size = 2.5
  ) +
  geom_point(
    data = filter(top5_scaling_data, Subject == "Pm3_23"),
    colour = "#7B3294",
    size = 2.5
  ) +
  facet_wrap(~ metric, scales = "free_y", nrow = 1) +
  labs(
    title = "Allometric Scaling of Top 5 Composite Eye Indices",
    x = "Total length (m, log scale)",
    y = "Composite Eye Index (log scale)"
  ) +
  theme_classic(base_size = 12) +
  theme(
    panel.border = element_rect(colour = "black", fill = NA),
    strip.background = element_rect(colour = "black", fill = "white")
  )

p_top5_scaling

save_plot(
  p_top5_scaling,
  "12_top5_allometric_scaling.png",
  width = 12,
  height = 5
)

# 5. TWO SEPARATE ORDERED PLOTS FOR SIZE-CORRECTED CONDITION (SCI)

# Plot 1: Ordered by Top 5 All Sections
p_top5_all <- ggplot(
  top5_whales, 
  aes(x = EI_top5_all_SCI, y = reorder(id, EI_top5_all_SCI))
) +
  geom_vline(xintercept = 0, linetype = 2, colour = "grey50") +
  geom_point(aes(colour = case_study), size = 2.5) +
  scale_colour_manual(values = c("FALSE" = "black", "TRUE" = "red3"), guide = "none") +
  labs(
    title = "Top 5 Composite EI: All Samples",
    subtitle = sprintf("Sections: %s", 
                       paste(sprintf("W%02d", top5_all), collapse = ", ")),
    x = "Size-corrected EI (% deviation from expected value)",
    y = NULL
  )  +
  geom_point(
    data = filter(top5_whales, Subject == "Pm3_23"),
    colour = "#7B3294", size = 2.5
  )+
  theme_classic(base_size = 12)

print(p_top5_all)
save_plot(p_top5_all, "12_top5_all_SCI_ordered.png", width = 8, height = 7)

# Plot 2: Ordered by Top 5 (>= 3 Frames)
p_top5_3f <- ggplot(
  top5_whales, 
  aes(x = EI_top5_3f_SCI, y = reorder(id, EI_top5_3f_SCI))
) +
  geom_vline(xintercept = 0, linetype = 2, colour = "grey50") +
  geom_point(aes(colour = case_study), size = 2.5) +
  scale_colour_manual(values = c("FALSE" = "black", "TRUE" = "red3"), guide = "none") +
  labs(
    title = "Top 5 Composite EI: >= 3 Frames Sample",
    subtitle = sprintf("Sections: %s", 
                       paste(sprintf("W%02d", top5_3frames), collapse = ", ")),
    x = "Size-corrected EI (% deviation from expected value)",
    y = NULL
  ) +
  geom_point(
    data = filter(top5_whales, Subject == "Pm3_23"),
    colour = "#7B3294", size = 2.5
  )+
  theme_classic(base_size = 12)

print(p_top5_3f)
save_plot(p_top5_3f, "12_top5_3frames_SCI_ordered.png", width = 8, height = 7)

# Rank summary table export
top5_rankings <- top5_whales %>%
  mutate(
    rank_EI_top5_all_raw = rank(EI_top5_all, ties.method = "min"),
    rank_EI_top5_all_SCI = rank(EI_top5_all_SCI, ties.method = "min"),
    rank_EI_top5_3f_raw  = rank(EI_top5_3f, ties.method = "min"),
    rank_EI_top5_3f_SCI  = rank(EI_top5_3f_SCI, ties.method = "min")
  ) %>%
  arrange(rank_EI_top5_all_SCI)

save_table(top5_rankings, "12_top5_composite_rankings.csv")