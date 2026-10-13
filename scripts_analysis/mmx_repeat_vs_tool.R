library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)

theme_set(theme_classic(base_size = 12))

S1_PATH   <- ""
S2_PATH   <- ""
TOOL_PATH <- ""

OUT_DIR <- ""
if (!dir.exists(OUT_DIR)) dir.create(OUT_DIR, recursive = TRUE)

# These two functions avoid repeating the path every time a file is saved.
save_table <- function(table, name) {
  write_csv(table, file.path(OUT_DIR, name))
}
save_plot <- function(plot, name, width = 8, height = 5) {
  ggsave(file.path(OUT_DIR, name), plot, width = width, height = height, dpi = 300)
}

# 1. MATCHED MEASUREMENTS

s1 <- read_csv(S1_PATH, show_col_types = FALSE)
s2 <- read_csv(S2_PATH, show_col_types = FALSE)

tool <- read_csv(TOOL_PATH, show_col_types = FALSE) %>%
  mutate(frame_id = sub("\\.csv$", "", mmx_file))

# Keep total length and body widths; retain frame_id as the matching key.
s1_small <- s1 %>%
  select(frame_id, L_px, starts_with("L_w")) %>%
  rename(mmx1_L_px = L_px)

s2_small <- s2 %>%
  select(frame_id, L_px, starts_with("L_w")) %>%
  rename(mmx2_L_px = L_px)

# Give the width columns a common naming convention before joining.
for (p in seq(5, 95, 5)) {
  old <- paste0("L_w", p, "_0_px")
  
  names(s1_small)[names(s1_small) == old] <- paste0("mmx1_w", p, "_px")
  
  names(s2_small)[names(s2_small) == old] <- paste0("mmx2_w", p, "_px")
}

tool_small <- tool %>%
  select(frame_id, gruppo, validation_mmx_L_px = mmx_L_px, tool_L_px, starts_with("tool_w"))

dat <- s1_small %>%
  inner_join(s2_small, by = "frame_id") %>%
  inner_join(tool_small, by = "frame_id")

cat("\nMatched frames:", nrow(dat), "\n")

if (nrow(dat) != 48) {
  warning("Expected 48 matched frames, but found ", nrow(dat))
}

# The MMX reference in the validation table must match session 1.
mmx_check <- max(abs(dat$validation_mmx_L_px - dat$mmx1_L_px), na.rm = TRUE)

cat("Maximum difference between Tool-validation MMX and MMX session 1:", mmx_check, "px\n")

if (mmx_check > 1e-8) {
  warning("Validation MMX values do not exactly match MMX session 1.")
} else {
  cat("CHECK PASSED: validation MMX values match MMX session 1.\n")
}

# 2. AGREEMENT STATISTICS

agreement_stats <- function(reference, comparison, label) {
  ok <- is.finite(reference) & is.finite(comparison)
  x <- reference[ok]
  y <- comparison[ok]
  
  difference <- y - x
  
  n <- length(difference)
  bias <- mean(difference)
  sd_diff <- sd(difference)
  
  shapiro_result <- shapiro.test(difference)
  
  tibble(
    comparison = label, n = n, bias_px = bias, SD_difference_px = sd_diff,
    LoA_lower_px = bias - 1.96 * sd_diff, LoA_upper_px = bias + 1.96 * sd_diff,
    MAE_px = mean(abs(difference)), MAPE_percent =
      mean(abs(difference / x)) * 100,
    Shapiro_W = unname(shapiro_result$statistic),
    Shapiro_p = shapiro_result$p.value
  )
}

# 3. TOTAL-LENGTH AGREEMENT
length_summary <- bind_rows(
  agreement_stats(dat$mmx1_L_px, dat$mmx2_L_px, "MMX session 2 - MMX session 1"),
  agreement_stats(dat$mmx1_L_px, dat$tool_L_px, "Tool - MMX session 1"),
  agreement_stats(dat$mmx2_L_px, dat$tool_L_px, "Tool - MMX session 2")
)

cat("\n=== TOTAL-LENGTH AGREEMENT ===\n")
print(length_summary, n = Inf)

save_table(length_summary, "length_agreement_summary.csv")

# Compute absolute percentage errors for each frame using MMX session 1.
length_errors <- dat %>%
  transmute(
    frame_id, gruppo, mmx_repeat_abs_pct =
      100 * abs(mmx2_L_px - mmx1_L_px) / mmx1_L_px,
    tool_vs_mmx1_abs_pct = 100 * abs(tool_L_px - mmx1_L_px) / mmx1_L_px
  )

save_table(length_errors, "length_frame_level_errors.csv")

# Average the three frames within each of the 16 groups/videos.
# The group, rather than the individual frame, is the unit of inference.
length_group <- length_errors %>%
  group_by(gruppo) %>%
  summarise(
    mmx_repeat_abs_pct = mean(mmx_repeat_abs_pct, na.rm = TRUE),
    tool_vs_mmx1_abs_pct = mean(tool_vs_mmx1_abs_pct, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(paired_difference = tool_vs_mmx1_abs_pct - mmx_repeat_abs_pct)

cat("\n=== GROUP-LEVEL LENGTH ERRORS ===\n")
print(length_group, n = Inf)

save_table(length_group, "length_group_level_errors.csv")

# Check normality of the paired GROUP differences before the paired t-test.
length_shapiro <- shapiro.test(length_group$paired_difference)

cat("\n=== SHAPIRO-WILK: LENGTH PAIRED DIFFERENCES ===\n")
print(length_shapiro)

length_t <- t.test(
  length_group$tool_vs_mmx1_abs_pct, length_group$mmx_repeat_abs_pct, paired = TRUE
)

length_test_results <- tibble(
  analysis = c(
    "Shapiro-Wilk test on paired group-level differences",
    "Paired t-test: Tool-MMX1 error vs MMX repeatability error"
  ), n_groups = nrow(length_group),
  statistic = c(unname(length_shapiro$statistic), unname(length_t$statistic)),
  df = c(NA_real_, unname(length_t$parameter)),
  p_value = c(length_shapiro$p.value, length_t$p.value),
  mean_paired_difference_percentage_points = c(NA_real_, mean(length_group$paired_difference))
)

cat("\n=== LENGTH INFERENTIAL TESTS ===\n")
print(length_test_results)

save_table(length_test_results, "length_inferential_tests.csv")

# 4. TOTAL-LENGTH PLOTS
length_ba <- bind_rows(
  dat %>%
    transmute(
      frame_id, comparison = "MMX2 - MMX1", average_px =
        (mmx1_L_px + mmx2_L_px) / 2,
      difference_px = mmx2_L_px - mmx1_L_px
    ),
  dat %>%
    transmute(
      frame_id, comparison = "Tool - MMX1", average_px =
        (mmx1_L_px + tool_L_px) / 2,
      difference_px = tool_L_px - mmx1_L_px
    ),
  dat %>%
    transmute(
      frame_id, comparison = "Tool - MMX2", average_px =
        (mmx2_L_px + tool_L_px) / 2,
      difference_px = tool_L_px - mmx2_L_px
    )
)

ba_lines <- length_ba %>%
  group_by(comparison) %>%
  summarise(
    bias = mean(difference_px),
    lower = bias - 1.96 * sd(difference_px),
    upper = bias + 1.96 * sd(difference_px),
    .groups = "drop"
  )

p_ba <- ggplot(length_ba, aes(x = average_px, y = difference_px)) +
  geom_point(size = 2.1, alpha = 0.75) +
  geom_hline(data = ba_lines, aes(yintercept = bias), linewidth = 0.7) +
  geom_hline(data = ba_lines, aes(yintercept = lower), linetype = 2) +
  geom_hline(data = ba_lines, aes(yintercept = upper), linetype = 2) +
  facet_wrap(~ comparison, scales = "free_y") +
  labs(
    x = "Mean of paired measurements (px)", y = "Difference between paired measurements (px)",
    title = "Bland-Altman comparison of total-length measurements"
  ) +
  theme_classic(base_size = 12)

save_plot(p_ba, "length_bland_altman.png", 12, 4)

p_ba

# Each line connects the two mean absolute errors for one group/video.
length_group_plot <- length_group %>%
  select(
    gruppo, `MMX repeatability` = mmx_repeat_abs_pct, `Tool vs MMX1` = tool_vs_mmx1_abs_pct
  ) %>%
  pivot_longer(cols = -gruppo, names_to = "comparison", values_to = "absolute_error_pct")

p_length_error <- ggplot(
  length_group_plot, aes(x = comparison, y = absolute_error_pct, group = gruppo)
) +
  geom_line(alpha = 0.35) +
  geom_point(size = 2.4, alpha = 0.8) +
  labs(
    x = NULL, y = "Mean absolute percentage error per group (%)",
    title = "Total-length disagreement relative to operator repeatability"
  ) +
  theme_classic(base_size = 12)

save_plot(p_length_error, "length_absolute_error_comparison.png", 6.5, 5)

# 5. BODY-WIDTH AGREEMENT

positions <- seq(5, 95, 5)

width_long <- bind_rows(
  lapply(
    positions, function(p) {
      tibble(
        frame_id = dat$frame_id, gruppo = dat$gruppo, position = p, mmx1 =
          dat[[paste0("mmx1_w", p, "_px")]],
        mmx2 = dat[[paste0("mmx2_w", p, "_px")]],
        tool = dat[[paste0("tool_w", p, "_px")]]
      )
    }
  )
) %>%
  mutate(
    mmx_repeat_abs_pct = 100 * abs(mmx2 - mmx1) / mmx1,
    tool_vs_mmx1_abs_pct = 100 * abs(tool - mmx1) / mmx1
  )

save_table(width_long, "width_frame_level_errors_long.csv")

# Summarise absolute percentage errors separately at each body position.
width_position_summary <- width_long %>%
  group_by(position) %>%
  summarise(
    n = sum(is.finite(mmx_repeat_abs_pct) & is.finite(tool_vs_mmx1_abs_pct)),
    MMX_repeatability_MAPE = mean(mmx_repeat_abs_pct, na.rm = TRUE),
    Tool_vs_MMX1_MAPE = mean(tool_vs_mmx1_abs_pct, na.rm = TRUE),
    .groups = "drop"
  )

cat("\n=== WIDTH MAPE BY BODY POSITION ===\n")
print(width_position_summary, n = Inf)

save_table(width_position_summary, "width_MAPE_by_position.csv")

# Average the 19 width errors within each frame before averaging frames.
width_frame <- width_long %>%
  group_by(frame_id, gruppo) %>%
  summarise(
    mmx_repeat_width_MAPE = mean(mmx_repeat_abs_pct, na.rm = TRUE),
    tool_vs_mmx1_width_MAPE = mean(tool_vs_mmx1_abs_pct, na.rm = TRUE),
    .groups = "drop"
  )

save_table(width_frame, "width_frame_mean_errors.csv")

# Average frames within groups/videos to obtain paired independent units.
width_group <- width_frame %>%
  group_by(gruppo) %>%
  summarise(
    mmx_repeat_width_MAPE = mean(mmx_repeat_width_MAPE, na.rm = TRUE),
    tool_vs_mmx1_width_MAPE = mean(tool_vs_mmx1_width_MAPE, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(paired_difference = tool_vs_mmx1_width_MAPE - mmx_repeat_width_MAPE)

save_table(width_group, "width_group_level_errors.csv")

# Normality is assessed on the paired group-level differences.
width_shapiro <- shapiro.test(width_group$paired_difference)

cat("\n=== SHAPIRO-WILK: WIDTH PAIRED DIFFERENCES ===\n")
print(width_shapiro)

width_t <- t.test(
  width_group$tool_vs_mmx1_width_MAPE, width_group$mmx_repeat_width_MAPE, paired = TRUE
)

width_test_results <- tibble(
  analysis = c(
    "Shapiro-Wilk test on paired group-level differences",
    "Paired t-test: Tool-MMX1 width error vs MMX repeatability error"
  ), n_groups = nrow(width_group),
  statistic = c(unname(width_shapiro$statistic), unname(width_t$statistic)),
  df = c(NA_real_, unname(width_t$parameter)),
  p_value = c(width_shapiro$p.value, width_t$p.value),
  mean_paired_difference_percentage_points = c(NA_real_, mean(width_group$paired_difference))
)

cat("\n=== WIDTH INFERENTIAL TESTS ===\n")
print(width_test_results)

save_table(width_test_results, "width_inferential_tests.csv")

# Overall width MAPE is the mean of the frame-level mean errors.
width_overall <- tibble(
  comparison = c("MMX session 2 - MMX session 1", "Tool - MMX session 1"), MAPE_percent = c(
    mean(width_frame$mmx_repeat_width_MAPE), mean(width_frame$tool_vs_mmx1_width_MAPE)
  )
)

cat("\n=== OVERALL WIDTH MAPE ===\n")
print(width_overall)

save_table(width_overall, "width_overall_MAPE.csv")

# Compare repeatability and tool disagreement along the body.
width_plot <- width_position_summary %>%
  pivot_longer(
    cols = c(MMX_repeatability_MAPE, Tool_vs_MMX1_MAPE), names_to = "comparison",
    values_to = "MAPE"
  ) %>%
  mutate(
    comparison = recode(
      comparison, "MMX_repeatability_MAPE" =
        "MMX repeatability",
      "Tool_vs_MMX1_MAPE" = "Tool vs MMX1"
    )
  )

p_width <- ggplot(
  width_plot,
  aes(x = position, y = MAPE, group = comparison, linetype = comparison, shape = comparison)
) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 2.3) +
  scale_x_continuous(breaks = positions) +
  labs(
    x = "Position along body length (%)", y = "Mean absolute percentage error (%)",
    linetype = NULL, shape = NULL,
    title = "Body-width measurement error along the longitudinal axis"
  ) +
  theme_classic(base_size = 12)

save_plot(p_width, "width_MAPE_profile.png", 8, 5)

# 6. FINAL SUMMARY
final_summary <- tibble(
  metric = c(
    "Length MAPE: MMX1 vs MMX2", "Length MAPE: Tool vs MMX1",
    "Length Shapiro p: paired group-level error differences", "Length paired t-test p",
    "Width MAPE: MMX1 vs MMX2", "Width MAPE: Tool vs MMX1",
    "Width Shapiro p: paired group-level error differences", "Width paired t-test p"
  ), value = c(
    length_summary$MAPE_percent[
      length_summary$comparison == "MMX session 2 - MMX session 1"
    ], length_summary$MAPE_percent[
      length_summary$comparison == "Tool - MMX session 1"
    ], length_shapiro$p.value, length_t$p.value, width_overall$MAPE_percent[
      width_overall$comparison == "MMX session 2 - MMX session 1"
    ], width_overall$MAPE_percent[
      width_overall$comparison == "Tool - MMX session 1"
    ], width_shapiro$p.value, width_t$p.value
  )
)

cat("\n=== FINAL SUMMARY ===\n")
print(final_summary)

save_table(final_summary, "final_summary.csv")

cat("\nAnalysis completed. Results saved in:", OUT_DIR, "\n")
