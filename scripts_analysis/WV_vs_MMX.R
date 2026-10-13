library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)

theme_set(theme_classic(base_size = 12))

CSV_PATH <- ""
OUT_DIR  <- ""

if (!dir.exists(OUT_DIR)) dir.create(OUT_DIR, recursive = TRUE)

# These two functions avoid repeating the path every time a file is saved.
save_table <- function(table, name) {
  write_csv(table, file.path(OUT_DIR, name))
}
save_plot <- function(plot, name, width = 8, height = 5) {
  ggsave(file.path(OUT_DIR, name), plot, width = width, height = height, dpi = 300)
}

# 1. MEASUREMENTS AND AUTOMATIC PROPOSALS

dat <- read_csv(CSV_PATH, show_col_types = FALSE) %>%
  mutate(
    row_id = row_number(),
    # tool_ancorata indicates whether an automatic proposal was shown.
    # Check its workflow labels before interpreting the proposal comparison.
    proposal = ifelse(tool_ancorata, "Proposal shown", "No proposal"),
    # Relative errors use the MMX measurement as their denominator.
    length_diff_px = tool_L_px - mmx_L_px, length_abs_diff_px = abs(length_diff_px),
    length_rel_diff_pct = 100 * length_diff_px / mmx_L_px,
    length_abs_rel_diff_pct = abs(length_rel_diff_pct),
    # Pairwise CV allows comparison with Bierlich et al. (2024).
    paired_CV_pct = 100 *
      (abs(tool_L_px - mmx_L_px) / sqrt(2)) /
      ((tool_L_px + mmx_L_px) / 2)
  )

# Cross-check the proposal flag against the availability of an automatic value.
cat("\nProposal indicator:\n")
print(table(dat$proposal, useNA = "ifany"))

cat("\nCross-check: proposal vs presence of tool_auto_px\n")
print(table(proposal = dat$proposal, auto_value_present = !is.na(dat$tool_auto_px)))

# 2. AGREEMENT AND CLUSTERED REGRESSION
ccc <- function(reference, test) {
  ok <- is.finite(reference) & is.finite(test)
  x <- reference[ok]
  y <- test[ok]
  
  vx <- var(x)
  vy <- var(y)
  cv <- cov(x, y)
  
  2 * cv / (vx + vy + (mean(x) - mean(y))^2)
}

# Summarise agreement in pixels, relative error and diagnostic tests.
agreement_stats <- function(reference, test) {
  ok <- is.finite(reference) & is.finite(test)
  x <- reference[ok]
  y <- test[ok]
  
  difference <- y - x
  average <- (y + x) / 2
  
  n <- length(difference)
  bias <- mean(difference)
  sd_diff <- sd(difference)
  
  loa_lower <- bias - 1.96 * sd_diff
  loa_upper <- bias + 1.96 * sd_diff
  
  mae <- mean(abs(difference))
  mape <- mean(abs(difference / x)) * 100
  
  se_bias <- sd_diff / sqrt(n)
  tcrit <- qt(0.975, df = n - 1)
  
  proportional_model <- lm(difference ~ average)
  proportional_p <- coef(summary(proportional_model))["average", "Pr(>|t|)"]
  
  tibble(
    n = n, bias = bias, bias_CI_lower = bias - tcrit * se_bias,
    bias_CI_upper = bias + tcrit * se_bias, SD_difference = sd_diff, LoA_lower = loa_lower,
    LoA_upper = loa_upper, MAE = mae, MAPE_percent = mape, CCC = ccc(x, y),
    Shapiro_p = shapiro.test(difference)$p.value, proportional_bias_p = proportional_p
  )
}

# Cluster-robust standard errors account approximately for frames from
# the same group/video. Tests use degrees of freedom = number of clusters - 1.
cluster_robust_lm <- function(formula, data, cluster) {
  fit <- lm(formula, data = data)
  
  X <- model.matrix(fit)
  u <- residuals(fit)
  cl <- as.factor(data[[cluster]])
  
  # Align the cluster labels with the rows retained by lm.
  keep <- as.numeric(names(u))
  if (length(keep) != nrow(data)) {
    X <- X
    cl <- cl[keep]
  }
  
  N <- nrow(X)
  K <- ncol(X)
  G <- nlevels(cl)
  
  bread <- solve(crossprod(X))
  
  meat <- matrix(0, nrow = K, ncol = K)
  
  for (g in levels(cl)) {
    idx <- which(cl == g)
    Xg <- X[idx, , drop = FALSE]
    ug <- u[idx]
    
    score_g <- crossprod(Xg, ug)
    meat <- meat + score_g %*% t(score_g)
  }
  
  correction <- (G / (G - 1)) * ((N - 1) / (N - K))
  V <- correction * bread %*% meat %*% bread
  
  se <- sqrt(diag(V))
  beta <- coef(fit)
  tval <- beta / se
  pval <- 2 * pt(abs(tval), df = G - 1, lower.tail = FALSE)
  
  ci_low <- beta - qt(0.975, df = G - 1) * se
  ci_high <- beta + qt(0.975, df = G - 1) * se
  
  tibble(
    term = names(beta), estimate = beta, robust_SE = se, t = tval, df = G - 1, p = pval,
    CI_lower = ci_low, CI_upper = ci_high
  )
}

# 3. TOTAL-LENGTH AGREEMENT
length_stats <- agreement_stats(dat$mmx_L_px, dat$tool_L_px)

cat("\n=== FINAL TOTAL-LENGTH AGREEMENT ===\n")
print(length_stats)
save_table(length_stats, "length_agreement_summary.csv")

# Summarise the paired CV for comparison with Bierlich et al. (2024).
cv_summary <- dat %>%
  summarise(
    n = n(), mean_CV_pct = mean(paired_CV_pct, na.rm = TRUE),
    median_CV_pct = median(paired_CV_pct, na.rm = TRUE),
    SD_CV_pct = sd(paired_CV_pct, na.rm = TRUE), max_CV_pct = max(paired_CV_pct, na.rm = TRUE)
  )

cat("\n=== PAIRED CV ===\n")
print(cv_summary)
save_table(cv_summary, "length_pairwise_CV.csv")

# The identity line represents equal measurements from the two methods.
p_agreement <- ggplot(dat, aes(x = mmx_L_px, y = tool_L_px)) +
  geom_point(size = 2.5, alpha = 0.8) +
  geom_abline(intercept = 0, slope = 1, linetype = 2) +
  coord_equal() +
  labs(
    x = "Morphometrix total length (px)", y = "Custom tool total length (px)",
    title = "Agreement in total-length measurements"
  ) +
  theme_classic(base_size = 13)

save_plot(p_agreement, "length_agreement.png", 6, 6)

# Bland-Altman differences are tool minus MMX; limits are bias +/- 1.96 SD.
bias <- mean(dat$length_diff_px, na.rm = TRUE)
sd_diff <- sd(dat$length_diff_px, na.rm = TRUE)
loa_low <- bias - 1.96 * sd_diff
loa_high <- bias + 1.96 * sd_diff

p_ba <- ggplot(dat, aes(x = (mmx_L_px + tool_L_px) / 2, y = length_diff_px)) +
  geom_point(size = 2.5, alpha = 0.8) +
  geom_hline(yintercept = bias, linewidth = 0.8) +
  geom_hline(yintercept = loa_low, linetype = 2) +
  geom_hline(yintercept = loa_high, linetype = 2) +
  labs(
    x = "Mean of custom tool and Morphometrix length (px)",
    y = "Custom tool - Morphometrix (px)", title = "Bland-Altman plot: total length",
    subtitle = paste0(
      "Bias = ", round(bias, 2), " px; 95% LoA = ", round(loa_low, 2), " to ",
      round(loa_high, 2), " px"
    )
  ) +
  theme_classic(base_size = 13)

save_plot(p_ba, "length_bland_altman.png", 7, 5)

# 4. BODY-WIDTH AGREEMENT

mmx_widths <- dat %>%
  select(row_id, starts_with("mmx_w")) %>%
  pivot_longer(
    cols = starts_with("mmx_w"), names_to = "position", names_pattern = "mmx_w([0-9]+)_px",
    values_to = "mmx_width_px"
  )

tool_widths <- dat %>%
  select(row_id, starts_with("tool_w")) %>%
  pivot_longer(
    cols = starts_with("tool_w"), names_to = "position", names_pattern = "tool_w([0-9]+)_px",
    values_to = "tool_width_px"
  )

width_long <- left_join(mmx_widths, tool_widths, by = c("row_id", "position")) %>%
  mutate(
    position = as.numeric(position), difference_px = tool_width_px - mmx_width_px,
    relative_difference_pct = 100 * difference_px / mmx_width_px,
    absolute_relative_difference_pct = abs(relative_difference_pct)
  )

# Position-specific intervals describe the mean relative bias.
width_summary <- width_long %>%
  group_by(position) %>%
  summarise(
    n = sum(is.finite(relative_difference_pct)), mean_relative_bias_pct =
      mean(relative_difference_pct, na.rm = TRUE),
    MAPE_pct = mean(absolute_relative_difference_pct, na.rm = TRUE),
    SD_relative_pct = sd(relative_difference_pct, na.rm = TRUE),
    SE_relative_pct = SD_relative_pct / sqrt(n),
    CI_lower = mean_relative_bias_pct - 1.96 * SE_relative_pct,
    CI_upper = mean_relative_bias_pct + 1.96 * SE_relative_pct,
    CCC = ccc(mmx_width_px, tool_width_px), .groups = "drop"
  )

overall_width_summary <- width_long %>%
  summarise(
    n_measurements = sum(is.finite(relative_difference_pct)), mean_relative_bias_pct =
      mean(relative_difference_pct, na.rm = TRUE),
    MAPE_pct = mean(absolute_relative_difference_pct, na.rm = TRUE),
    median_absolute_percentage_error = median(absolute_relative_difference_pct, na.rm = TRUE)
  )

cat("\n=== WIDTH AGREEMENT BY BODY POSITION ===\n")
print(width_summary, n = Inf)

cat("\n=== OVERALL WIDTH AGREEMENT ===\n")
print(overall_width_summary)

save_table(width_summary, "width_agreement_by_position.csv")

save_table(overall_width_summary, "width_agreement_overall.csv")

p_width <- ggplot(width_summary, aes(x = position, y = mean_relative_bias_pct)) +
  geom_ribbon(aes(ymin = CI_lower, ymax = CI_upper), alpha = 0.2) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 2.5) +
  geom_hline(yintercept = 0, linetype = 2) +
  scale_x_continuous(breaks = seq(5, 95, 5)) +
  labs(
    x = "Position along body length (%)",
    y = "Mean relative difference: Tool - Morphometrix (%)",
    title = "Agreement of body-width measurements"
  ) +
  theme_classic(base_size = 13)

save_plot(p_width, "width_profile_difference.png", 8, 5)

# 5. EXPLORATORY ANCHORING ANALYSIS

anchoring_summary <- dat %>%
  group_by(proposal) %>%
  summarise(
    n = n(), mean_signed_error_px =
      mean(length_diff_px, na.rm = TRUE),
    SD_signed_error_px = sd(length_diff_px, na.rm = TRUE),
    mean_relative_error_pct = mean(length_rel_diff_pct, na.rm = TRUE),
    SD_relative_error_pct = sd(length_rel_diff_pct, na.rm = TRUE),
    mean_absolute_error_px = mean(length_abs_diff_px, na.rm = TRUE),
    mean_absolute_relative_error_pct = mean(length_abs_rel_diff_pct, na.rm = TRUE),
    median_absolute_relative_error_pct = median(length_abs_rel_diff_pct, na.rm = TRUE),
    .groups = "drop"
  )

cat("\n=== ANCHORING DESCRIPTIVE SUMMARY ===\n")
print(anchoring_summary)

save_table(anchoring_summary, "anchoring_summary.csv")

# Unclustered Welch tests are secondary checks.
welch_signed <- t.test(length_rel_diff_pct ~ proposal, data = dat)

welch_absolute <- t.test(length_abs_rel_diff_pct ~ proposal, data = dat)

cat("\n=== SIMPLE WELCH TESTS ===\n")
print(welch_signed)
print(welch_absolute)

# The preferred comparison uses cluster-robust regression by group/video.
# The reference level is "No proposal".
dat$proposal <- factor(dat$proposal, levels = c("No proposal", "Proposal shown"))

anchor_signed_cluster <- cluster_robust_lm(
  length_rel_diff_pct ~ proposal, data = dat, cluster = "gruppo"
)

anchor_absolute_cluster <- cluster_robust_lm(
  length_abs_rel_diff_pct ~ proposal, data = dat, cluster = "gruppo"
)

cat("\n=== CLUSTER-ROBUST MODEL: SIGNED RELATIVE ERROR ===\n")
print(anchor_signed_cluster)

cat("\n=== CLUSTER-ROBUST MODEL: ABSOLUTE RELATIVE ERROR ===\n")
print(anchor_absolute_cluster)

save_table(anchor_signed_cluster, "anchoring_signed_error_cluster_model.csv")

save_table(anchor_absolute_cluster, "anchoring_absolute_error_cluster_model.csv")

# Show signed and absolute relative errors by proposal availability.
p_anchor_signed <- ggplot(dat, aes(x = proposal, y = length_rel_diff_pct)) +
  geom_boxplot(width = 0.45, outlier.shape = NA) +
  geom_jitter(width = 0.08, height = 0, alpha = 0.7, size = 2) +
  geom_hline(yintercept = 0, linetype = 2) +
  labs(
    x = NULL, y = "Final Tool - Morphometrix (%)",
    title = "Final measurement error with and without an automatic proposal"
  ) +
  theme_classic(base_size = 13)

save_plot(p_anchor_signed, "anchoring_signed_error.png", 6.5, 5)

p_anchor_abs <- ggplot(dat, aes(x = proposal, y = length_abs_rel_diff_pct)) +
  geom_boxplot(width = 0.45, outlier.shape = NA) +
  geom_jitter(width = 0.08, height = 0, alpha = 0.7, size = 2) +
  labs(
    x = NULL, y = "Absolute difference from Morphometrix (%)",
    title = "Measurement accuracy with and without an automatic proposal"
  ) +
  theme_classic(base_size = 13)

save_plot(p_anchor_abs, "anchoring_absolute_error.png", 6.5, 5)

# 6. AUTOMATIC AND FINAL MEASUREMENT ERRORS
# Thesis: Secondary check for a relationship between proposal and final errors.
# Use only frames with an available automatic estimate. A correlation is
# compatible with anchoring, but both errors also share the same source image.
auto_subset <- dat %>%
  filter(
    !is.na(tool_auto_px), is.finite(tool_auto_px), is.finite(tool_L_px), is.finite(mmx_L_px)
  ) %>%
  mutate(
    automatic_error_px = tool_auto_px - mmx_L_px,
    final_error_px = tool_L_px - mmx_L_px,
    automatic_abs_relative_error_pct = 100 * abs(automatic_error_px) / mmx_L_px,
    final_abs_relative_error_pct = 100 * abs(final_error_px) / mmx_L_px
  )

if (nrow(auto_subset) >= 3) {
  auto_final_cor <- cor.test(
    auto_subset$automatic_error_px, auto_subset$final_error_px, method = "pearson"
  )
  
  auto_vs_final <- tibble(
    n = nrow(auto_subset), automatic_MAPE_pct =
      mean(auto_subset$automatic_abs_relative_error_pct),
    final_MAPE_pct = mean(auto_subset$final_abs_relative_error_pct),
    error_correlation_r = unname(auto_final_cor$estimate),
    error_correlation_p = auto_final_cor$p.value
  )
  
  cat("\n=== SECONDARY ANCHORING CHECK ===\n")
  print(auto_vs_final)
  
  save_table(auto_vs_final, "automatic_vs_final_error.csv")
  
  p_auto_error <- ggplot(auto_subset, aes(x = automatic_error_px, y = final_error_px)) +
    geom_point(size = 2.5, alpha = 0.8) +
    geom_hline(yintercept = 0, linetype = 2) +
    geom_vline(xintercept = 0, linetype = 2) +
    geom_smooth(method = "lm", se = TRUE) +
    labs(
      x = "Automatic proposal error vs Morphometrix (px)",
      y = "Final measurement error vs Morphometrix (px)",
      title = "Does the final measurement follow the automatic proposal?"
    ) +
    theme_classic(base_size = 13)
  
  save_plot(p_auto_error, "automatic_vs_final_error.png", 6.5, 5.5)
}
