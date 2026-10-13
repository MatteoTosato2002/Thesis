library(tidyverse)

theme_set(theme_classic(base_size = 12))

# 1. FLIGHT-LEVEL CORRECTIONS
# Each row of the already merged table represents one flight.
ba <- read_csv("")

# 2. BLAND-ALTMAN AGREEMENT
# Differences are reference-object minus LiDAR correction, in METRES.
n <- nrow(ba)
bias <- mean(ba$difference_m, na.rm = TRUE)
sd_diff <- sd(ba$difference_m, na.rm = TRUE)

loa_lower <- bias - 1.96 * sd_diff
loa_upper <- bias + 1.96 * sd_diff

# Absolute discrepancies are also expressed as a percentage of flight altitude.
mae <- mean(ba$absolute_difference_m, na.rm = TRUE)

mean_percent_discrepancy <- mean(ba$absolute_difference_percent_altitude, na.rm = TRUE)

max_percent_discrepancy <- max(ba$absolute_difference_percent_altitude, na.rm = TRUE)

# The bias confidence interval uses the flight-level sample size.
se_bias <- sd_diff / sqrt(n)
t_crit <- qt(0.975, df = n - 1)

bias_ci_lower <- bias - t_crit * se_bias
bias_ci_upper <- bias + t_crit * se_bias

# 3. AGREEMENT DIAGNOSTICS
# Check normality of differences and proportional bias against the method mean.
shapiro_result <- shapiro.test(ba$difference_m)

prop_bias_model <- lm(difference_m ~ mean_correction_m, data = ba)

prop_bias_p <- summary(prop_bias_model)$coefficients[
  "mean_correction_m", "Pr(>|t|)"
]

# Report estimates, confidence intervals and diagnostic p-values.
cat(
  "n =", n, "\n", "Bias =", round(bias, 4), "m\n", "95% CI bias =",
  round(bias_ci_lower, 4), "to", round(bias_ci_upper, 4), "m\n",
  "SD differences =", round(sd_diff, 4), "m\n", "Lower LoA =", round(loa_lower, 4), "m\n",
  "Upper LoA =", round(loa_upper, 4), "m\n", "MAE =", round(mae, 4), "m\n",
  "Mean absolute discrepancy =", round(mean_percent_discrepancy, 2), "% of flight altitude\n",
  "Maximum absolute discrepancy =",
  round(max_percent_discrepancy, 2), "% of flight altitude\n",
  "Shapiro-Wilk p =", round(shapiro_result$p.value, 4), "\n",
  "Proportional-bias p =", round(prop_bias_p, 4), "\n"
)

# 4. BLAND-ALTMAN PLOT
p_ba <- ggplot(ba, aes(x = mean_correction_m, y = difference_m)) +
  geom_point(size = 3) +
  geom_hline(yintercept = bias, linewidth = 0.8) +
  geom_hline(yintercept = loa_lower, linetype = "dashed", linewidth = 0.7) +
  geom_hline(yintercept = loa_upper, linetype = "dashed", linewidth = 0.7) +
  annotate(
    "text", x = Inf, y = bias, label = paste0("Bias = ", round(bias, 2), " m"), hjust = 1.05,
    vjust = -0.5
  ) +
  annotate(
    "text", x = Inf, y = loa_upper-0.1,
    label = paste0("+1.96 SD = ", round(loa_upper, 2), " m"), hjust = 1.05, vjust = -0.5
  ) +
  annotate(
    "text", x = Inf, y = loa_lower+0.1,
    label = paste0("-1.96 SD = ", round(loa_lower, 2), " m"), hjust = 1.05, vjust = 1.5
  ) +
  labs(
    x = "Mean correction of the two methods (m)",
    y = "Reference-object correction - LiDAR correction (m)",
    title = "Bland-Altman comparison of altitude-correction methods"
  ) +
  theme_classic(base_size = 13)

p_ba