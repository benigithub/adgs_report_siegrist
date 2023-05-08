response <- half_hourly_fluxes$GPP_NT_VUT_REF
predictors <- c("TA_F", "SW_IN_F", "LW_IN_F", "VPD_F", "PA_F", "P_F", "WS_F", "TA_F_MDS", "SW_IN_F_MDS", "LW_IN_F_MDS", "VPD_F_MDS", "CO2_F_MDS", "PPFD_IN", "USTAR")

# initialize the minimum AIC value and the best fitting model
min_aic <- Inf
best_model <- NULL

# loop through all possible models with 1 to 14 predictors
for (i in 1:14) {
  # get all possible combinations of i predictors
  comb <- combn(predictors, i)
  # loop through each combination of predictors
  for (j in 1:ncol(comb)) {
    # create the formula string with the predictors
    formel <- formula(paste("GPP_NT_VUT_REF ~", paste(comb[, j], collapse = "+")))
    # fit the linear model
    model <- lm(formel, data = half_hourly_fluxes)
    # compute the AIC value
    aic <- AIC(model)
    # check if the AIC value is the lowest so far
    if (aic < min_aic) {
      min_aic <- aic
      best_model <- model
    } else {
      break
    }
  }
}

# print the best fitting model and its AIC value
summary(best_model)
cat("AIC value:", min_aic, "\n")
