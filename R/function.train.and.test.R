
# The function need the number of k as input!
knn.model <- function(database_train, number.of.k){

  # Model and pre-processing formulation, use all variables but LW_IN_F
  pp <- recipes::recipe(GPP_NT_VUT_REF ~ SW_IN_F + VPD_F + TA_F,
                        data = database_train |> drop_na()) |>
    recipes::step_BoxCox(all_predictors()) |>
    recipes::step_center(all_numeric(), -all_outcomes()) |>
    recipes::step_scale(all_numeric(), -all_outcomes())

  # Fit KNN model
  mod_knn <- caret::train(pp, data = database_train |>
    drop_na(), method = "knn", trControl = caret::trainControl(method = "none"),
    tuneGrid = data.frame(k = number.of.k), metric = "RMSE")
  return(mod_knn)
}

# function do not need any input. You can call it without a parameter
lm.model <- function(database_train){
  # Model and pre-processing formulation, use all variables but LW_IN_F
  pp <- recipes::recipe(GPP_NT_VUT_REF ~ SW_IN_F + VPD_F + TA_F,
                        data = database_train |> drop_na()) |>
    recipes::step_BoxCox(all_predictors()) |>
    recipes::step_center(all_numeric(), -all_outcomes()) |>
    recipes::step_scale(all_numeric(), -all_outcomes())

  # Fit linear regression model
  mod_lm <- caret::train(pp, data = database_train|>
    drop_na(), method = "lm", trControl = caret::trainControl(method = "none"),
    metric = "RMSE")
  return(mod_lm)
}
