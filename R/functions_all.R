
library(tidyverse)
library(ggplot2)
library(dplyr)
library(rsample)
library(recipes)
daily_fluxes <- read_csv("./data/FLX_CH-Dav_FLUXNET2015_FULLSET_DD_1997-2014_1-3.csv") |>  
  
# select only the variables we are interested in
dplyr::select(TIMESTAMP,
                GPP_NT_VUT_REF,    # the target
                ends_with("_QC"),  # quality control info
                ends_with("_F"),   # includes all all meteorological covariates
                -contains("JSB")   # weird useless variable
) |>
  
# convert to a nice date object
dplyr::mutate(TIMESTAMP = ymd(TIMESTAMP)) |>
  
# set all -9999 to NA
mutate(across(where(is.numeric), ~na_if(., -9999))) |> # NOTE: Newer tidyverse version no longer support this statement
  # instead, use `mutate(across(where(is.numeric), ~na_if(., -9999))) |> `
  
  # retain only data based on >=80% good-quality measurements
  # overwrite bad data with NA (not dropping rows)
dplyr::mutate(GPP_NT_VUT_REF = ifelse(NEE_VUT_REF_QC < 0.8, NA, GPP_NT_VUT_REF),
                TA_F           = ifelse(TA_F_QC        < 0.8, NA, TA_F),
                SW_IN_F        = ifelse(SW_IN_F_QC     < 0.8, NA, SW_IN_F),
                LW_IN_F        = ifelse(LW_IN_F_QC     < 0.8, NA, LW_IN_F),
                VPD_F          = ifelse(VPD_F_QC       < 0.8, NA, VPD_F),
                PA_F           = ifelse(PA_F_QC        < 0.8, NA, PA_F),
                P_F            = ifelse(P_F_QC         < 0.8, NA, P_F),
                WS_F           = ifelse(WS_F_QC        < 0.8, NA, WS_F)) |> 
  
  # drop QC variables (no longer needed)
dplyr::select(-ends_with("_QC"))

#formula notion linear regression

lm(GPP_NT_VUT_REF ~ SW_IN_F + VPD_F + TA_F, data = daily_fluxes)


#train for a method (here lm)
caret::train(
  form = GPP_NT_VUT_REF ~ SW_IN_F + VPD_F + TA_F, 
  data = daily_fluxes |> drop_na(),  # drop missing values
  trControl = caret::trainControl(method = "none"),  # no resampling
  method = "lm"
)

#here knn
caret::train(
  form = GPP_NT_VUT_REF ~ SW_IN_F + VPD_F + TA_F, 
  data = daily_fluxes |> drop_na(), 
  trControl = caret::trainControl(method = "none"),
  method = "knn"
)
  
#data splitting for testing, splits are usually 60-80%
set.seed(123)  # for reproducibility
split <- rsample::initial_split(daily_fluxes, prop = 0.7, strata = "VPD_F")
daily_fluxes_train <- rsample::training(split)
daily_fluxes_test <- rsample::testing(split)

#plot the data of training/testing sets
plot_data <- daily_fluxes_train |> 
  dplyr::mutate(split = "train") |> 
  dplyr::bind_rows(daily_fluxes_test |> 
                     dplyr::mutate(split = "test")) |> 
  tidyr::pivot_longer(cols = 2:9, names_to = "variable", values_to = "value")

plot_data |> 
  ggplot(aes(x = value, y = ..density.., color = split)) +
  geom_density() +
  facet_wrap(~variable, scales = "free")


#preprocessing

pp <- recipes::recipe(GPP_NT_VUT_REF ~ SW_IN_F + VPD_F + TA_F, data = daily_fluxes_train) |> 
  recipes::step_center(all_numeric(), -all_outcomes()) |>
  recipes::step_scale(all_numeric(), -all_outcomes())

caret::train(
  pp, 
  data = daily_fluxes_train, 
  method = "knn",
  trControl = caret::trainControl(method = "none")
)

#standariation

daily_fluxes |> 
  summarise(across(where(is.numeric), ~quantile(.x, probs = c(0, 0.25, 0.5, 0.75, 1), na.rm = TRUE))) |> 
  t() |> 
  as_tibble(rownames = "variable") |> 
  setNames(c("variable", "min", "q25", "q50", "q75", "max"))

#prepare recipe
pp_prep <- recipes::prep(pp, training = daily_fluxes_train) 

#juice the prepared recipe
daily_fluxes_juiced <- recipes::juice(pp_prep)

daily_fluxes_baked <- recipes::bake(pp_prep, new_data = daily_fluxes_train)

# confirm that juice and bake return identical objects when given the same data
all_equal(daily_fluxes_juiced, daily_fluxes_baked)

# prepare data for plotting
plot_data_original <- daily_fluxes_train |> 
  dplyr::select(one_of(c("SW_IN_F", "VPD_F", "TA_F"))) |> 
  tidyr::pivot_longer(cols = c(SW_IN_F, VPD_F, TA_F), names_to = "var", values_to = "val")

plot_data_juiced <- daily_fluxes_juiced |> 
  dplyr::select(one_of(c("SW_IN_F", "VPD_F", "TA_F"))) |> 
  tidyr::pivot_longer(cols = c(SW_IN_F, VPD_F, TA_F), names_to = "var", values_to = "val")

# plot density
plot_1 <- ggplot(data = plot_data_original, aes(val, ..density..)) +
  geom_density() +
  facet_wrap(~var)

# plot density by var
plot_2 <- ggplot(data = plot_data_juiced, aes(val, ..density..)) +
  geom_density() +
  facet_wrap(~var)

# combine both plots
cowplot::plot_grid(plot_1, plot_2, nrow = 2)

#missing data visualisation
visdat::vis_miss(
  daily_fluxes,
  cluster = FALSE, 
  warn_large_data = FALSE
)

#imputation = replacing missing values with best guess, here median.
pp |> 
  step_impute_median(all_predictors())

#impute, use KNN with five neighbours
pp |> 
  step_impute_knn(all_predictors(), neighbors = 5)





# one hot endocing
# original data frame
df <- tibble(id = 1:4, color = c("red", "red", "green", "blue"))
df

# after one-hot encoding
dmy <- dummyVars("~ .", data = df, sep = "_")
data.frame(predict(dmy, newdata = df))

##target engeneering

getwd()

plot_1 <- ggplot(data = daily_fluxes, aes(x = WS_F, y = ..density..)) +
  geom_histogram() +
  labs(title = "Original")

plot_2 <- ggplot(data = daily_fluxes, aes(x = log(WS_F), y = ..density..)) +
  geom_histogram() +
  labs(title = "Log-transformed")

cowplot::plot_grid(plot_1, plot_2)

recipes::recipe(WS_F ~ ., data = daily_fluxes) |>   # it's of course non-sense to model wind speed like this
  recipes::step_log(all_outcomes())$
  





##putting it all together


daily_fluxes |> 
  ggplot(aes(x = GPP_NT_VUT_REF, y = ..count..)) + 
  geom_histogram()



# Data splitting
set.seed(1982)  # for reproducibility
split <- rsample::initial_split(daily_fluxes, prop = 0.7, strata = "VPD_F")
daily_fluxes_train <- rsample::training(split)
daily_fluxes_test <- rsample::testing(split)

# Model and pre-processing formulation, use all variables but LW_IN_F
pp <- recipes::recipe(GPP_NT_VUT_REF ~ SW_IN_F + VPD_F + TA_F, 
                      data = daily_fluxes_train |> drop_na()) |> 
  recipes::step_BoxCox(all_predictors()) |> 
  recipes::step_center(all_numeric(), -all_outcomes()) |>
  recipes::step_scale(all_numeric(), -all_outcomes())






# Fit linear regression model
mod_lm <- caret::train(
  pp, 
  data = daily_fluxes_train |> drop_na(), 
  method = "lm",
  trControl = caret::trainControl(method = "none"),
  metric = "RMSE"
)

# Fit KNN model
mod_knn <- caret::train(
  pp, 
  data = daily_fluxes_train |> drop_na(), 
  method = "knn",
  trControl = caret::trainControl(method = "none"),
  tuneGrid = data.frame(k = 8),
  metric = "RMSE"
)


# make model evaluation into a function to reuse code
eval_model <- function(mod, df_train, df_test){
  
  # add predictions to the data frames
  df_train <- df_train |> 
    drop_na()
  df_train$fitted <- predict(mod, newdata = df_train)
  
  df_test <- df_test |> 
    drop_na()
  df_test$fitted <- predict(mod, newdata = df_test)
  
  # get metrics tables
  metrics_train <- df_train |> 
    yardstick::metrics(GPP_NT_VUT_REF, fitted)
  
  metrics_test <- df_test |> 
    yardstick::metrics(GPP_NT_VUT_REF, fitted)
  
  # extract values from metrics tables
  rmse_train <- metrics_train |> 
    filter(.metric == "rmse") |> 
    pull(.estimate)
  rsq_train <- metrics_train |> 
    filter(.metric == "rsq") |> 
    pull(.estimate)
  
  rmse_test <- metrics_test |> 
    filter(.metric == "rmse") |> 
    pull(.estimate)
  rsq_test <- metrics_test |> 
    filter(.metric == "rsq") |> 
    pull(.estimate)
  
  # visualise as a scatterplot
  # adding information of metrics as sub-titles
  plot_1 <- ggplot(data = df_train, aes(GPP_NT_VUT_REF, fitted)) +
    geom_point(alpha = 0.3) +
    geom_smooth(method = "lm", se = FALSE, color = "red") +
    geom_abline(slope = 1, intercept = 0, linetype = "dotted") +
    labs(subtitle = bquote( italic(R)^2 == .(format(rsq_train, digits = 2)) ~~
                              RMSE == .(format(rmse_train, digits = 3))),
         title = "Training set") +
    theme_classic()
  
  plot_2 <- ggplot(data = df_test, aes(GPP_NT_VUT_REF, fitted)) +
    geom_point(alpha = 0.3) +
    geom_smooth(method = "lm", se = FALSE, color = "red") +
    geom_abline(slope = 1, intercept = 0, linetype = "dotted") +
    labs(subtitle = bquote( italic(R)^2 == .(format(rsq_test, digits = 2)) ~~
                              RMSE == .(format(rmse_test, digits = 3))),
         title = "Test set") +
    theme_classic()
  
  out <- cowplot::plot_grid(plot_1, plot_2)
  
  return(out)
}

# linear regression model
eval_model(mod = mod_lm, df_train = daily_fluxes_train, df_test = daily_fluxes_test)

# KNN
eval_model(mod = mod_knn, df_train = daily_fluxes_train, df_test = daily_fluxes_test)