
library(tidyverse)
library(ggplot2)
library(dplyr)
library(rsample)
library(caret)
library(rsample)
library(recipes)
library(cowplot)


#loading laengen data
daily_fluxes_lae <- read_csv("FLX_CH-Dav_FLUXNET2015_FULLSET_DD_1997-2014_1-3.csv") |>

dplyr::select(TIMESTAMP,
              GPP_NT_VUT_REF,    # the target
              ends_with("_QC"),  # quality control info
              ends_with("_F"),   # includes all all meteorological covariates
              -contains("JSB")   # weird useless variable
) |>
  
  # convert to a nice date object
  dplyr::mutate(TIMESTAMP = ymd(TIMESTAMP)) |>
  
  # set all -9999 to NA
  mutate(across(where(is.numeric), ~na_if(., -9999))) |>
  
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






#loading davos data
daily_fluxes_davos <- read_csv("FLX_CH-Dav_FLUXNET2015_FULLSET_DD_1997-2014_1-3.csv") |>
  
  dplyr::select(TIMESTAMP,
                GPP_NT_VUT_REF,    # the target
                ends_with("_QC"),  # quality control info
                ends_with("_F"),   # includes all all meteorological covariates
                -contains("JSB")   # weird useless variable
  ) |>
  
  # convert to a nice date object
  dplyr::mutate(TIMESTAMP = ymd(TIMESTAMP)) |>
  
  # set all -9999 to NA
  mutate(across(where(is.numeric), ~na_if(., -9999))) |>
  
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


#data splitting
set.seed(123)  # for reproducibility
split <- rsample::initial_split(daily_fluxes_davos, prop = 0.8, strata = "VPD_F")
daily_fluxes_davos_train <- rsample::training(split)
daily_fluxes_davos_test <- rsample::testing(split)

split <- rsample::initial_split(daily_fluxes_lae, prop = 0.8, strata = "VPD_F")
daily_fluxes_lae_train <- rsample::training(split)
daily_fluxes_lae_test <- rsample::testing(split)

 
#model formulation davos
pp_davos <- recipes::recipe(GPP_NT_VUT_REF ~ SW_IN_F + VPD_F + TA_F, 
                                  data = daily_fluxes_davos_train) |> 
  recipes::step_center(all_numeric(), -all_outcomes()) |>
  recipes::step_scale(all_numeric(), -all_outcomes())


#model formulation laengen
pp_lae <- recipes::recipe(GPP_NT_VUT_REF ~ SW_IN_F + VPD_F + TA_F, 
                            data = daily_fluxes_lae_train) |> 
  recipes::step_center(all_numeric(), -all_outcomes()) |>
  recipes::step_scale(all_numeric(), -all_outcomes())



#10-fold cross validation
set.seed(1982)
davos_mod_cv <- caret::train(pp_davos,
                      data = daily_fluxes_davos_train |> drop_na(),
                      method = "knn",
                      trControl = caret::trainControl(method = "cv", number = 10),
                      tuneGrid = data.frame(k = c(2, 5, 10, 15 ,20 ,25 ,30 ,40 ,60 ,100)),
                      metric = "MAE")
set.seed(1982)
lae_mod_cv <- caret::train(pp_lae,
                        data = daily_fluxes_lae_train |> drop_na(),
                        method = "knn",
                        trControl = caret::trainControl(method = "cv", number = 10),
                        tuneGrid = data.frame(k = c(2, 5, 10, 15 ,20 ,25 ,30 ,40 ,60 ,100)),
                        metric = "MAE")


ggplot(davos_mod_cv)
ggplot(lae_mod_cv)
print(davos_mod_cv)
print(lae_mod_cv)


## metrics within site davos
davos_within_site <- eval_model(mod = davos_mod_cv, df_train = daily_fluxes_davos_train, df_test = daily_fluxes_davos_test)
print(davos_within_site)

#metrics within site laegern
laegern_within_site <- eval_model(mod = lae_mod_cv, df_train = daily_fluxes_lae_train, df_test = daily_fluxes_lae_test)
print(laegern_within_site)


#laegern acros site
laegern_across_site_eval <- eval_model(mod = davos_mod_cv,
                                       df_train = daily_fluxes_davos_train,
                                       df_test = daily_fluxes_lae_test)

print(laegern_across_site_eval)


#davos across site
davos_across_site_eval <- eval_model(mod = lae_mod_cv,
                                     df_train = daily_fluxes_lae_train,
                                     df_test = daily_fluxes_davos_test)
print(davos_across_site_eval)




####poooled things.
daily_fluxes_davos_test <- na.omit(daily_fluxes_davos_test)
daily_fluxes_lae_test <- na.omit(daily_fluxes_lae_test)

set.seed(1982)
pooled_train_data <- rbind(daily_fluxes_davos_train, daily_fluxes_lae_train)
pooled_train_data <- na.omit(pooled_train_data)

pp_pooled <- recipes::recipe(GPP_NT_VUT_REF ~ SW_IN_F + VPD_F + TA_F, 
                             data = pooled_train_data) |>
  recipes::step_center(all_numeric(), -all_outcomes()) |>
  recipes::step_scale(all_numeric(), -all_outcomes())

# Train the pooled model
pooled_mod <- caret::train(pp_pooled,
                           data = pooled_train_data,
                           method = "knn",
                           trControl = caret::trainControl(method = "cv", number = 10),
                           tuneGrid = data.frame(k = c(2, 5, 10, 15, 20, 25, 30, 40, 60, 100)),
                           metric = "MAE")




# Evaluate the model on the test data for both sites
pooled_predictions_davos <- predict(pooled_mod, newdata = daily_fluxes_davos_test)
pooled_metrics_davos <- eval_model(mod = pooled_mod,
                                   df_train = pooled_train_data,
                                   df_test = daily_fluxes_davos_test)
print(pooled_metrics_davos)



pooled_predictions_lae <- predict(pooled_mod, newdata = daily_fluxes_lae_test)
pooled_metrics_lae <- eval_model(mod = pooled_mod,
                                 df_train = pooled_train_data,
                                 df_test = daily_fluxes_lae_test)
print(pooled_metrics_lae)





