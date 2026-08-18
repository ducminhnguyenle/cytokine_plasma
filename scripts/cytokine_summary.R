library(gtsummary)
library(flextable)
library(tidyverse)

# Load cytokine data
cytokine_df <- read_csv("data/CV_plasma_Combined_2025_11_21_AN-PD_raw.csv")

# Format categorial variables
cytokine_df <- cytokine_df %>%
    mutate(
        Pack_years_label = if_else(Pack_years < 10, "less than 10 pack-years", "10 or more pack-years"),
    ) %>%
    mutate(
        Cohort = factor(Cohort),
        Gender = factor(Gender, levels = c(1, 2), labels = c("male", "female")),
        Race = factor(Race, levels = c(1:6), labels = c("white non-Hispanic", "white Hispanic", "black non-Hispanic", "black Hispanic", "Asian", "American Indian")),
        Tobacco_use = factor(Tobacco_use, levels = c(1, 2, 3), labels = c("never", "former", "current")),
        Pack_years_label = factor(Pack_years_label, levels = c("less than 10 pack-years", "10 or more pack-years")),
        Alcohol_use = factor(Alcohol_use, levels = c(1, 2, 3), labels = c("never", "former", "current")),
        Pathologic_stage = factor(Pathologic_stage, levels = c(1, 2, 3, 4, 5), labels = c("stage I", "stage II", "stage III", "stage IV", "stage V")),
        LVI = factor(LVI, levels = c(0, 1), labels = c("no", "yes")),
        PNI = factor(PNI, levels = c(0, 1), labels = c("no", "yes")),
    )

cytokine_df %>%
    arrange(sample_names, Cohort) %>%
    write_csv("data/CV_plasma_Combined_2025_11_21_AN-PD_formatted.csv")

# Cytokines with missing values
cytokines_missing <- cytokine_df %>%
    select(where(is.numeric) & where(~anyNA(.))) %>%
    select(-Pack_years) %>%
    colnames()

# Cytokines with complete values
cytokines_complete <- cytokine_df %>%
    select(where(is.numeric)) %>%
    select(-c(any_of(cytokines_missing), Ages, Pack_years)) %>%
    colnames()

# Summarise the median (IQR) of 16 cytokines
cytokine_summary_table <- cytokine_df %>%
    tbl_summary(
        include = cytokines_complete,
        by = Cohort,
        statistic = list(
            all_continuous() ~ "{median} ({p25}-{p75})",
            all_categorical() ~ "{n} / {N} ({p}%)"
        ),
        missing_text = "Missing",
        digits = all_continuous() ~ 2
    ) %>%
    # add_overall() %>%
    add_p(
        pvalue_fun = function(x) style_pvalue(x, digits = 2)
    ) %>%
    modify_spanning_header(c("stat_1", "stat_2") ~ "**Cohort**") %>%
    bold_labels() %>%
    bold_p() %>%
    italicize_levels() %>%
    as_flex_table() %>%
    flextable::save_as_docx(path = "summary_stats/cytokine_summary.docx")
    # as_gt() %>%
    # gt::gtsave("summary_stats/cytokine_summary.html")

# Summary statistics for cytokines with missing values
cytokines_missing_summary <- cytokine_df %>%
    select(all_of(cytokines_missing), Cohort) %>%
    group_by(Cohort) %>%
    summarise(
        across(everything(), list(
            median = ~round(median(., na.rm = TRUE), 2),
            p25 = ~round(quantile(., 0.25, na.rm = TRUE), 2),
            p75 = ~round(quantile(., 0.75, na.rm = TRUE), 2),
            missing_count = ~sum(is.na(.)),
            total_count = ~n()
        ), .names = "{col}_{fn}")
    )

# Summarise demographic and clinical variables
demographic_clinical_vars <- c("Gender", "Race", "Tobacco_use", "Pack_years_label", "Alcohol_use", "Pathologic_stage", "LVI", "PNI")
demographic_summary_table <- cytokine_df %>%
    tbl_summary(
        include = demographic_clinical_vars,
        by = Cohort,
        statistic = list(
            all_continuous() ~ "{median} ({p25}-{p75})",
            all_categorical() ~ "{n} / {N} ({p}%)"
        ),
        missing_text = "Missing",
        digits = all_continuous() ~ 2
    ) %>%
    add_overall() %>%
    add_p(
        pvalue_fun = function(x) style_pvalue(x, digits = 2)
    ) %>%
    modify_spanning_header(c("stat_1", "stat_2") ~ "**Cohort**") %>%
    bold_labels() %>%
    bold_p() %>%
    italicize_levels() %>%
    as_flex_table() %>%
    flextable::save_as_docx(path = "summary_stats/demographic_clinical_summary.docx")
    # as_gt() %>%
    # gt::gtsave("summary_stats/demographic_clinical_summary.html")

cytokine_df %>%
    select(demographic_clinical_vars, Cohort) %>%
    group_by(Cohort) %>%
    count(PNI)
