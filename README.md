# Explainable EvoX-Boost: A SHAP-Based Framework for Clinically Interpretable Multiclass Thyroid Disease Diagnosis

## Overview

**Explainable EvoX-Boost** is a machine learning framework developed for multiclass thyroid disease diagnosis using clinical and laboratory features.

The framework combines optimized XGBoost classification with SHapley Additive exPlanations (SHAP) to investigate model predictions and improve their interpretability.

The research focuses on developing a diagnostic classification pipeline and examining how explainable artificial intelligence can provide insights into the model's predictions.

This repository accompanies the research manuscript:

> **Explainable EvoX-Boost: A SHAP-Based Framework for Clinically Interpretable Multiclass Thyroid Disease Diagnosis**

## Research Objectives

The primary objectives of this research are:

1. Develop an optimized XGBoost-based framework for multiclass thyroid disease classification.
2. Investigate model performance using structured clinical and laboratory data.
3. Integrate SHAP-based explainability to analyze feature contributions to model predictions.
4. Examine global, class-specific, interaction, and individual prediction explanations.
5. Evaluate classification performance and explanation stability using experimental validation.

## Dataset

The study uses a thyroid disease dataset containing:

| Dataset characteristic | Description |
|---|---|
| Total records | 22,632 |
| Original disease classes | 15 |
| Data type | Clinical and laboratory attributes |
| Prediction task | Multiclass thyroid disease classification |

The dataset contains thyroid-related clinical and laboratory information used to train and evaluate the classification framework.

### Dataset access

The original dataset is not automatically distributed with this repository.

Please refer to the dataset source and its original terms of use before downloading or using the data.

**Dataset source:** To be added with the verified original dataset URL and citation.

Processed data, if included, will be documented separately. Dataset redistribution will follow the original source's licensing and usage conditions.

## Methodology

The framework consists of the following experimental stages:

1. **Data preprocessing**  
   Prepare the clinical dataset for machine learning analysis.

2. **Model development**  
   Develop and optimize the XGBoost-based multiclass classification model.

3. **Experimental validation**  
   Evaluate model performance using repeated stratified five-fold cross-validation and an independent 20% held-out test set.

4. **Explainability analysis**  
   Apply SHAP to investigate global feature importance, multiclass explanations, feature interactions, and individual predictions.

5. **Performance and stability analysis**  
   Examine classification performance and the consistency of explanations across experimental evaluations.

## Model Evaluation

The manuscript reports the following experimental results:

| Evaluation metric | Reported result |
|---|---:|
| Accuracy | 98.85% |
| Observed-class macro F1-score | 74.18% |
| Full 15-class macro F1-score | 64.29% |
| Balanced accuracy over observed 13 classes | 71.94% |

These metrics should be interpreted together.

The high overall accuracy should not be considered sufficient on its own because the dataset contains class imbalance. Macro-level metrics provide additional information about performance across disease categories.

The reported results correspond to the experimental setup described in the manuscript. They should not be interpreted as independent clinical validation.

## Explainable Artificial Intelligence

SHAP is used to investigate how input features contribute to the model's predictions.

The explainability analysis includes:

- **Global feature importance:** Identifies features contributing to model predictions across the dataset.
- **Multiclass explanations:** Examines feature contributions for individual disease classes.
- **Feature interaction analysis:** Investigates relationships between features and their combined contributions.
- **Local explanations:** Examines feature contributions for individual predictions.
- **Explanation stability:** Investigates the consistency of feature explanations across experimental evaluations.

These analyses are intended to improve transparency and support the interpretation of model behavior.

SHAP explanations describe the behavior of the trained model. They do not establish causal relationships or replace clinical judgment.

## Experimental Validation

The manuscript describes the following validation components:

- Repeated stratified five-fold cross-validation.
- Independent 20% held-out test evaluation.
- Classification performance analysis.
- Explanation stability analysis.
- Calibration analysis.

The repository will provide the corresponding implementation, experimental configurations, and results where these materials are available and approved for public release.

## Requirements

The implementation uses Python-based machine learning and data analysis tools.

The final dependency list will be provided in `requirements.txt` after verifying the actual environment used for the experiments.

Please use the package versions specified in the repository when reproducing the experiments.

## Reproducibility

To support reproducibility, the repository will include available experimental materials such as:

- Data preprocessing procedures.
- Model training and evaluation notebooks.
- Model configurations.
- SHAP analysis procedures.
- Evaluation metrics and result tables.
- Experimental environment information.

The exact execution instructions will be provided after the source files and dependencies have been verified.

**Note:** Results may vary depending on dataset versions, preprocessing, class distribution, random seeds, software versions, and hardware configuration.

## Responsible Use

This project is intended for academic research and methodological investigation.

The framework is not a clinically validated diagnostic system and should not be used as a substitute for professional medical diagnosis, treatment, or clinical decision-making.

The dataset and research outputs must be used in accordance with their applicable terms and conditions.

## Manuscript

**Research title:**

*Explainable EvoX-Boost: A SHAP-Based Framework for Clinically Interpretable Multiclass Thyroid Disease Diagnosis*

The manuscript provides the detailed methodology, experimental design, evaluation, and explainability analysis associated with this repository.

The final publication citation and DOI will be added when the manuscript is officially published.

## Citation

If you use this repository in academic work, please cite the associated manuscript after its publication.

The final citation information will be added to `CITATION.cff`.

## License

The repository's license and permitted use are specified in the `LICENSE` file.

Please review the license before using, modifying, or redistributing any repository materials.

## Contact

For research-related questions, please use the contact information provided in the associated manuscript or the repository's GitHub discussion and issue facilities.
