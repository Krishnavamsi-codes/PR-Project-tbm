# Results (mean ± std over 5 runs)

## All methods

| Group                       | Method                    | Accuracy        | MacroF1         |   Seconds |
|:----------------------------|:--------------------------|:----------------|:----------------|----------:|
| 1 Baseline (raw)            | DecisionTree              | 0.6853 ± 0.0006 | 0.6043 ± 0.0085 |         4 |
| 1 Baseline (raw)            | LDA                       | 0.5874 ± 0.0016 | 0.4889 ± 0.0017 |         7 |
| 1 Baseline (raw)            | NaiveBayes                | 0.4211 ± 0.0019 | 0.3994 ± 0.0027 |         6 |
| 1 Baseline (raw)            | QDA                       | 0.4188 ± 0.0034 | 0.3914 ± 0.0046 |         7 |
| 2 Feature extraction        | LDA + te_eng              | 0.7306 ± 0.0020 | 0.6826 ± 0.0028 |         8 |
| 2 Feature extraction        | NaiveBayes + te_eng_lda   | 0.7319 ± 0.0024 | 0.6814 ± 0.0037 |         7 |
| 2 Feature extraction        | LDA + te_eng_mi           | 0.7285 ± 0.0017 | 0.6799 ± 0.0024 |       320 |
| 2 Feature extraction        | LDA + te_eng_fis          | 0.7282 ± 0.0017 | 0.6792 ± 0.0024 |        11 |
| 2 Feature extraction        | LDA + te                  | 0.7275 ± 0.0017 | 0.6769 ± 0.0027 |         6 |
| 2 Feature extraction        | DecisionTree + te_eng_fis | 0.7330 ± 0.0009 | 0.6742 ± 0.0035 |        19 |
| 2 Feature extraction        | DecisionTree + te_eng_mi  | 0.7326 ± 0.0006 | 0.6734 ± 0.0032 |       180 |
| 2 Feature extraction        | DecisionTree + te_eng     | 0.7325 ± 0.0023 | 0.6723 ± 0.0056 |         8 |
| 2 Feature extraction        | DecisionTree + te         | 0.7328 ± 0.0025 | 0.6706 ± 0.0028 |         7 |
| 2 Feature extraction        | LDA + te_eng_pca          | 0.7193 ± 0.0016 | 0.6676 ± 0.0023 |         5 |
| 2 Feature extraction        | DecisionTree + te_eng_lda | 0.7311 ± 0.0017 | 0.6673 ± 0.0052 |         8 |
| 2 Feature extraction        | kNN + te_eng_lda          | 0.7205 ± 0.0018 | 0.6581 ± 0.0021 |        11 |
| 2 Feature extraction        | DecisionTree + te_eng_pca | 0.7189 ± 0.0007 | 0.6500 ± 0.0042 |        19 |
| 2 Feature extraction        | NaiveBayes + te_eng_mi    | 0.4789 ± 0.0055 | 0.4763 ± 0.0060 |       136 |
| 2 Feature extraction        | NaiveBayes + te_eng_fis   | 0.4655 ± 0.0020 | 0.4600 ± 0.0019 |         8 |
| 2 Feature extraction        | NaiveBayes + te           | 0.4465 ± 0.0022 | 0.4344 ± 0.0024 |         4 |
| 2 Feature extraction        | NaiveBayes + te_eng       | 0.4438 ± 0.0018 | 0.4321 ± 0.0020 |         5 |
| 2 Feature extraction        | NaiveBayes + te_eng_pca   | 0.4589 ± 0.0064 | 0.4212 ± 0.0074 |         4 |
| 3 Modified methods          | CostSens-LDA              | 0.7256 ± 0.0038 | 0.6853 ± 0.0030 |        29 |
| 3 Modified methods          | CostSens-QDA              | 0.5573 ± 0.0097 | 0.5458 ± 0.0085 |        35 |
| 3 Modified methods          | CostSens-NaiveBayes       | 0.4461 ± 0.0018 | 0.4351 ± 0.0020 |        17 |
| 4 Deep learning (reference) | MLP (te_eng)              | 0.7415 ± 0.0017 | 0.6861 ± 0.0035 |        75 |
| 4 Deep learning (reference) | MLP (raw)                 | 0.6799 ± 0.0021 | 0.6106 ± 0.0049 |       160 |

## Best per group: per-class precision / recall

| Group                       | Method       | Class            |   Precision |   Recall |
|:----------------------------|:-------------|:-----------------|------------:|---------:|
| 1 Baseline (raw)            | DecisionTree | Grade 1 (low)    |      0.6436 |   0.3558 |
| 1 Baseline (raw)            | DecisionTree | Grade 2 (medium) |      0.6882 |   0.8267 |
| 1 Baseline (raw)            | DecisionTree | Grade 3 (high)   |      0.687  |   0.54   |
| 2 Feature extraction        | LDA + te_eng | Grade 1 (low)    |      0.5785 |   0.597  |
| 2 Feature extraction        | LDA + te_eng | Grade 2 (medium) |      0.7437 |   0.819  |
| 2 Feature extraction        | LDA + te_eng | Grade 3 (high)   |      0.756  |   0.619  |
| 3 Modified methods          | CostSens-LDA | Grade 1 (low)    |      0.5532 |   0.6442 |
| 3 Modified methods          | CostSens-LDA | Grade 2 (medium) |      0.7614 |   0.773  |
| 3 Modified methods          | CostSens-LDA | Grade 3 (high)   |      0.7226 |   0.6687 |
| 4 Deep learning (reference) | MLP (te_eng) | Grade 1 (low)    |      0.6667 |   0.5135 |
| 4 Deep learning (reference) | MLP (te_eng) | Grade 2 (medium) |      0.7414 |   0.8479 |
| 4 Deep learning (reference) | MLP (te_eng) | Grade 3 (high)   |      0.762  |   0.6262 |

## Ablation

| Classifier   | Step           |   MacroF1 |    Acc |
|:-------------|:---------------|----------:|-------:|
| kNN          | +LDA proj      |    0.6581 | 0.7205 |
| DecisionTree | raw            |    0.6043 | 0.6853 |
| DecisionTree | +target enc    |    0.6706 | 0.7328 |
| DecisionTree | +te+eng        |    0.6723 | 0.7325 |
| DecisionTree | +MI top-20     |    0.6734 | 0.7326 |
| DecisionTree | +Fisher top-20 |    0.6742 | 0.733  |
| DecisionTree | +PCA 95%       |    0.65   | 0.7189 |
| DecisionTree | +LDA proj      |    0.6673 | 0.7311 |
| NaiveBayes   | raw            |    0.3994 | 0.4211 |
| NaiveBayes   | +target enc    |    0.4344 | 0.4465 |
| NaiveBayes   | +te+eng        |    0.4321 | 0.4438 |
| NaiveBayes   | +MI top-20     |    0.4763 | 0.4789 |
| NaiveBayes   | +Fisher top-20 |    0.46   | 0.4655 |
| NaiveBayes   | +PCA 95%       |    0.4212 | 0.4589 |
| NaiveBayes   | +LDA proj      |    0.6814 | 0.7319 |

## Significance (paired t-test and Nadeau-Bengio corrected t-test)

| Comparison           | A            | B            | Metric   |   MeanDiff |   p_paired |   p_corrected |
|:---------------------|:-------------|:-------------|:---------|-----------:|-----------:|--------------:|
| Feature vs Baseline  | LDA + te_eng | DecisionTree | macro-F1 |     0.0783 |    3.8e-05 |       0.00019 |
| Feature vs Baseline  | LDA + te_eng | DecisionTree | accuracy |     0.0453 |    2.3e-06 |       1.1e-05 |
| Modified vs Baseline | CostSens-LDA | DecisionTree | macro-F1 |     0.081  |    7.2e-05 |       0.00035 |
| Modified vs Baseline | CostSens-LDA | DecisionTree | accuracy |     0.0403 |    1.7e-05 |       8.7e-05 |
| Modified vs Feature  | CostSens-LDA | LDA + te_eng | macro-F1 |     0.0027 |    0.056   |       0.15    |
| Modified vs Feature  | CostSens-LDA | LDA + te_eng | accuracy |    -0.005  |    0.049   |       0.13    |
| Modified vs MLP      | CostSens-LDA | MLP (te_eng) | macro-F1 |    -0.0008 |    0.48    |       0.63    |
| Modified vs MLP      | CostSens-LDA | MLP (te_eng) | accuracy |    -0.0158 |    0.0012  |       0.0052  |
| Feature vs MLP       | LDA + te_eng | MLP (te_eng) | macro-F1 |    -0.0035 |    0.075   |       0.19    |
| Feature vs MLP       | LDA + te_eng | MLP (te_eng) | accuracy |    -0.0108 |    6.1e-06 |       3.1e-05 |