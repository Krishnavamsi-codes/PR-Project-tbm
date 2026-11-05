# Results (mean ± std over 15 runs)

## All methods

| Group                       | Method                         | Accuracy        | MacroF1         |   Seconds |
|:----------------------------|:-------------------------------|:----------------|:----------------|----------:|
| 1 Baseline (raw)            | DecisionTree                   | 0.6610 ± 0.0064 | 0.5808 ± 0.0091 |         1 |
| 1 Baseline (raw)            | kNN                            | 0.6012 ± 0.0059 | 0.5158 ± 0.0090 |         8 |
| 1 Baseline (raw)            | LDA                            | 0.5860 ± 0.0056 | 0.4955 ± 0.0062 |         1 |
| 1 Baseline (raw)            | SVM-RBF                        | 0.6116 ± 0.0043 | 0.4767 ± 0.0089 |        98 |
| 1 Baseline (raw)            | QDA                            | 0.4759 ± 0.0158 | 0.4518 ± 0.0193 |         1 |
| 1 Baseline (raw)            | NaiveBayes                     | 0.4366 ± 0.0394 | 0.4029 ± 0.0194 |         1 |
| 2 Feature extraction        | NaiveBayes + te_eng_lda        | 0.7094 ± 0.0062 | 0.6552 ± 0.0079 |         1 |
| 2 Feature extraction        | LDA + te_eng                   | 0.7075 ± 0.0061 | 0.6547 ± 0.0077 |         3 |
| 2 Feature extraction        | LDA + te                       | 0.7061 ± 0.0059 | 0.6506 ± 0.0079 |         3 |
| 2 Feature extraction        | LDA + te_eng_mi                | 0.7032 ± 0.0058 | 0.6503 ± 0.0073 |        11 |
| 2 Feature extraction        | LDA + te_eng_fis               | 0.7033 ± 0.0058 | 0.6499 ± 0.0076 |         1 |
| 2 Feature extraction        | LDA + te_eng_pca               | 0.6931 ± 0.0063 | 0.6389 ± 0.0075 |         1 |
| 2 Feature extraction        | SVM-RBF + te_eng_lda           | 0.7128 ± 0.0042 | 0.6342 ± 0.0069 |        36 |
| 2 Feature extraction        | SVM-RBF + te_eng_fis           | 0.7094 ± 0.0052 | 0.6296 ± 0.0069 |        30 |
| 2 Feature extraction        | kNN + te_eng_lda               | 0.7008 ± 0.0059 | 0.6291 ± 0.0085 |         2 |
| 2 Feature extraction        | kNN + te_eng_fis               | 0.6972 ± 0.0056 | 0.6282 ± 0.0087 |         2 |
| 2 Feature extraction        | SVM-RBF + te_eng_mi            | 0.7093 ± 0.0060 | 0.6271 ± 0.0085 |        41 |
| 2 Feature extraction        | kNN + te_eng_mi                | 0.6991 ± 0.0065 | 0.6267 ± 0.0093 |        12 |
| 2 Feature extraction        | DecisionTree + te_eng_lda      | 0.6978 ± 0.0038 | 0.6252 ± 0.0083 |         2 |
| 2 Feature extraction        | SVM-RBF + te_eng               | 0.7097 ± 0.0051 | 0.6212 ± 0.0085 |       294 |
| 2 Feature extraction        | SVM-RBF + te                   | 0.7086 ± 0.0042 | 0.6180 ± 0.0080 |       228 |
| 2 Feature extraction        | kNN + te                       | 0.6909 ± 0.0059 | 0.6145 ± 0.0105 |         2 |
| 2 Feature extraction        | DecisionTree + te_eng_mi       | 0.6885 ± 0.0050 | 0.6127 ± 0.0089 |        11 |
| 2 Feature extraction        | DecisionTree + te_eng_fis      | 0.6884 ± 0.0053 | 0.6126 ± 0.0095 |         1 |
| 2 Feature extraction        | kNN + te_eng                   | 0.6906 ± 0.0044 | 0.6121 ± 0.0086 |         9 |
| 2 Feature extraction        | SVM-RBF + te_eng_pca           | 0.7000 ± 0.0051 | 0.6103 ± 0.0093 |        53 |
| 2 Feature extraction        | DecisionTree + te              | 0.6902 ± 0.0060 | 0.6101 ± 0.0164 |         3 |
| 2 Feature extraction        | DecisionTree + te_eng          | 0.6883 ± 0.0086 | 0.6094 ± 0.0127 |         2 |
| 2 Feature extraction        | kNN + te_eng_pca               | 0.6879 ± 0.0052 | 0.6081 ± 0.0081 |         2 |
| 2 Feature extraction        | DecisionTree + te_eng_pca      | 0.6721 ± 0.0062 | 0.5878 ± 0.0120 |         2 |
| 2 Feature extraction        | kNN + eng                      | 0.6009 ± 0.0051 | 0.5157 ± 0.0090 |         2 |
| 2 Feature extraction        | NaiveBayes + te_eng_pca        | 0.5881 ± 0.0631 | 0.5025 ± 0.0454 |         1 |
| 2 Feature extraction        | SVM-RBF + eng                  | 0.6120 ± 0.0037 | 0.4880 ± 0.0077 |       121 |
| 2 Feature extraction        | NaiveBayes + te_eng_mi         | 0.4802 ± 0.0087 | 0.4758 ± 0.0110 |        11 |
| 2 Feature extraction        | NaiveBayes + te_eng_fis        | 0.4642 ± 0.0073 | 0.4588 ± 0.0097 |         1 |
| 2 Feature extraction        | NaiveBayes + te_eng            | 0.4713 ± 0.0584 | 0.4552 ± 0.0443 |         3 |
| 2 Feature extraction        | NaiveBayes + te                | 0.4729 ± 0.0582 | 0.4541 ± 0.0426 |         1 |
| 3 Modified methods          | CostSens-LDA                   | 0.7021 ± 0.0079 | 0.6552 ± 0.0087 |         2 |
| 3 Modified methods          | Soft-voting ensemble           | 0.7066 ± 0.0044 | 0.6536 ± 0.0073 |       444 |
| 3 Modified methods          | Ensemble + class-weight tuning | 0.7066 ± 0.0065 | 0.6531 ± 0.0102 |      1729 |
| 3 Modified methods          | CostSens-kNN                   | 0.6840 ± 0.0092 | 0.6390 ± 0.0088 |         3 |
| 3 Modified methods          | SVM-balanced                   | 0.6633 ± 0.0052 | 0.6328 ± 0.0055 |       103 |
| 3 Modified methods          | CostSens-QDA                   | 0.6392 ± 0.0349 | 0.5992 ± 0.0241 |         2 |
| 3 Modified methods          | Balanced kNN                   | 0.6248 ± 0.0084 | 0.5977 ± 0.0079 |         2 |
| 3 Modified methods          | Mahalanobis-kNN                | 0.6863 ± 0.0050 | 0.5974 ± 0.0097 |         2 |
| 3 Modified methods          | Mahalanobis+Balanced kNN       | 0.6234 ± 0.0084 | 0.5947 ± 0.0078 |         2 |
| 3 Modified methods          | CostSens-NaiveBayes            | 0.4727 ± 0.0577 | 0.4568 ± 0.0425 |         2 |
| 4 Deep learning (reference) | MLP (te_eng)                   | 0.7098 ± 0.0039 | 0.6348 ± 0.0110 |        45 |
| 4 Deep learning (reference) | MLP (raw)                      | 0.6171 ± 0.0072 | 0.5222 ± 0.0183 |        43 |

## Best per group: per-class precision / recall

| Group                       | Method                  | Class            |   Precision |   Recall |
|:----------------------------|:------------------------|:-----------------|------------:|---------:|
| 1 Baseline (raw)            | DecisionTree            | Grade 1 (low)    |      0.5379 |   0.3702 |
| 1 Baseline (raw)            | DecisionTree            | Grade 2 (medium) |      0.6734 |   0.8008 |
| 1 Baseline (raw)            | DecisionTree            | Grade 3 (high)   |      0.6616 |   0.5071 |
| 2 Feature extraction        | NaiveBayes + te_eng_lda | Grade 1 (low)    |      0.5623 |   0.5344 |
| 2 Feature extraction        | NaiveBayes + te_eng_lda | Grade 2 (medium) |      0.7243 |   0.8038 |
| 2 Feature extraction        | NaiveBayes + te_eng_lda | Grade 3 (high)   |      0.7244 |   0.5994 |
| 3 Modified methods          | CostSens-LDA            | Grade 1 (low)    |      0.5254 |   0.5892 |
| 3 Modified methods          | CostSens-LDA            | Grade 2 (medium) |      0.7368 |   0.7628 |
| 3 Modified methods          | CostSens-LDA            | Grade 3 (high)   |      0.7054 |   0.6316 |
| 4 Deep learning (reference) | MLP (te_eng)            | Grade 1 (low)    |      0.6092 |   0.408  |
| 4 Deep learning (reference) | MLP (te_eng)            | Grade 2 (medium) |      0.7124 |   0.832  |
| 4 Deep learning (reference) | MLP (te_eng)            | Grade 3 (high)   |      0.7299 |   0.5889 |

## Ablation

| Classifier   | Step           |   MacroF1 |    Acc |
|:-------------|:---------------|----------:|-------:|
| kNN          | raw            |    0.5158 | 0.6012 |
| kNN          | +eng           |    0.5157 | 0.6009 |
| kNN          | +target enc    |    0.6145 | 0.6909 |
| kNN          | +te+eng        |    0.6121 | 0.6906 |
| kNN          | +MI top-20     |    0.6267 | 0.6991 |
| kNN          | +Fisher top-20 |    0.6282 | 0.6972 |
| kNN          | +PCA 95%       |    0.6081 | 0.6879 |
| kNN          | +LDA proj      |    0.6291 | 0.7008 |
| SVM-RBF      | raw            |    0.4767 | 0.6116 |
| SVM-RBF      | +eng           |    0.488  | 0.612  |
| SVM-RBF      | +target enc    |    0.618  | 0.7086 |
| SVM-RBF      | +te+eng        |    0.6212 | 0.7097 |
| SVM-RBF      | +MI top-20     |    0.6271 | 0.7093 |
| SVM-RBF      | +Fisher top-20 |    0.6296 | 0.7094 |
| SVM-RBF      | +PCA 95%       |    0.6103 | 0.7    |
| SVM-RBF      | +LDA proj      |    0.6342 | 0.7128 |
| DecisionTree | raw            |    0.5808 | 0.661  |
| DecisionTree | +target enc    |    0.6101 | 0.6902 |
| DecisionTree | +te+eng        |    0.6094 | 0.6883 |
| DecisionTree | +MI top-20     |    0.6127 | 0.6885 |
| DecisionTree | +Fisher top-20 |    0.6126 | 0.6884 |
| DecisionTree | +PCA 95%       |    0.5878 | 0.6721 |
| DecisionTree | +LDA proj      |    0.6252 | 0.6978 |
| NaiveBayes   | raw            |    0.4029 | 0.4366 |
| NaiveBayes   | +target enc    |    0.4541 | 0.4729 |
| NaiveBayes   | +te+eng        |    0.4552 | 0.4713 |
| NaiveBayes   | +MI top-20     |    0.4758 | 0.4802 |
| NaiveBayes   | +Fisher top-20 |    0.4588 | 0.4642 |
| NaiveBayes   | +PCA 95%       |    0.5025 | 0.5881 |
| NaiveBayes   | +LDA proj      |    0.6552 | 0.7094 |

## Significance (paired t-test and Nadeau-Bengio corrected t-test)

| Comparison           | A                       | B                       | Metric   |   MeanDiff |   p_paired |   p_corrected |
|:---------------------|:------------------------|:------------------------|:---------|-----------:|-----------:|--------------:|
| Feature vs Baseline  | NaiveBayes + te_eng_lda | DecisionTree            | macro-F1 |     0.0745 |    3.9e-13 |       1.3e-08 |
| Feature vs Baseline  | NaiveBayes + te_eng_lda | DecisionTree            | accuracy |     0.0485 |    4.9e-11 |       1e-06   |
| Modified vs Baseline | CostSens-LDA            | DecisionTree            | macro-F1 |     0.0744 |    2.6e-14 |       1e-09   |
| Modified vs Baseline | CostSens-LDA            | DecisionTree            | accuracy |     0.0412 |    5e-09   |       4.8e-05 |
| Modified vs Feature  | CostSens-LDA            | NaiveBayes + te_eng_lda | macro-F1 |    -0      |    0.99    |       1       |
| Modified vs Feature  | CostSens-LDA            | NaiveBayes + te_eng_lda | accuracy |    -0.0073 |    1.6e-05 |       0.011   |
| Modified vs MLP      | CostSens-LDA            | MLP (te_eng)            | macro-F1 |     0.0204 |    3.8e-05 |       0.017   |
| Modified vs MLP      | CostSens-LDA            | MLP (te_eng)            | accuracy |    -0.0077 |    0.00011 |       0.029   |
| Feature vs MLP       | NaiveBayes + te_eng_lda | MLP (te_eng)            | macro-F1 |     0.0204 |    4.2e-06 |       0.005   |
| Feature vs MLP       | NaiveBayes + te_eng_lda | MLP (te_eng)            | accuracy |    -0.0004 |    0.77    |       0.89    |