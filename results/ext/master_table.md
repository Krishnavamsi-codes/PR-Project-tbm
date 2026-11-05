# Master results table (all methods, sorted by macro-F1)

| Group                       | Method                                     | Acc (mean±std)   | F1 (mean±std)   |   Rec_g1 |   Rec_g2 |   Rec_g3 |   Seconds |
|:----------------------------|:-------------------------------------------|:-----------------|:----------------|---------:|---------:|---------:|----------:|
| E Trees and ensembles       | Random forest, balanced                    | 0.7078 ± 0.0057  | 0.6579 ± 0.0073 |    0.544 |    0.777 |    0.637 |        87 |
| T Tuned (nested CV)         | Linear SVM (OvR) [tuned]                   | 0.7004 ± 0.0045  | 0.6568 ± 0.0059 |    0.631 |    0.746 |    0.643 |       596 |
| B Linear discriminative     | Linear SVM, balanced                       | 0.7004 ± 0.0043  | 0.6563 ± 0.0055 |    0.628 |    0.747 |    0.642 |        14 |
| 2 Feature extraction        | NaiveBayes + te_eng_lda                    | 0.7094 ± 0.0062  | 0.6552 ± 0.0079 |    0.534 |    0.804 |    0.599 |         1 |
| 3 Modified methods          | CostSens-LDA                               | 0.7021 ± 0.0079  | 0.6552 ± 0.0087 |    0.589 |    0.763 |    0.632 |         2 |
| 2 Feature extraction        | LDA + te_eng                               | 0.7075 ± 0.0061  | 0.6547 ± 0.0077 |    0.555 |    0.802 |    0.591 |         3 |
| A Generative                | LDA (shared covariance)                    | 0.7075 ± 0.0061  | 0.6547 ± 0.0077 |    0.555 |    0.802 |    0.591 |        21 |
| T Tuned (nested CV)         | LDA, Ledoit-Wolf shrinkage [tuned]         | 0.7050 ± 0.0060  | 0.6539 ± 0.0081 |    0.567 |    0.794 |    0.593 |         9 |
| 3 Modified methods          | Soft-voting ensemble                       | 0.7066 ± 0.0044  | 0.6536 ± 0.0073 |    0.509 |    0.77  |    0.656 |       444 |
| 3 Modified methods          | Ensemble + class-weight tuning             | 0.7066 ± 0.0065  | 0.6531 ± 0.0102 |    0.534 |    0.785 |    0.623 |      1729 |
| A Generative                | LDA, Ledoit-Wolf shrinkage                 | 0.7044 ± 0.0065  | 0.6527 ± 0.0087 |    0.563 |    0.796 |    0.589 |         4 |
| 2 Feature extraction        | LDA + te                                   | 0.7061 ± 0.0059  | 0.6506 ± 0.0079 |    0.537 |    0.803 |    0.59  |         3 |
| 2 Feature extraction        | LDA + te_eng_mi                            | 0.7032 ± 0.0058  | 0.6503 ± 0.0073 |    0.554 |    0.797 |    0.586 |        11 |
| 2 Feature extraction        | LDA + te_eng_fis                           | 0.7033 ± 0.0058  | 0.6499 ± 0.0076 |    0.55  |    0.799 |    0.585 |         1 |
| C Kernel methods            | Nystroem RBF + LDA (approx. kernel Fisher) | 0.7052 ± 0.0041  | 0.6485 ± 0.0053 |    0.533 |    0.802 |    0.59  |        33 |
| E Trees and ensembles       | LightGBM, balanced                         | 0.6785 ± 0.0060  | 0.6475 ± 0.0065 |    0.727 |    0.66  |    0.696 |         6 |
| E Trees and ensembles       | Stacking (LDA,kNN,RF,HGB -> logreg)        | 0.7160 ± 0.0045  | 0.6449 ± 0.0075 |    0.416 |    0.833 |    0.603 |       123 |
| T Tuned (nested CV)         | LightGBM [tuned]                           | 0.7147 ± 0.0055  | 0.6444 ± 0.0085 |    0.426 |    0.831 |    0.6   |       882 |
| E Trees and ensembles       | Gradient boosting (sklearn)                | 0.7146 ± 0.0050  | 0.6435 ± 0.0070 |    0.419 |    0.833 |    0.598 |       226 |
| T Tuned (nested CV)         | Hist gradient boosting [tuned]             | 0.7144 ± 0.0064  | 0.6433 ± 0.0102 |    0.419 |    0.832 |    0.599 |       976 |
| F Neural networks           | MLP 128-64 relu, L2 alpha=0.1              | 0.7114 ± 0.0041  | 0.6431 ± 0.0109 |    0.441 |    0.827 |    0.593 |        12 |
| C Kernel methods            | Nystroem poly-2 + logistic reg             | 0.7146 ± 0.0051  | 0.6415 ± 0.0069 |    0.412 |    0.836 |    0.595 |        23 |
| B Linear discriminative     | Logistic regression, balanced              | 0.6713 ± 0.0055  | 0.6413 ± 0.0056 |    0.77  |    0.637 |    0.702 |         2 |
| B Linear discriminative     | Logistic regression (softmax)              | 0.7149 ± 0.0061  | 0.6411 ± 0.0093 |    0.405 |    0.837 |    0.596 |         2 |
| F Neural networks           | MLP 128-64 sigmoid                         | 0.7122 ± 0.0051  | 0.6406 ± 0.0111 |    0.426 |    0.831 |    0.593 |        71 |
| E Trees and ensembles       | Extra-trees                                | 0.7143 ± 0.0041  | 0.6405 ± 0.0083 |    0.406 |    0.837 |    0.594 |        72 |
| E Trees and ensembles       | Hist gradient boosting                     | 0.7123 ± 0.0049  | 0.6398 ± 0.0102 |    0.414 |    0.828 |    0.602 |        20 |
| E Trees and ensembles       | Random forest                              | 0.7144 ± 0.0051  | 0.6395 ± 0.0081 |    0.398 |    0.837 |    0.597 |        83 |
| T Tuned (nested CV)         | Logistic regression (softmax) [tuned]      | 0.7014 ± 0.0204  | 0.6395 ± 0.0068 |    0.504 |    0.781 |    0.623 |       113 |
| E Trees and ensembles       | LightGBM                                   | 0.7115 ± 0.0047  | 0.6392 ± 0.0067 |    0.415 |    0.826 |    0.603 |        23 |
| 3 Modified methods          | CostSens-kNN                               | 0.6840 ± 0.0092  | 0.6390 ± 0.0088 |    0.584 |    0.736 |    0.625 |         3 |
| F Neural networks           | MLP 128-64 tanh                            | 0.7107 ± 0.0047  | 0.6389 ± 0.0112 |    0.426 |    0.831 |    0.589 |        62 |
| 2 Feature extraction        | LDA + te_eng_pca                           | 0.6931 ± 0.0063  | 0.6389 ± 0.0075 |    0.551 |    0.792 |    0.566 |         1 |
| F Neural networks           | MLP 1x64 relu                              | 0.7107 ± 0.0029  | 0.6383 ± 0.0094 |    0.414 |    0.828 |    0.597 |        25 |
| T Tuned (nested CV)         | MLP 128-64 relu [tuned]                    | 0.7098 ± 0.0042  | 0.6379 ± 0.0100 |    0.422 |    0.83  |    0.588 |       267 |
| C Kernel methods            | MI-20 + degree-2 poly features + logreg    | 0.7105 ± 0.0066  | 0.6376 ± 0.0081 |    0.418 |    0.832 |    0.588 |       129 |
| T Tuned (nested CV)         | Random forest [tuned]                      | 0.7136 ± 0.0047  | 0.6361 ± 0.0091 |    0.389 |    0.839 |    0.593 |      3237 |
| T Tuned (nested CV)         | Nystroem RBF + logistic reg [tuned]        | 0.7122 ± 0.0048  | 0.6351 ± 0.0068 |    0.393 |    0.839 |    0.589 |       369 |
| F Neural networks           | MLP 128-64 relu                            | 0.7098 ± 0.0039  | 0.6348 ± 0.0110 |    0.408 |    0.832 |    0.589 |        47 |
| 4 Deep learning (reference) | MLP (te_eng)                               | 0.7098 ± 0.0039  | 0.6348 ± 0.0110 |    0.408 |    0.832 |    0.589 |        45 |
| 2 Feature extraction        | SVM-RBF + te_eng_lda                       | 0.7128 ± 0.0042  | 0.6342 ± 0.0069 |    0.392 |    0.849 |    0.573 |        36 |
| C Kernel methods            | SVM RBF, balanced                          | 0.6633 ± 0.0052  | 0.6328 ± 0.0055 |    0.781 |    0.628 |    0.69  |       372 |
| 3 Modified methods          | SVM-balanced                               | 0.6633 ± 0.0052  | 0.6328 ± 0.0055 |    0.781 |    0.628 |    0.69  |       103 |
| F Neural networks           | MLP 256-128-64 relu (deep)                 | 0.7099 ± 0.0055  | 0.6320 ± 0.0184 |    0.391 |    0.826 |    0.605 |       107 |
| E Trees and ensembles       | Bagged trees                               | 0.7067 ± 0.0052  | 0.6319 ± 0.0096 |    0.402 |    0.829 |    0.587 |        81 |
| T Tuned (nested CV)         | Decision tree depth 10 [tuned]             | 0.7008 ± 0.0066  | 0.6304 ± 0.0100 |    0.433 |    0.82  |    0.575 |       115 |
| 2 Feature extraction        | SVM-RBF + te_eng_fis                       | 0.7094 ± 0.0052  | 0.6296 ± 0.0069 |    0.388 |    0.854 |    0.556 |        30 |
| E Trees and ensembles       | AdaBoost (depth-3 stumps)                  | 0.6967 ± 0.0054  | 0.6292 ± 0.0079 |    0.432 |    0.803 |    0.592 |        71 |
| 2 Feature extraction        | kNN + te_eng_lda                           | 0.7008 ± 0.0059  | 0.6291 ± 0.0085 |    0.42  |    0.819 |    0.581 |         2 |
| B Linear discriminative     | Linear SVM (OvR)                           | 0.7105 ± 0.0048  | 0.6290 ± 0.0066 |    0.374 |    0.839 |    0.588 |        13 |
| 2 Feature extraction        | kNN + te_eng_fis                           | 0.6972 ± 0.0056  | 0.6282 ± 0.0087 |    0.43  |    0.811 |    0.58  |         2 |
| 2 Feature extraction        | SVM-RBF + te_eng_mi                        | 0.7093 ± 0.0060  | 0.6271 ± 0.0085 |    0.376 |    0.854 |    0.559 |        41 |
| 2 Feature extraction        | kNN + te_eng_mi                            | 0.6991 ± 0.0065  | 0.6267 ± 0.0093 |    0.414 |    0.817 |    0.581 |        12 |
| T Tuned (nested CV)         | SVM RBF [tuned]                            | 0.7105 ± 0.0035  | 0.6267 ± 0.0064 |    0.37  |    0.852 |    0.568 |      4289 |
| C Kernel methods            | SVM polynomial deg 2                       | 0.7104 ± 0.0049  | 0.6263 ± 0.0075 |    0.372 |    0.854 |    0.563 |       293 |
| T Tuned (nested CV)         | SVM polynomial deg 2 [tuned]               | 0.7104 ± 0.0047  | 0.6263 ± 0.0062 |    0.371 |    0.855 |    0.563 |      2679 |
| 2 Feature extraction        | DecisionTree + te_eng_lda                  | 0.6978 ± 0.0038  | 0.6252 ± 0.0083 |    0.416 |    0.82  |    0.572 |         2 |
| T Tuned (nested CV)         | RDA / QDA reg=0.5 [tuned]                  | 0.6625 ± 0.0058  | 0.6248 ± 0.0074 |    0.635 |    0.656 |    0.681 |        78 |
| C Kernel methods            | SVM polynomial deg 3                       | 0.7076 ± 0.0038  | 0.6246 ± 0.0058 |    0.375 |    0.844 |    0.572 |       210 |
| B Linear discriminative     | Ridge classifier (least squares)           | 0.7088 ± 0.0042  | 0.6241 ± 0.0061 |    0.358 |    0.841 |    0.585 |         1 |
| B Linear discriminative     | SGD modified Huber                         | 0.7051 ± 0.0052  | 0.6220 ± 0.0242 |    0.387 |    0.831 |    0.583 |         3 |
| C Kernel methods            | SVM RBF                                    | 0.7097 ± 0.0051  | 0.6212 ± 0.0085 |    0.35  |    0.857 |    0.563 |       322 |
| 2 Feature extraction        | SVM-RBF + te_eng                           | 0.7097 ± 0.0051  | 0.6212 ± 0.0085 |    0.35  |    0.857 |    0.563 |       294 |
| B Linear discriminative     | SGD linear SVM (hinge)                     | 0.7090 ± 0.0056  | 0.6209 ± 0.0164 |    0.352 |    0.845 |    0.581 |         2 |
| D Instance based            | Nearest centroid (prototype)               | 0.6494 ± 0.0060  | 0.6197 ± 0.0072 |    0.695 |    0.602 |    0.716 |         3 |
| 2 Feature extraction        | SVM-RBF + te                               | 0.7086 ± 0.0042  | 0.6180 ± 0.0080 |    0.342 |    0.857 |    0.563 |       228 |
| C Kernel methods            | Nystroem RBF + logistic reg                | 0.7073 ± 0.0054  | 0.6172 ± 0.0072 |    0.339 |    0.845 |    0.579 |        10 |
| 2 Feature extraction        | kNN + te                                   | 0.6909 ± 0.0059  | 0.6145 ± 0.0105 |    0.398 |    0.818 |    0.559 |         2 |
| 2 Feature extraction        | DecisionTree + te_eng_mi                   | 0.6885 ± 0.0050  | 0.6127 ± 0.0089 |    0.403 |    0.81  |    0.565 |        11 |
| 2 Feature extraction        | DecisionTree + te_eng_fis                  | 0.6884 ± 0.0053  | 0.6126 ± 0.0095 |    0.403 |    0.81  |    0.564 |         1 |
| D Instance based            | kNN k=15                                   | 0.6906 ± 0.0044  | 0.6121 ± 0.0086 |    0.387 |    0.826 |    0.548 |         9 |
| 2 Feature extraction        | kNN + te_eng                               | 0.6906 ± 0.0044  | 0.6121 ± 0.0086 |    0.387 |    0.826 |    0.548 |         9 |
| T Tuned (nested CV)         | kNN k=15 [tuned]                           | 0.6912 ± 0.0071  | 0.6121 ± 0.0101 |    0.382 |    0.824 |    0.554 |       230 |
| 2 Feature extraction        | SVM-RBF + te_eng_pca                       | 0.7000 ± 0.0051  | 0.6103 ± 0.0093 |    0.339 |    0.847 |    0.555 |        53 |
| 2 Feature extraction        | DecisionTree + te                          | 0.6902 ± 0.0060  | 0.6101 ± 0.0164 |    0.387 |    0.817 |    0.562 |         3 |
| E Trees and ensembles       | Decision tree depth 10                     | 0.6883 ± 0.0086  | 0.6094 ± 0.0127 |    0.388 |    0.81  |    0.567 |         4 |
| 2 Feature extraction        | DecisionTree + te_eng                      | 0.6883 ± 0.0086  | 0.6094 ± 0.0127 |    0.388 |    0.81  |    0.567 |         2 |
| 2 Feature extraction        | kNN + te_eng_pca                           | 0.6879 ± 0.0052  | 0.6081 ± 0.0081 |    0.379 |    0.824 |    0.545 |         2 |
| D Instance based            | kNN k=45 distance-weighted                 | 0.6983 ± 0.0045  | 0.6039 ± 0.0090 |    0.323 |    0.854 |    0.543 |         9 |
| D Instance based            | kNN k=5                                    | 0.6678 ± 0.0063  | 0.6009 ± 0.0093 |    0.433 |    0.772 |    0.558 |         9 |
| 3 Modified methods          | CostSens-QDA                               | 0.6392 ± 0.0349  | 0.5992 ± 0.0241 |    0.544 |    0.617 |    0.705 |         2 |
| 3 Modified methods          | Balanced kNN                               | 0.6248 ± 0.0084  | 0.5977 ± 0.0079 |    0.794 |    0.561 |    0.685 |         2 |
| 3 Modified methods          | Mahalanobis-kNN                            | 0.6863 ± 0.0050  | 0.5974 ± 0.0097 |    0.347 |    0.84  |    0.523 |         2 |
| 3 Modified methods          | Mahalanobis+Balanced kNN                   | 0.6234 ± 0.0084  | 0.5947 ± 0.0078 |    0.762 |    0.571 |    0.672 |         2 |
| T Tuned (nested CV)         | Parzen window / KDE-Bayes (PCA-8) [tuned]  | 0.6810 ± 0.0067  | 0.5946 ± 0.0113 |    0.352 |    0.822 |    0.537 |       415 |
| A Generative                | Parzen window / KDE-Bayes (PCA-8)          | 0.6844 ± 0.0054  | 0.5899 ± 0.0083 |    0.334 |    0.856 |    0.494 |        11 |
| 2 Feature extraction        | DecisionTree + te_eng_pca                  | 0.6721 ± 0.0062  | 0.5878 ± 0.0120 |    0.349 |    0.796 |    0.554 |         2 |
| 1 Baseline (raw)            | DecisionTree                               | 0.6610 ± 0.0064  | 0.5808 ± 0.0091 |    0.37  |    0.801 |    0.507 |         1 |
| A Generative                | GMM-Bayes, 3 diag comp/class (PCA-12)      | 0.6253 ± 0.0146  | 0.5799 ± 0.0137 |    0.559 |    0.668 |    0.572 |         1 |
| T Tuned (nested CV)         | GMM-Bayes, 3 comp/class (PCA-12) [tuned]   | 0.5943 ± 0.0452  | 0.5657 ± 0.0339 |    0.553 |    0.539 |    0.7   |        99 |
| B Linear discriminative     | Perceptron                                 | 0.6339 ± 0.0349  | 0.5643 ± 0.0311 |    0.429 |    0.698 |    0.585 |         1 |
| A Generative                | RDA / QDA reg=0.5                          | 0.5852 ± 0.0402  | 0.5588 ± 0.0305 |    0.521 |    0.476 |    0.788 |         7 |
| A Generative                | GMM-Bayes, 3 comp/class (PCA-12)           | 0.5714 ± 0.0428  | 0.5454 ± 0.0303 |    0.611 |    0.513 |    0.66  |         2 |
| C Kernel methods            | SVM sigmoid (tanh) kernel                  | 0.6130 ± 0.0075  | 0.5448 ± 0.0105 |    0.417 |    0.68  |    0.556 |       201 |
| 4 Deep learning (reference) | MLP (raw)                                  | 0.6171 ± 0.0072  | 0.5222 ± 0.0183 |    0.341 |    0.822 |    0.349 |        43 |
| 1 Baseline (raw)            | kNN                                        | 0.6012 ± 0.0059  | 0.5158 ± 0.0090 |    0.328 |    0.774 |    0.387 |         8 |
| 2 Feature extraction        | kNN + eng                                  | 0.6009 ± 0.0051  | 0.5157 ± 0.0090 |    0.337 |    0.785 |    0.364 |         2 |
| 2 Feature extraction        | NaiveBayes + te_eng_pca                    | 0.5881 ± 0.0631  | 0.5025 ± 0.0454 |    0.286 |    0.613 |    0.633 |         1 |
| T Tuned (nested CV)         | Gaussian Naive Bayes [tuned]               | 0.5101 ± 0.0212  | 0.5009 ± 0.0179 |    0.539 |    0.298 |    0.861 |        32 |
| 1 Baseline (raw)            | LDA                                        | 0.5860 ± 0.0056  | 0.4955 ± 0.0062 |    0.512 |    0.831 |    0.192 |         1 |
| 2 Feature extraction        | SVM-RBF + eng                              | 0.6120 ± 0.0037  | 0.4880 ± 0.0077 |    0.29  |    0.88  |    0.25  |       121 |
| A Generative                | QDA reg=0.1                                | 0.4991 ± 0.0304  | 0.4827 ± 0.0262 |    0.46  |    0.293 |    0.86  |         6 |
| A Generative                | GMM-Bayes, 1 comp/class (PCA-12)           | 0.4992 ± 0.0106  | 0.4822 ± 0.0107 |    0.496 |    0.286 |    0.863 |         3 |
| 1 Baseline (raw)            | SVM-RBF                                    | 0.6116 ± 0.0043  | 0.4767 ± 0.0089 |    0.247 |    0.88  |    0.26  |        98 |
| 2 Feature extraction        | NaiveBayes + te_eng_mi                     | 0.4802 ± 0.0087  | 0.4758 ± 0.0110 |    0.659 |    0.206 |    0.896 |        11 |
| 2 Feature extraction        | NaiveBayes + te_eng_fis                    | 0.4642 ± 0.0073  | 0.4588 ± 0.0097 |    0.644 |    0.17  |    0.912 |         1 |
| 3 Modified methods          | CostSens-NaiveBayes                        | 0.4727 ± 0.0577  | 0.4568 ± 0.0425 |    0.585 |    0.226 |    0.86  |         2 |
| 2 Feature extraction        | NaiveBayes + te_eng                        | 0.4713 ± 0.0584  | 0.4552 ± 0.0443 |    0.586 |    0.221 |    0.863 |         3 |
| A Generative                | Gaussian Naive Bayes                       | 0.4713 ± 0.0584  | 0.4552 ± 0.0443 |    0.586 |    0.221 |    0.863 |         2 |
| 2 Feature extraction        | NaiveBayes + te                            | 0.4729 ± 0.0582  | 0.4541 ± 0.0426 |    0.575 |    0.228 |    0.86  |         1 |
| 1 Baseline (raw)            | QDA                                        | 0.4759 ± 0.0158  | 0.4518 ± 0.0193 |    0.388 |    0.296 |    0.807 |         1 |
| 1 Baseline (raw)            | NaiveBayes                                 | 0.4366 ± 0.0394  | 0.4029 ± 0.0194 |    0.501 |    0.189 |    0.84  |         1 |


# Best method per group

| Group                       | Method                                     | Acc (mean±std)   | F1 (mean±std)   |   Rec_g1 |   Rec_g2 |   Rec_g3 |   Seconds |
|:----------------------------|:-------------------------------------------|:-----------------|:----------------|---------:|---------:|---------:|----------:|
| 1 Baseline (raw)            | DecisionTree                               | 0.6610 ± 0.0064  | 0.5808 ± 0.0091 |    0.37  |    0.801 |    0.507 |         1 |
| 2 Feature extraction        | NaiveBayes + te_eng_lda                    | 0.7094 ± 0.0062  | 0.6552 ± 0.0079 |    0.534 |    0.804 |    0.599 |         1 |
| 3 Modified methods          | CostSens-LDA                               | 0.7021 ± 0.0079  | 0.6552 ± 0.0087 |    0.589 |    0.763 |    0.632 |         2 |
| 4 Deep learning (reference) | MLP (te_eng)                               | 0.7098 ± 0.0039  | 0.6348 ± 0.0110 |    0.408 |    0.832 |    0.589 |        45 |
| A Generative                | LDA (shared covariance)                    | 0.7075 ± 0.0061  | 0.6547 ± 0.0077 |    0.555 |    0.802 |    0.591 |        21 |
| B Linear discriminative     | Linear SVM, balanced                       | 0.7004 ± 0.0043  | 0.6563 ± 0.0055 |    0.628 |    0.747 |    0.642 |        14 |
| C Kernel methods            | Nystroem RBF + LDA (approx. kernel Fisher) | 0.7052 ± 0.0041  | 0.6485 ± 0.0053 |    0.533 |    0.802 |    0.59  |        33 |
| D Instance based            | Nearest centroid (prototype)               | 0.6494 ± 0.0060  | 0.6197 ± 0.0072 |    0.695 |    0.602 |    0.716 |         3 |
| E Trees and ensembles       | Random forest, balanced                    | 0.7078 ± 0.0057  | 0.6579 ± 0.0073 |    0.544 |    0.777 |    0.637 |        87 |
| F Neural networks           | MLP 128-64 relu, L2 alpha=0.1              | 0.7114 ± 0.0041  | 0.6431 ± 0.0109 |    0.441 |    0.827 |    0.593 |        12 |
| T Tuned (nested CV)         | Linear SVM (OvR) [tuned]                   | 0.7004 ± 0.0045  | 0.6568 ± 0.0059 |    0.631 |    0.746 |    0.643 |       596 |


# Tuned (nested CV) vs untuned, macro-F1

| Model                             |   Untuned_F1 |   Tuned_F1 |    Gain |   p_corrected |   Untuned_Acc |   Tuned_Acc |
|:----------------------------------|-------------:|-----------:|--------:|--------------:|--------------:|------------:|
| RDA / QDA reg=0.5                 |       0.5588 |     0.6248 |  0.066  |        0.0021 |        0.5852 |      0.6625 |
| Gaussian Naive Bayes              |       0.4552 |     0.5009 |  0.0457 |        0.0221 |        0.4713 |      0.5101 |
| Linear SVM (OvR)                  |       0.629  |     0.6568 |  0.0278 |        0      |        0.7105 |      0.7004 |
| Decision tree depth 10            |       0.6094 |     0.6304 |  0.021  |        0.0034 |        0.6883 |      0.7008 |
| GMM-Bayes, 3 comp/class (PCA-12)  |       0.5454 |     0.5657 |  0.0203 |        0.4934 |        0.5714 |      0.5943 |
| Nystroem RBF + logistic reg       |       0.6172 |     0.6351 |  0.0179 |        0.0021 |        0.7073 |      0.7122 |
| SVM RBF                           |       0.6212 |     0.6267 |  0.0055 |        0.0518 |        0.7097 |      0.7105 |
| LightGBM                          |       0.6392 |     0.6444 |  0.0053 |        0.1698 |        0.7115 |      0.7147 |
| Parzen window / KDE-Bayes (PCA-8) |       0.5899 |     0.5946 |  0.0047 |        0.2464 |        0.6844 |      0.681  |
| Hist gradient boosting            |       0.6398 |     0.6433 |  0.0035 |        0.2517 |        0.7123 |      0.7144 |
| MLP 128-64 relu                   |       0.6348 |     0.6379 |  0.003  |        0.6519 |        0.7098 |      0.7098 |
| LDA, Ledoit-Wolf shrinkage        |       0.6527 |     0.6539 |  0.0012 |        0.6088 |        0.7044 |      0.705  |
| SVM polynomial deg 2              |       0.6263 |     0.6263 | -0      |        0.9836 |        0.7104 |      0.7104 |
| kNN k=15                          |       0.6121 |     0.6121 | -0.0001 |        0.9842 |        0.6906 |      0.6912 |
| Logistic regression (softmax)     |       0.6411 |     0.6395 | -0.0017 |        0.6953 |        0.7149 |      0.7014 |
| Random forest                     |       0.6395 |     0.6361 | -0.0034 |        0.3962 |        0.7144 |      0.7136 |


# What the inner searches picked (most frequent winner over outer folds)

| Model          |   Searches |   Inner_F1 | Most_chosen                                                                            | Times   |   Secs_per_search |
|:---------------|-----------:|-----------:|:---------------------------------------------------------------------------------------|:--------|------------------:|
| GMM_3          |         15 |     0.5689 | {"clf__n_components": 4, "clf__reg_covar": 0.1}                                        | 9/15    |                94 |
| HistGB         |         15 |     0.6356 | {"clf__l2_regularization": 1.0, "clf__learning_rate": 0.03, "clf__max_leaf_nodes": 15} | 4/15    |               929 |
| LDA_shrink     |         15 |     0.6483 | {"clf__shrinkage": 0.05}                                                               | 8/15    |                 8 |
| LightGBM       |         15 |     0.636  | {"clf__learning_rate": 0.03, "clf__min_child_samples": 20, "clf__num_leaves": 15}      | 8/15    |               875 |
| LinSVC         |         22 |     0.6512 | {"clf__C": 0.01, "clf__class_weight": "balanced"}                                      | 10/22   |               526 |
| LogReg         |         15 |     0.6337 | {"clf__C": 0.3, "clf__class_weight": null}                                             | 7/15    |               110 |
| MLP_128_64     |         15 |     0.6316 | {"clf__alpha": 0.01, "clf__hidden_layer_sizes": "(128, 64)"}                           | 4/15    |               256 |
| NB             |         15 |     0.4862 | {"clf__var_smoothing": 0.1}                                                            | 15/15   |                30 |
| Nys_RBF_LogReg |         15 |     0.6254 | {"clf__C": 1.0, "nys__gamma": 0.005}                                                   | 15/15   |               209 |
| Parzen         |         15 |     0.5868 | {"clf__bandwidth": 0.4}                                                                | 14/15   |               378 |
| QDA_r05        |         15 |     0.6138 | {"clf__reg_param": 0.8}                                                                | 15/15   |                60 |
| RF             |         15 |     0.6325 | {"clf__max_depth": null, "clf__max_features": "sqrt", "clf__min_samples_leaf": 3}      | 3/15    |              3193 |
| SVM_poly2      |         15 |     0.6215 | {"clf__C": 1, "clf__degree": 2}                                                        | 9/15    |              1626 |
| SVM_rbf        |         15 |     0.6214 | {"clf__C": 10, "clf__gamma": 0.003}                                                    | 11/15   |              2709 |
| Tree           |         15 |     0.6284 | {"clf__max_depth": 4, "clf__min_samples_leaf": 50}                                     | 7/15    |               114 |
| kNN15          |         15 |     0.6014 | {"clf__n_neighbors": 11, "clf__weights": "uniform"}                                    | 6/15    |               212 |