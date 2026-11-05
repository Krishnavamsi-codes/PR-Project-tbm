# Presentation outline (about 10 min, ~3 min per person)

Numbers match `Report_Earthquake_Damage_PR.docx`. MF1 = macro-F1; 20k = 20,000-row subsample (15 runs); full = all 260,601 rows (5 runs). Model families: decision tree, random forest, SVM, linear models, MLP (reference only).

1. **Problem and data (member 1, 1 min).** Gorkha 2015 damage grading, 260,601 buildings, 3 imbalanced grades (9.6 / 56.9 / 33.5 %); why macro-F1 and not accuracy. Geography matters: grade-3 share per region 8.6 % to 80.8 %.
2. **Protocol (member 1, 1 min).** 20k stratified subsample, 5-fold x 3 repeats, every learned step fitted inside the training fold, corrected significance tests. Best raw baseline: decision tree, MF1 0.581.
3. **Feature extraction (member 2, 3 min).** Out-of-fold hierarchical target encoding of the geo ids is the key step (SVM +0.141, MLP +0.113, tree +0.029 MF1). PCA hurts (SVM -0.011); MI / Fisher selection gains are within noise. Figure: ablation plot.
4. **Class balancing, models, tuning (member 3, 3 min).** Balanced random forest is best on 20k (MF1 0.658), tied with the balanced linear SVM (0.656, 14 s versus 87 s). Balancing raises grade-1 recall (SVM 0.35 to 0.78) and helps forest (+0.018) and linear SVM (+0.027), not logistic regression. Nested tuning helps only poor defaults (linear SVM +0.028, tree +0.021). MLP is a reference only.
5. **Full-data confirmation (member 3, 30 s).** On all rows: balanced RF 0.707 MF1 / 0.742 accuracy, significantly ahead of the balanced linear SVM and MLP (both 0.686) and logistic regression (0.674); flexible models gain most from more data.
6. **Conclusion (all, 1 min).** Features first, then the decision rule for macro-F1. Small data: balanced linear SVM (0.656 in seconds). Full data: balanced random forest. Limits: 20k subsample, fixed hyper-parameters, one earthquake. Future work: exact kernel SVMs on all rows, ordinal losses, spatial features.
7. **Demo.** Show `python code/main.py` output and the repo link: https://github.com/Krishnavamsi-codes/PR-Project-tbm
