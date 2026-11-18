# Final Model Report: High-FLOPs Custom CNN

## 1. Summary
- Dataset: EuroSAT_RGB (10 classes)
- Final model: Baseline (outperformed the best HPT configuration on validation)
- Evaluated once on the test set: 4050 images

| Metric | Value |
|---|---|
| Accuracy | 0.9620 |
| Macro F1 | 0.9620 |
| Weighted F1 | 0.9619 |
| Loss | 0.1060 |
| Misclassified | 154 / 4050 |

## 2. Per-Class Results
| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| AnnualCrop | 0.9789 | 0.9267 | 0.9521 | 450 |
| Forest | 0.9375 | 1.0000 | 0.9677 | 450 |
| HerbaceousVegetation | 0.9289 | 0.9578 | 0.9431 | 450 |
| Highway | 0.9684 | 0.9813 | 0.9748 | 375 |
| Industrial | 0.9758 | 0.9680 | 0.9719 | 375 |
| Pasture | 0.9603 | 0.9667 | 0.9635 | 300 |
| PermanentCrop | 0.9457 | 0.9280 | 0.9367 | 375 |
| Residential | 0.9612 | 0.9911 | 0.9759 | 450 |
| River | 0.9730 | 0.9600 | 0.9664 | 375 |
| SeaLake | 0.9976 | 0.9400 | 0.9680 | 450 |

- Strongest (F1): Residential (0.9759), Highway (0.9748), Industrial (0.9719)
- Weakest (F1): PermanentCrop (0.9367), HerbaceousVegetation (0.9431), AnnualCrop (0.9521)

## 3. Main Confusions (true -> predicted)
| Pair | Count |
|---|---|
| PermanentCrop -> HerbaceousVegetation | 17 |
| SeaLake -> Forest | 15 |
| Industrial -> Residential | 10 |
| AnnualCrop -> PermanentCrop | 9 |
| HerbaceousVegetation -> PermanentCrop | 8 |

Note: each pair has a small count, so these are descriptive patterns, not statistically tested claims. The full numeric matrix is in the Confusion matrix (CSV/JSON) files listed below.

## 4. Model Complexity
| Item | Value |
|---|---|
| Total parameters | 3,544,266 |
| FLOPs / image | 3,945,769,404 |
| Checkpoint size (MB) | 67.8107 |
| Device | Tesla T4 |
| Latency batch=1 (ms) | 18.9422 (high variance, mostly launch overhead) |
| Throughput batch=32 (img/s) | 1197.23 |

FLOPs are measured with the TensorFlow profiler (about 2 x MACs).

## 5. Output Files
| Item | Path |
|---|---|
| Final test metrics (JSON) | `outputs/high/high_final_evaluation/high_final-evaluation_final_test_test_metrics.json` |
| Classification report | `outputs/high/high_final_evaluation/high_final-evaluation_final_test_test_classification_report.txt` |
| Confusion matrix (plot) | `outputs/high/high_final_evaluation/high_final-evaluation_final_test_test_confusion_matrix.png` |
| Confusion matrix (CSV) | `outputs/high/high_final_evaluation/high_final_evaluation_confusion_matrix.csv` |
| Confusion matrix (JSON) | `outputs/high/high_final_evaluation/high_final_evaluation_confusion_matrix.json` |
| Model complexity (JSON) | `outputs/high/high_final_evaluation/high_final_evaluation_model_complexity.json` |
| Handoff package | `outputs/high/high_final_evaluation/handoff` |

## 6. Provenance
- Checkpoint SHA256: `b6be1306c4018473c7ade9c9ee50224f258af11531a7f9b9bed9cc4afa70236e`
- W&B run: high_final_evaluation (bj1q54kc)
- Generated (UTC): 2026-11-18T10:58:21+00:00
