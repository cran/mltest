# mltest 1.0.1

* added bibliography in the help page of the function "mltest::ml_test()"
* removed "mltest.R" file which previously described the package
* redone/simplified the implementation of the calculation of local variables "TP" and "TN"   
  within "mltest::ml_test()"
* added variables "FP" and "FN" (semantiaclly corresponding to FP and FN of a confusion 
  matrix) within "mltest::ml_test()"
* reimplemented/expressed evaluation-metrics "precision", "recall", "specificity" in simple  
  terms of TP, TN, FP and FN
* implemented new evaluation-metrics: "DOR", "error.rate", "FDR"/"FNR"/"FOR"/"FPR", 
  "geometric.mean", "Jaccard",
  "L"/"lambda" (representing LR(+) and LR(-) respectively), "MCC", "MK" (Markedness), "NPV", 
  "OP" (optimization precision),
  "Youden"('s index)
* added an extra argument "output.as.table" which is FALSE by default: this returns a list 
  with named vectors (class-mtrics)
  for fast access through referencing/subsetting a la 
  <all-mtrics-list>$<particular-metric-type>. when the above
  "output.as.table" = TRUE the dataframe with all the metrics is returned (except for 
  "accuracy" and "error.rate")
