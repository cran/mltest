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

# mltest 1.0.2

* \strong{} removed from within the the \item{} of the return in the #' @return
  part of the roxygen2 header
  - thanks to Dr. Hornik for the advice
* \bugfix: `TN` is now caclulated correctly (#107):
  TN <- sapply(1:length(TP), function(y) {sum(confusion_mtx[-y, -y], na.rm = TRUE)})
  - thanks to Dr. Stewart for the advice
* ORCID number is added in the description file

# mltest 1.0.3

* added CRAN-assigned DOI to the inst/CITATION file
* removed balanced.accuracy mentioned twice in the #' @description
* removed 2nd parenthesis pair around `((\strong{FOR}))` in the .R file
* replaced `\emph{calculated as:}` with `=` in the #' @return of the .R file
* removed redundant parentheses in \item{F0.5}, \item{1} and \item{F2}
