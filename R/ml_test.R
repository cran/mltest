#' @title multi-class classifier evaluation metrics based on a confusion matrix (contingency table)
#'
#' @description Calculates multi-class classification evaluation metrics: \strong{balanced.accuracy}, balanced accuracy (\strong{balanced.accuracy}),
#' diagnostic odds ratio (\strong{DOR}), error rate (\strong{error.rate}), F.beta (\strong{F0.5}, \strong{F1} (F-measure, F-score), \strong{F2} with where beta is 0.5, 1 and 2 respectively),
#' false positive rate (\strong{FPR}), false negative rate (\strong{FNR}), false omission rate ((\strong{FOR})), false discovery rate (\strong{FDR}),
#' geometric mean (\strong{geometric.mean}), \strong{Jaccard}, positive likelihood ratio (p+, LR(+) or simply \strong{L}),
#' negative likelihood ratio (p-, LR(-) or simply \strong{lambda}),  Matthews corellation coefficient (\strong{MCC}), markedness (\strong{MK}), negative predictive value (\strong{NPV}),
#' optimization precision \strong{OP}, \strong{precision}, \strong{recall} (sensitivity), \strong{specificity} and finally \strong{Youden}'s index.
#' The function calculates the aforementioned metrics from a confusion matrix (contingency matrix)
#' where \emph{TP}, \emph{TN}, \emph{FP} \emph{FN} are abbreviations for \emph{true positives}, \emph{true negatives},
#' \emph{false positives} and \emph{false negatives} respectively.
#'
#' @param predicted class labels predicted by the classifier model (a set of classes convertible into type factor with levels representing labels)
#'
#' @param true true class labels (a set of classes convertible into type factor of the same length and with the same levels as predicted)
#'
#' @param output.as.table the function returns all metrics except for \strong{accuracy} and \strong{error.rate} in a tabular format if this argument is set to \emph{TRUE}
#'
#' @author G. Dudnik
#'
#' @return the function returns a list of following metrics:
#'   \item{ \strong{accuracy} }{ \emph{calculated as:} (TP+TN) / (TP+FP+TN+FN) \emph{(doesn't show up when output.as.table = TRUE)}}
#'   \item{ \strong{balanced.accuracy} }{ \emph{calculated as:} (TP / (TP+FN)+TN / (TN+FP)) / 2 = (recall+specificity) / 2}
#'   \item{ \strong{DOR} }{ \emph{calculated as:}  TP*TN / (FP*FN) = L / lambda}
#'   \item{ \strong{error.rate} }{ \emph{calculated as:}  (FP+FN) / (TP+TN+FP+FN) = 1-accuracy \emph{(doesn't show up when output.as.table = TRUE)}}
#'   \item{ \strong{F0.5} }{ \emph{calculated as:}  1.25*(recall*precision/(0.25*precision+recall))}
#'   \item{ \strong{F1} }{ \emph{calculated as:}  2*(precision*recall / (precision+recall))}
#'   \item{ \strong{F2} }{ \emph{calculated as:}  5*(precision*recall / (4*precision+recall))}
#'   \item{ \strong{FDR} }{ \emph{calculated as:}  1-precision}
#'   \item{ \strong{FNR} }{ \emph{calculated as:}  1-recall}
#'   \item{ \strong{FOR} }{ \emph{calculated as:}  1-NPV}
#'   \item{ \strong{FPR} }{ \emph{calculated as:}  1-specificity}
#'   \item{ \strong{geometric.mean} }{ \emph{calculated as:}  (recall*specificity)^0.5}
#'   \item{ \strong{Jaccard} }{ \emph{calculated as:}  TP / (TP+FP+FN)}
#'   \item{ \strong{L} }{ \emph{calculated as:}  recall / (1-specificity)}
#'   \item{ \strong{lambda} }{ \emph{calculated as:}  (1-recall) / (specificity)}
#'   \item{ \strong{MCC} }{ \emph{calculated as:}  (TP*TN-FP*FN) / (((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^0.5)}
#'   \item{ \strong{MK} }{ \emph{calculated as:}  precision + NPV - 1}
#'   \item{ \strong{NPV} }{ \emph{calculated as:}  TN / (TN+FN)}
#'   \item{ \strong{OP} }{ \emph{calculated as:}  accuracy - |recall-specificity| / (recall+specificity)}
#'   \item{ \strong{precision} }{ \emph{calculated as:} TP / (TP+FP)}
#'   \item{ \strong{recall} }{ \emph{calculated as:}  TP / (TP+FN)}
#'   \item{ \strong{specificity} }{ \emph{calculated as:}  TN / (TN+FP)}
#'   \item{ \strong{Youden} }{ \emph{calculated as:}  recall+specificity-1}
#'
#' @keywords utilities
#'
#' @references
#' \enumerate{
#'   \item Sasaki Y. (2007). The truth of the F-measure.:1–5. \url{https://www.researchgate.net/publication/268185911_The_truth_of_the_F-measure}.
#'   \item Powers DMW. (2011). Evaluation: from Precision, Recall and F-measure to ROC, Informedness, Markedness & Correlation. Arch Geschwulstforsch. 2(1):37–63. \url{https://www.researchgate.net/publication/313610493_Evaluation_From_precision_recall_and_fmeasure_to_roc_informedness_markedness_and_correlation}.
#'   \item Bekkar M, Djemaa HK, Alitouche TA. (2013). Evaluation Measures for Models Assessment over Imbalanced Data Sets. J Inf Eng Appl. 3(10):27–38. \url{https://www.iiste.org/Journals/index.php/JIEA/article/view/7633}.
#'   \item Jeni LA, Cohn JF, De La Torre F. (2013). Facing Imbalanced Data Recommendations for the Use of Performance Metrics. Conference on Affective Computing and Intelligent Interaction. IEEE. p. 245–51. \url{http://ieeexplore.ieee.org/document/6681438/}.
#'   \item López V, Fernández A, García S, Palade V, Herrera F. (2013). An insight into classification with imbalanced data: Empirical results and current trends on using data intrinsic characteristics. Inf Sci. 250:113–41. \url{http://dx.doi.org/10.1016/j.ins.2013.07.007}.
#'   \item Tharwat A. (2018). Classification assessment methods. Appl Comput Informatics . \url{https://linkinghub.elsevier.com/retrieve/pii/S2210832718301546}.
#' }
#'
#' @examples
#' library(mltest)
#'
#' # class labels ("cat, "dog" and "rat") predicted by the classifier model
#' predicted_labels <- as.factor(c("dog", "cat", "dog", "rat", "rat"))
#'
#' # true labels (test set)
#' true_labels <- as.factor(c("dog", "cat", "dog", "rat", "dog"))
#'
#' classifier_metrics <- ml_test(predicted_labels, true_labels, output.as.table = FALSE)
#'
#' # overall classification accuracy
#' accuracy <- classifier_metrics$accuracy
#'
#' # F1-measures for classes "cat", "dog" and "rat"
#' F1 <- classifier_metrics$F1
#'
#' # tabular view of the metrics (except for 'accuracy' and 'error.rate')
#' classifier_metrics <- ml_test(predicted_labels, true_labels, output.as.table = TRUE)
#'
#' @encoding UTF-8
#'
#' @export
ml_test <- function(predicted, true, output.as.table = FALSE){
  # testing for correct input
  predicted <- as.factor(predicted)
  true <- as.factor(true)
  if(!is.factor(predicted) | !is.factor(true)) stop(
    "Please make sure that both 'predicted' and 'true' could be converted
    into input of type 'factor' and retry."
  )
  if(length(predicted) != length(true)) stop(
    "Please make sure that both 'predicted' and 'true' are of the same length and retry."
  )
  if(!identical(levels(predicted), levels(true))) stop(
    "Please make sure that both 'predicted' and 'true' have identical factor-levels representing classes and retry."
  )
  if(!is.logical(output.as.table)) stop(
    "Please make sure that 'output.as.table' is either TRUE or FALSE and retry."
  )

  # creating a confusion matrix
  confusion_mtx <- table(predicted, true) # true classes are column-names
  rm(predicted, true)
  class_label_names <- rownames(confusion_mtx)

  # 'TP', 'FP' and 'FN' - 'True Positives', 'False Positives' and 'False Negatives'
  TP <- diag(confusion_mtx); FP <- rowSums(confusion_mtx)-TP; FN <- colSums(confusion_mtx)-TP
  # 'TN' - 'True Negatives'
  TN <- sapply(1:length(TP), function(y, TP){ sum(TP[-y], na.rm = TRUE) }, TP)

  # 'accuracy' = (TP+TN)/(TP+FP+TN+FN)
  accuracy <- sum(TP)/sum(confusion_mtx, na.rm = TRUE)
  rm(confusion_mtx)
  # 'recall' ('sensitivity', 'TPR') = TP/RP = TP/(TP+FN)
  precision <- TP/(TP+FP)
  # 'specificity' = TN/RN = TN/(TN+FP)
  recall <- TP/(TP+FN)
  # 'precision' ('PPV') = TP/PP = TP/(TP+FP)
  specificity <- TN/(TN+FP)
  names(specificity) <- class_label_names

  # 'Jaccard' = TP/(TP+FP+FN)
  Jaccard <- TP/(TP+FP+FN)

  # 'NPV' = TN/(TN+FN)
  NPV <- TN/(TN+FN)

  # 'MCC'
  MCC <- (TP*TN-FP*FN)/sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
  rm(TP, TN, FP, FN)

  # 'balanced accuracy'
  balanced.accuracy <- (recall+specificity)/2
  # 'error.rate' = (FP+FN)/(TP+TN+FP+FN) = 1-accuracy
  error.rate <- 1-accuracy

  # 'F-scores' ('F.beta', where where beta is equal to 0.5, 1 or 2)
  F0.5 <- 1.25*(recall*precision/(0.25*precision+recall))
  F1 <- 2*(precision*recall/(precision+recall)) # Dice
  F2 <- 5*(precision*recall/(4*precision+recall))

  # 'FDR' = 1 - precision
  FDR <- 1-precision
  # 'FNR' = FN/(FN+TP) = 1 - recall
  FNR <- 1-recall
  # 'FOR' = 1 - NPV
  FOR <- 1-NPV
  # 'FPR' ('FAR') = FP/(FP+TN) = 1 - specificity
  FPR <- 1-specificity

  # 'geometric mean' (G)
  geometric.mean <- sqrt(recall*specificity)

  # 'L' ('LR(+)')
  L <- recall/(1-specificity)
  # 'lambda' ('LR(-)')
  lambda <- (1-recall)/specificity

  # 'DOR' = L/lambda = TP*TN/(FP*FN)
  DOR <- L/lambda

  # 'MK' = precision + NPV - 1
  MK <- precision + NPV - 1

  # 'OP' = accuracy - \recall-specificity\/(recall+specificity)
  OP <- accuracy - abs(recall-specificity)/(recall+specificity)

  # 'Youden's index'
  Youden <- recall+specificity-1

  # FINAL OUTPUT
  metrics_list <- list(accuracy = accuracy,
                       balanced.accuracy = balanced.accuracy,
                       DOR = DOR,
                       error.rate = error.rate,
                       F0.5 = F0.5,
                       F1 = F1,
                       F2 = F2,
                       FDR = FDR,
                       FNR = FNR,
                       FOR = FOR,
                       FPR = FPR,
                       geometric.mean = geometric.mean,
                       Jaccard = Jaccard,
                       L = L,
                       lambda = lambda,
                       MCC = MCC,
                       MK = MK,
                       NPV = NPV,
                       OP = OP,
                       precision = precision,
                       recall = recall,
                       specificity = specificity,
                       Youden = Youden)
  rm(balanced.accuracy, error.rate, F0.5, F1, F2, geometric.mean, L, lambda,
     MCC, recall, precision, specificity, Youden)
  if(output.as.table) return(as.data.frame(metrics_list, row.names = class_label_names)[,-c(1,4)])
  return(metrics_list)
}
