## Precision-Recall Curve and ROC Curve for Binary and Multi-class classification

### Binary classification
Given a list of different ML models for binary classification, e.g., $P_i(y|x)=p_i\in[0,1]$, there are two questions we want to explore:

* how to choose the best model $i$ from the list
* given a model $i$, how do you choose the best threshold for production?

A common strategy to answer these two questions is to use the Precision-Recall Curve or the ROC Curve. Let's first define these two curves. Given a list of strictly increasing thresholds $t_0=0, t_1, ...,t_k,..., t_N=1.0$:
* Precision-Recall Curve: the plot of $Precision_k=\frac{TP_k}{TP_k+FP_k}$ and $Recall_k=\frac{TP_k}{TP_k + FN_k}$. Note that when $t_0=0,FN_0=0->Recall_0=1$, and when threshold increases, Recall drops and Precision increases.
* ROC Curve: the plot of $FPR_k=\frac{FP_k}{FP_k+TN_k}$ and $Recall_k=\frac{TP_k}{TP_k + FN_k}$. Note that same as above, $t_0=0,FN_0=TN_0=0->Recall_0=FPR_0=1$, but when the threshold increases, both Recall and FPR drop to 0 at $t_n=1$.

A straightforward comparison of the Precision-Recall and ROC Curve can be found [here](https://stackoverflow.com/questions/59519995/roc-curve-and-precision-recall-curve)

* To answer the first question above (how to choose the best model $i$), we can use the Precision-Recall AUC or ROC AUC. AUC stands for "Area Under Curve". For different models, different Precision-Recall and ROC curves could be plotted and compared. The larger the AUC is, the better the model is (see [detailed tutorial with sklearn](https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-imbalanced-classification/)), e.g.,
```math
\text{best model}=argmax_i AUC(model_i)
```

* To answer the second question above (how to choose the best threshold $t_k$), we can compare the distances between $<FPR, Recall>$ or $<Precision, Recall>$ pairs and choose the pair with the minimal absolute difference (see [example](https://www.iguazio.com/glossary/classification-threshold/#:~:text=The%20ROC%20curve%20gives%20a,FPR%E2%80%94is%20the%20optimal%20threshold.))

How do you intuitively understand the ROC and PR curves? For example, if we have a binomial distribution of samples (P and N):
![image](https://github.com/user-attachments/assets/3bb8746e-6c48-452f-a36e-1acfba1d98c4)

For any machine learning model designed, the goal is to successfully separate the P samples from the N samples. If the model has a very high threshold then $TP$ and $FP$ are both $0$, which makes $FPR_k=\frac{FP_k}{FP_k+TN_k}=0$ and $Recall_k=\frac{TP_k}{TP_k + FN_k}=0$, and it corresponds the ROC curve on the bottom left corner:
![image](https://github.com/user-attachments/assets/411c6e29-12c5-4b08-80c8-4858be4de188)

If the model has a very low threshold then $TN$ and $FN$ are both $0$, which makes $FPR_k=\frac{FP_k}{FP_k+TN_k}=1$ and $Recall_k=\frac{TP_k}{TP_k + FN_k}=1$, which corresponds the point on the ROC curve on top right:
![image](https://github.com/user-attachments/assets/1e319632-fd1d-4cf5-8b1e-ae5cd3f4b588)

For PR curve it's the other way around. There is a "perfect" split so the PR curve will reach its furthest point, and then either precision or recall will drop if the model shifts toward left or right:
![image](https://github.com/user-attachments/assets/3ab37fdc-e9f7-49ae-adef-bd6ac7ed8658)



### Q&A
1. When to choose Precision-Recall Curve and when to choose ROC Curve?\
   * To
   * Ref: https://stats.stackexchange.com/questions/7207/roc-vs-precision-and-recall-curves
