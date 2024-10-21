# Supervised Classification

In this chapter, we introduce the following supervised classfication algorithms with examples:
* Binary Logistic Regression
* Naive Bayes
* Support Vector Machine (SVM)
* Random Forest


## Binary Logistic Regression
We start with the Study Hour example from [wiki](https://en.wikipedia.org/wiki/Logistic_regression):
<em>A group of 20 students spends between 0 and 6 hours studying for an exam. How does the number of hours spent studying affect the probability of the student passing the exam? (pass - 1, fail - 0)</em>

| Hours ($x_i$) | 0.50 |	0.75 |	1.00 | 1.25	| 1.50 | 1.75 | 2.00 | 2.25 | 2.50 | 2.75 | 3.00 | 3.25 | 3.50 | 4.00 | 4.25 | 4.50 | 4.75 | 5.00 | 5.50 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pass ($y_i$) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 1 | 0 | 1 | 1 | 1 | 1 | 1 | 1 |

Using logistic regression, we build the model in the following way:

```math
\text{pred}_i = \frac{1}{1 + e^{-(w_i*x_i+b_i)}}
```

To solve the classification problem, we use the cross entropy loss:
```math
L_{CE}(\text{pred}_i, y_i) = -y_i*\log(\text{pred}_i) - (1-y_i) * \log(1 - \text{pred}_i)
```

We can use SGD for the loss optimization.
