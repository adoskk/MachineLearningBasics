# Loss Functions

## Mean Square Error (MSE) Loss
```math
L_{MSE}(\text{pred}, y) = (\text{pred} - y)^2
```
Note: MSE loss is not convex for classification problems like logistic regression, see proof [here](https://towardsai.net/p/l/the-non-convexity-debate-in-machine-learning#84fe).

## Cross Entropy Loss
```math
L_{CE}(\text{pred}, y) = -y*log(\text{pred}) - (1 - y) * log(1 - \text{pred})
```
Note: Cross entropy loss can be derived from Bernoulli distribution for binary classification and Multinomial distribution for multi-class classification. See ref [here](https://web.stanford.edu/~jurafsky/slp3/5.pdf).
