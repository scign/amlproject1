# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
This dataset contains data about bank marketing calls. We seek to predict whether a call will result in a successful term loan subscription.

The best performing model was a stacked ensemble model, identified using Azure AutoML.

## HyperDrive optimization

### Parameter Sampler
I set the parameter sampler to explore both `lbfgs` and `saga` algorithms, with `l1` and `l2` penalty regularizatiion, as well as explore the regularization strength parameter `C`. I also set the sampler to sample from a logarithmic space for `C` as nature of the inverse of the regularization strength is logarithmic. This allowed HyperDrive to search the space for `C` more efficiently, while additionally checking whether the regularization penalty as well as the solver algorithm itself have any significant effect on the model performance. I used a random sampler so that as much of the space could be explored in a short time to gain insight into the distribution across the parameter space.

### Early Stopping Policy
Due to time constraints I needed an early stopping policy that would terminate low performing runs sooner than later. The Bandit policy with a 20% slack ratio terminates runs if the primary metric does not match the best run so far within 20%. Assume we use accuracy as the primary metric. This dataset has a large class imbalance such that a majority-class classifier would achieve an 88.8% accuracy. Assume at least one run achieves an accuracy of 88.8%; setting the slack ratio to 20% and evaluation iterations to 10 means that any subsequent run that fails to achieve at least a 74% accuracy by the 10th iteration will be terminated. This is a reasonable expectation for this classification problem therefore this policy is appropriate.

## AutoML
**In 1-2 sentences, describe the model and hyperparameters generated by AutoML.**

## Pipeline comparison
**Compare the two models and their performance. What are the differences in accuracy? In architecture? If there was a difference, why do you think there was one?**

## Future work
1. More attention to the dataset cleaning. There are many variables in the model that should have been cleaned more diligently, for example call duration is not known at the start of a call therefore should not be part of the model, and the 'month' feature should also be discarded as data is over multiple years and socioeconomic features correlate with month. I have included a supplementary script file that contains a better data cleaning routine which yields a model that achieves a `norm_macro_recall` of almost 0.5.
1. Analysis of the feature influences on the outcome will yield insights into the marketing practices of this bank and may assist to refine call targets. For example while the education and employment features weigh strongly on the outcome, the socioeconomic factors are actually very strong influencers so in an uncertain market the bank may look to focusing efforts on a subset of their target population to maximize uptake of loan offers.
1. Analysis of feature influences will also yield insights into marketing practice fairness and equity. While the ratio of term loan subscription among literate people ranges between 7% and 15%, the subscription ratio among illiterate people is 26% (although the sample size is small: just 15 individuals in the dataset). This could be indicative of predatory lending practices at this bank which if not catered for in the model, may exacerbate the practice if the model indicates that illiterate people are a prime target for such loan offers.
1. A model trained on this data could be used in two ways: While an obvious use case would be to predict the outcome of a call prior to the call being made, another use case may be in marketing data auditing, to check whether the data collected by the marketing department 'fits' this model. Anomalies in any feature may be indicative of erroneous data, which could be a result of customer drift, system errors and corrupted data, or fraud.

## Proof of cluster clean up
Completed in code