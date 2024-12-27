# weighted-pls-regression

## Overview

This project implements Weighted Partial Least Squares (WPLS) Regression, a method that incorporates sample weights into the standard PLS regression model.

## Motivation

The default PLSRegression implementation in scikit-learn does not support sample weights. I couldn't also find Python implementations with sample weights.


## Environment

This implementation has been tested in the following environment:
- OS: Ubuntu 22.04.4 LTS
- Python: 3.10.14
- Libraries:
    - numpy: 1.26.4
	- scikit-learn: 1.5.0

## Features
- Supports sample weights for flexible regression modeling.
- Compatible with scikit-learn’s API, enabling integration into pipelines.


## Limitation
- Standardization Toggle: 
Currently, standardization cannot be turned off. 
This feature may be added in future updates.

## Tutorial

A tutorial (tutorial.ipynb) shows how to use this implementation and compares it with scikit-learn’s PLSRegression.
In my environment, the results (MSE, coefficients, intercepts) matched.


## License
This project is licensed under the MIT License.

## Contact
For questions, suggestions, or bug reports, please feel free to:
- Open an issue.
- Email me (junyaihira[@]gmail.com).

Your feedback is highly appreciated!