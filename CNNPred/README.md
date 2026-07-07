### Implementation Variations
Each of the following scripts corresponds to a specific idea proposed in the paper:

*   **`CNNPred_with_Bayesian.py`**: Implementation of the model using Bayesian optimization/inference approaches.
*   **`CNNPred_with_statistical_features.py`**: Incorporates advanced statistical features into the CNNPred.
*   **`CNNPred_with_t-test.py`**: Leverages t-test statistics for feature validation or selection.
*   **`CNNPred_with_BTMF.py`**:  Integrates the Bayesian Temporal Modeling Framework (BTMF) to dynamically update and refine prediction probabilities using market indicators like volatility, and returns.
## Configuration

The hyperparameters of the statistical methods — including the **p-value threshold**, 
the **window size for skewness and kurtosis computation**, and the **number of Monte Carlo (MC) samples** — 
were selected based on a validation set constructed from the last **10% of the training data**.

## Usage

To run any proposed idea:

1. Place the repository folder inside your project directory.
2. Execute the corresponding script for the desired idea.
