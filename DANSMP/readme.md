### Implementation Variations
Each of the following scripts corresponds to a specific idea proposed in the paper:

*   **`DANSMP_with_Bayesian.py`**: Implementation of the model using Bayesian optimization/inference approaches.
*   **`DANSMP_with_statistical_features.py`**: Incorporates advanced statistical features into the DANSMP.
*   **`DANSMP_with_t-test.py`**: Leverages t-test statistics for feature validation or selection.
*   **`DANSMP_with_BTMF.py`**:  Integrates the GraphCNN (DANSMP) model with sentiment analysis and the Bayesian Temporal Modeling Framework (BTMF)
## Configuration

The hyperparameters of the statistical methods were selected based on a validation set constructed from the last **10% of the training data**.

## Usage

To run any proposed idea:

1. Place the repository folder inside your project directory.
2. Execute the corresponding script for the desired idea.
