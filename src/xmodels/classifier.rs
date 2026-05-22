use std::sync::Arc;

use faer::prelude::Solve;
use faer::{Col, Mat};
use rayon::prelude::*;

use crate::feature_map::KernelFeatureMap;

/// A Binary Classifier using Nystrom features and Iteratively Reweighted Least Squares (IRLS).
///
/// It solves the Logistic Regression problem in the transformed feature space $Z$:
/// $$P(y=1|z) = \sigma(z^T w + b)$$
/// where $\sigma(x) = \frac{1}{1 + e^{-x}}$ is the sigmoid function,
/// $w$ are the coefficients, and $b$ is the intercept.
///
/// The model is fitted using the IRLS algorithm, which iteratively updates weights $w$:
/// $w_{new} = w_{old} + (Z^T R Z + \lambda I)^{-1} Z^T (y - \mu)$
/// where $R$ is a diagonal matrix with $R_{ii} = \mu_i (1 - \mu_i)$.
pub struct Classifier {
    /// The kernel_feature_map responsible for kernel-based feature mapping.
    pub kernel_feature_map: Option<Arc<KernelFeatureMap>>,
    /// The Ridge regularization penalty factor.
    pub penalty: f32,
    /// The maximum number of iterations for the IRLS algorithm.
    pub max_iter: usize,
    /// The global mean of the target variable, used as an implicit bias (intercept).
    pub base_value: f32,
    /// Learned weight coefficients for each feature block.
    pub coefficients: Vec<Col<f32>>,
}

impl Classifier {
    /// Creates a new Classifier instance.
    pub fn new(penalty: f32, max_iter: usize) -> Self {
        Self {
            kernel_feature_map: None,
            penalty,
            max_iter,
            base_value: 0.0,
            coefficients: Vec::new(),
        }
    }

    /// Sigmoid function: sigma(x) = 1 / (1 + exp(-x))
    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    /// Fits the binary classifier using the IRLS algorithm.
    ///
    /// This implementation uses target centering (y - mean) to align with the Regressor's logic.
    /// The `base_value` serves as the learned intercept, eliminating the need for an explicit bias column.
    pub fn fit(&mut self, x: &Mat<f32>, y: &Col<f32>) {
        // Initialize and fit the kernel map
        let mut map = KernelFeatureMap::new();
        map.fit(x);

        // Allocate space for coefficients and compute initial values
        let num_rows = x.nrows();
        let num_features = map.num_features;
        let num_bases = map.num_bases;

        // Validate that the number of rows matches the number of target values
        if num_rows != y.nrows() {
            panic!(
                "Mismatched dimensions: The number of rows in X ({}) must match the number of target values ({}).",
                num_rows,
                y.nrows()
            );
        }

        // Calculate the mean of target 'y' and initialize base_value in logit space
        let mean_y = y.iter().sum::<f32>() / num_rows as f32;
        let eps = 1e-6;
        let p_clamped = mean_y.clamp(eps, 1.0 - eps);
        self.base_value = (p_clamped / (1.0 - p_clamped)).ln();

        let total_dim = num_features * num_bases;

        // Initialize weights
        let mut w = Col::<f32>::zeros(total_dim);

        // IRLS Iteration
        for _ in 0..self.max_iter {
            // Current linear prediction: a = Z * w + base_value
            // Pass 1: Compute curr_raw_pred row-by-row
            let mut curr_raw_pred = Col::<f32>::zeros(num_rows);
            for r in 0..num_rows {
                let z_r = map.transform_row(x, r);
                curr_raw_pred[r] =
                    z_r.iter().zip(w.iter()).map(|(&a, &b)| a * b).sum::<f32>() + self.base_value;
            }

            // Transform predictions to probabilities: mu = sigmoid(a)
            let curr_prob = curr_raw_pred.map(|&v| Self::sigmoid(v));

            // Compute the diagonal weight matrix R: r_ii = mu * (1 - mu).
            let r_diag = curr_prob.map(|m| (m * (1.0 - m)).max(1e-5));

            // Calculate the error (gradient component): y - mu
            let error = y - &curr_prob;

            // Construct the Hessian (H = Z^T * R * Z + lambda * I) and RHS (Z^T * error)
            let mut hessian = Mat::<f32>::zeros(total_dim, total_dim);
            let mut rhs = Col::<f32>::zeros(total_dim);

            // Pass 2: Accumulate Hessian and RHS row-by-row
            for r in 0..num_rows {
                let z_r = map.transform_row(x, r);
                let err = error[r];
                let r_val = r_diag[r];

                // RHS contribution: Z_r^T * error
                for k in 0..total_dim {
                    rhs[k] += z_r[k] * err;
                }

                // Hessian contribution: Z_r^T * R * Z_r
                for k in 0..total_dim {
                    let val_k = z_r[k] * r_val;
                    for l in 0..total_dim {
                        hessian[(k, l)] += val_k * z_r[l];
                    }
                }
            }

            // Add L2 regularization (Ridge)
            for i in 0..total_dim {
                hessian[(i, i)] += self.penalty;
            }

            // Solve the normal equations (H * delta_w = gradient) using LDLT decomposition.
            let delta_w = hessian.ldlt(faer::Side::Lower).unwrap().solve(&rhs);

            // Convergence check based on the update magnitude.
            if delta_w.iter().map(|x| x.abs()).sum::<f32>() <= 1e-6 {
                break;
            }
            w += delta_w;
        }

        // De-stack the weight vector into per-feature coefficients.
        self.coefficients = (0..num_features)
            .into_par_iter()
            .map(|f_idx| {
                let start = f_idx * num_bases;
                w.as_ref().subrows(start, num_bases).to_owned()
            })
            .collect();

        // Store the kernel map.
        self.kernel_feature_map = Some(Arc::new(map));
    }

    /// Predicts class probabilities for the given input matrix X.
    ///
    /// Returns a vector of probabilities for the positive class (1).
    pub fn predict(&self, x: &Mat<f32>) -> Col<f32> {
        let map = self
            .kernel_feature_map
            .as_ref()
            .expect("Model must be fitted before prediction.");
        // Validate that the number of columns in the input matches the number of features in the feature map
        let num_features = map.num_features;
        let num_rows = x.nrows();
        if num_features != x.ncols() {
            panic!(
                "Mismatched dimensions: The number of columns in the feature map ({}) must match the number of input columns ({}).",
                num_features,
                x.ncols()
            );
        }
        // Map raw input to the feature space
        let z_matrices = map.transform(x);

        // Parallel computation of y_pred = Sum(Z_i * coeff_i)
        let linear_pred = (0..num_features)
            .into_par_iter()
            .map(|f_idx| &z_matrices[f_idx] * &self.coefficients[f_idx])
            .reduce(
                || Col::<f32>::zeros(num_rows),
                |mut acc, res| {
                    acc += res;
                    acc
                },
            );
        // Apply sigmoid activation, incorporating the base_value as the global intercept.
        linear_pred.map(|v| Self::sigmoid(v + self.base_value))
    }

    /// Explains the model's prediction by decomposing it into individual feature contributions.
    ///
    /// For each feature $i$, it calculates the contribution $C_i = Z_i \cdot \alpha_i$,
    /// resulting in a matrix where each column represents the contribution of a specific feature.
    pub fn explain(&self, x: &Mat<f32>) -> Mat<f32> {
        let map = self
            .kernel_feature_map
            .as_ref()
            .expect("Model must be fitted before explanation.");
        // Validate that the number of columns in the input matches the number of features in the feature map
        let num_features = map.num_features;
        if num_features != x.ncols() {
            panic!(
                "Mismatched dimensions: The number of columns in the feature map ({}) must match the number of input columns ({}).",
                num_features,
                x.ncols()
            );
        }

        // Map raw input to the feature space
        let z_matrices = map.transform(x);

        // Parallel computation of comtribution vec
        let contributions_vec: Vec<Col<f32>> = (0..x.ncols())
            .into_par_iter()
            .map(|f_idx| &z_matrices[f_idx] * &self.coefficients[f_idx])
            .collect();
        Mat::from_fn(x.nrows(), x.ncols(), |i, j| contributions_vec[j][i])
    }
}
