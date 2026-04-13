use std::sync::Arc;

use faer::prelude::Solve;
use faer::{Col, Mat};
use rayon::prelude::*;

use crate::feature_map::KernelFeatureMap;

/// A Binary Classifier using Nystrom features and Iteratively Reweighted Least Squares (IRLS).
pub struct Classifier {
    /// The kernel_feature_map responsible for kernel-based feature mapping.
    pub kernel_feature_map: Arc<KernelFeatureMap>,
    /// The global mean of the target variable, used as an implicit bias (intercept).
    pub base_value: f32,
    /// Learned weight coefficients for each feature block.
    pub coefficients: Vec<Col<f32>>,
}

impl Classifier {
    /// Creates a new Classifier instance with a fitted KernelFeatureMap.
    pub fn new(kernel_feature_map: Arc<KernelFeatureMap>) -> Self {
        Self {
            kernel_feature_map,
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
    pub fn fit(&mut self, y: &Col<f32>, max_iter: usize) {
        let num_rows = self.kernel_feature_map.num_rows;
        let num_features = self.kernel_feature_map.num_features;
        let num_bases = self.kernel_feature_map.num_bases;

        // Validate that the number of rows in the feature map matches the number of target values
        if num_rows != y.nrows() {
            panic!(
                "Mismatched dimensions: The number of rows in the feature map ({}) must match the number of target values ({}).",
                num_rows,
                y.nrows()
            );
        }
        // Calculate the mean of target 'y' to center the data
        self.base_value = y.iter().sum::<f32>() / num_rows as f32;
        let y_centered = y - Col::<f32>::full(num_rows, self.base_value);
        let total_dim = num_features * num_bases;

        // Aggregate all feature matrices from the kernel_feature_map into a single design matrix (Z)
        let mut z_stacked = Mat::<f32>::zeros(num_rows, total_dim);
        for (f_idx, z) in self.kernel_feature_map.z_matrices.iter().enumerate() {
            let offset = f_idx * num_bases;
            z_stacked
                .as_mut()
                .submatrix_mut(0, offset, num_rows, num_bases)
                .copy_from(z);
        }

        // Initialize weights
        let mut w = Col::<f32>::zeros(total_dim);
        let lambda = 0.01; // Ridge regularization for stability

        // IRLS Iteration
        for _ in 0..max_iter {
            // Current linear prediction: a = Z * w
            let curr_raw_pred = &z_stacked * &w;

            // Transform predictions to probabilities: mu = sigmoid(a)
            let curr_prob = curr_raw_pred.map(|&v| Self::sigmoid(v));

            // Compute the diagonal weight matrix R: r_ii = mu * (1 - mu).
            let r_diag = curr_prob.map(|m| (m * (1.0 - m)).max(1e-5));

            // Calculate the error (gradient component).
            let error = &y_centered - &curr_prob;

            // Construct the weighted design matrix (Z_w = R * Z).
            let mut zw = z_stacked.clone();
            for i in 0..num_rows {
                for j in 0..total_dim {
                    zw[(i, j)] *= r_diag[i];
                }
            }

            // Compute the Hessian matrix: H = Z^T * R * Z + lambda * I.
            let mut hessian = z_stacked.transpose() * &zw;
            for i in 0..total_dim {
                hessian[(i, i)] += lambda;
            }

            // Solve the normal equations (H * delta_w = gradient) using LDLT decomposition.
            let rhs = z_stacked.transpose() * &error;
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
    }

    /// Predicts class probabilities for the given input matrix X.
    ///
    /// Returns a vector of probabilities for the positive class (1).
    pub fn predict(&self, x: &Mat<f32>) -> Col<f32> {
        // Validate that the number of columns in the input matches the number of features in the feature map
        let num_features = self.kernel_feature_map.num_features;
        let num_rows = x.nrows();
        if num_features != x.ncols() {
            panic!(
                "Mismatched dimensions: The number of columns in the feature map ({}) must match the number of input columns ({}).",
                num_features,
                x.ncols()
            );
        }
        // Map raw input to the feature space
        let z_matrices = self.kernel_feature_map.transform(x);

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
        // Validate that the number of columns in the input matches the number of features in the feature map
        let num_features = self.kernel_feature_map.num_features;
        if num_features != x.ncols() {
            panic!(
                "Mismatched dimensions: The number of columns in the feature map ({}) must match the number of input columns ({}).",
                num_features,
                x.ncols()
            );
        }

        // Map raw input to the feature space
        let z_matrices = self.kernel_feature_map.transform(x);

        // Parallel computation of comtribution vec
        let contributions_vec: Vec<Col<f32>> = (0..x.ncols())
            .into_par_iter()
            .map(|f_idx| &z_matrices[f_idx] * &self.coefficients[f_idx])
            .collect();
        Mat::from_fn(x.nrows(), x.ncols(), |i, j| contributions_vec[j][i])
    }
}
