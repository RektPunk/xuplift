use std::sync::Arc;

use faer::prelude::Solve;
use faer::{Col, Mat};
use rayon::prelude::*;

use crate::feature_map::KernelFeatureMap;

/// A Ridge Regressor that uses transformed non-linear features.
pub struct Regressor {
    /// The kernel_feature_map responsible for kernel-based feature mapping.
    pub kernel_feature_map: Arc<KernelFeatureMap>,
    /// The Ridge regularization penalty factor.
    pub penalty: f32,
    /// The global mean of the target variable (used for centering).
    pub base_value: f32,
    /// Learned coefficients for each feature block.
    pub coefficients: Vec<Col<f32>>,
}

impl Regressor {
    /// Creates a new Regressor instance with a fitted KernelFeatureMap.
    pub fn new(kernel_feature_map: Arc<KernelFeatureMap>, penalty: f32) -> Self {
        Self {
            kernel_feature_map,
            penalty,
            base_value: 0.0,
            coefficients: Vec::new(),
        }
    }

    /// Fits the model using Global Ridge Regression.
    ///
    /// This method solves the system: (Z^T * Z + lambda * I) * alpha = Z^T * y_centered
    pub fn fit(&mut self, y: &Col<f32>) {
        let num_rows = self.kernel_feature_map.num_rows;
        let weights = Col::<f32>::full(num_rows, 1.0);
        self.fit_weighted(y, &weights);
    }

    /// Fits the model using Weighted Global Ridge Regression.
    ///
    /// This method solves the system: (Z^T * W * Z + lambda * I) * alpha = Z^T * W * y_centered
    /// where W is a diagonal weight matrix.
    pub fn fit_weighted(&mut self, y: &Col<f32>, weights: &Col<f32>) {
        let num_rows = self.kernel_feature_map.num_rows;
        let num_features = self.kernel_feature_map.num_features;
        let num_bases = self.kernel_feature_map.num_bases;

        if num_rows != y.nrows() || num_rows != weights.nrows() {
            panic!(
                "Mismatched dimensions: The number of rows in the feature map ({}) must match the number of target values ({}) and weights ({}).",
                num_rows,
                y.nrows(),
                weights.nrows()
            );
        }

        let total_weight: f32 = weights.iter().sum();
        self.base_value = if total_weight > 1e-6 {
            weights
                .iter()
                .zip(y.iter())
                .map(|(&w, &yi)| w * yi)
                .sum::<f32>()
                / total_weight
        } else {
            y.iter().sum::<f32>() / num_rows as f32
        };

        let y_centered = y - Col::<f32>::full(num_rows, self.base_value);
        let total_dim = num_features * num_bases;

        // Initialize the Hessian (LHS) and Gradient (RHS) for the normal equations
        let mut ridge_lhs = Mat::<f32>::zeros(total_dim, total_dim);
        let mut rhs = Col::<f32>::zeros(total_dim);

        // Block-based accumulation to save memory:
        // We compute ridge_lhs = Z^T * W * Z and rhs = Z^T * W * y_centered
        // by iterating over feature blocks Z_i and Z_j.
        for i in 0..num_features {
            let z_i = &self.kernel_feature_map.z_matrices[i];
            let offset_i = i * num_bases;

            // Compute RHS contribution: Z_i^T * W * y_centered
            for r in 0..num_rows {
                let w_y = weights[r] * y_centered[r];
                for k in 0..num_bases {
                    rhs[offset_i + k] += z_i[(r, k)] * w_y;
                }
            }

            for j in 0..num_features {
                let z_j = &self.kernel_feature_map.z_matrices[j];
                let offset_j = j * num_bases;

                // Accumulate Z_i^T * W * Z_j into the global Hessian matrix
                for r in 0..num_rows {
                    let w = weights[r];
                    for k in 0..num_bases {
                        let val_i = z_i[(r, k)] * w;
                        for l in 0..num_bases {
                            ridge_lhs[(offset_i + k, offset_j + l)] += val_i * z_j[(r, l)];
                        }
                    }
                }
            }
        }

        // Add L2 regularization (Ridge) to the diagonal
        for i in 0..total_dim {
            ridge_lhs[(i, i)] += self.penalty;
        }

        // Solve the linear system using LDLT decomposition
        let alpha_total = ridge_lhs.ldlt(faer::Side::Lower).unwrap().solve(&rhs);

        self.coefficients = (0..num_features)
            .into_par_iter()
            .map(|f_idx| {
                let start = f_idx * num_bases;
                alpha_total.as_ref().subrows(start, num_bases).to_owned()
            })
            .collect();
    }
    /// Predicts target values for the given input matrix X.
    ///
    /// It maps X to the kernel space and calculates the weighted sum of contributions.
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
        let prediction = (0..num_features)
            .into_par_iter()
            .map(|f_idx| &z_matrices[f_idx] * &self.coefficients[f_idx]) // Z_i * coeff_i
            .reduce(
                || Col::<f32>::zeros(num_rows),
                |mut acc, res| {
                    acc += res;
                    acc
                },
            );
        // Restore the target scale by adding back the base value (mean)
        prediction.map(|v| v + self.base_value)
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
        let contributions_vec: Vec<Col<f32>> = (0..num_features)
            .into_par_iter()
            .map(|f_idx| &z_matrices[f_idx] * &self.coefficients[f_idx])
            .collect();
        Mat::from_fn(x.nrows(), num_features, |i, j| contributions_vec[j][i])
    }
}
