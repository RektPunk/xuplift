use std::sync::Arc;

use faer::prelude::Solve;
use faer::{Col, Mat};
use rayon::prelude::*;

use crate::feature_map::KernelFeatureMap;

/// A Ridge Regressor that uses transformed non-linear features.
pub struct Regressor {
    /// The kernel_feature_map responsible for kernel-based feature mapping.
    pub kernel_feature_map: Arc<KernelFeatureMap>,
    /// The global mean of the target variable (used for centering).
    pub base_value: f32,
    /// Learned coefficients for each feature block.
    pub coefficients: Vec<Col<f32>>,
}

impl Regressor {
    /// Creates a new Regressor instance with a fitted KernelFeatureMap.
    pub fn new(kernel_feature_map: Arc<KernelFeatureMap>) -> Self {
        Self {
            kernel_feature_map,
            base_value: 0.0,
            coefficients: Vec::new(),
        }
    }

    /// Fits the model using Global Ridge Regression.
    ///
    /// This method solves the system: (Z^T * Z + lambda * I) * alpha = Z^T * y_centered
    pub fn fit(&mut self, y: &Col<f32>) {
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

        // Aggregate all feature matrices from the kernel_feature_map into a single design matrix (Z)
        let total_dim = num_features * num_bases;
        let mut z_stacked = Mat::<f32>::zeros(num_rows, total_dim);
        for (f_idx, z) in self.kernel_feature_map.z_matrices.iter().enumerate() {
            let offset = f_idx * num_bases;
            z_stacked
                .as_mut()
                .submatrix_mut(0, offset, num_rows, num_bases)
                .copy_from(z);
        }
        // Construct and solve the Normal Equation: (Z^T * Z + lambda * I)
        let lhs = z_stacked.transpose() * &z_stacked;
        let mut ridge_lhs = lhs;

        // Add L2 regularization (Ridge) to the diagonal for numerical stability
        let lambda = 0.01;
        for i in 0..total_dim {
            ridge_lhs[(i, i)] += lambda; // Ridge regularization
        }
        let rhs = z_stacked.transpose() * &y_centered;

        // Solve the linear system using LDLT decomposition
        let alpha_total = ridge_lhs.ldlt(faer::Side::Lower).unwrap().solve(&rhs);

        // Partition the global coefficient vector back into per-feature blocks
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
