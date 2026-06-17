use std::sync::Arc;

use faer::prelude::Solve;
use faer::{Col, ColRef, Mat, MatRef};
use rayon::iter::IndexedParallelIterator;
use rayon::prelude::*;

use crate::xmodels::feature_map::KernelFeatureMap;

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

    /// Sigmoid function: $\sigma(x) = \frac{1}{1 + \exp(-x)}$
    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    /// Fits the binary classifier using the IRLS algorithm.
    ///
    /// This implementation uses target centering (y - mean) to align with the Regressor's logic.
    /// The `base_value` serves as the learned intercept, eliminating the need for an explicit bias column:
    /// $w_{new} = w_{old} + (Z^T R Z + \lambda I)^{-1} Z^T (y - \mu)$
    pub fn fit(&mut self, x: MatRef<'_, f32>, y: ColRef<'_, f32>) {
        if self.kernel_feature_map.is_none() {
            let mut map = KernelFeatureMap::new();
            map.fit(x);
            self.kernel_feature_map = Some(Arc::new(map));
        }
        let map = self.kernel_feature_map.as_ref().unwrap();

        // Allocate space for coefficients and compute initial values
        let n_samples = x.nrows();
        let n_features = map.num_features;
        let n_bases = map.num_bases;
        let total_dim = n_features * n_bases;

        // Validate that the number of rows matches the number of target values and weights
        if n_samples != y.nrows() {
            panic!(
                "Mismatched dimensions: The number of rows in X ({}) must match the number of target values ({}).",
                n_samples,
                y.nrows()
            );
        }

        // Initialize intercept in logit space based on target mean
        let mean_y = y.iter().sum::<f32>() / n_samples as f32;
        let eps = 1e-6;
        let p_clamped = mean_y.clamp(eps, 1.0 - eps);
        self.base_value = (p_clamped / (1.0 - p_clamped)).ln();

        // Pre-compute Z matrix once
        let mut z = Mat::<f32>::zeros(n_samples, total_dim);
        z.as_mut()
            .par_col_partition_mut(n_features)
            .enumerate()
            .for_each(|(f_idx, mut z_f): (usize, faer::MatMut<'_, f32>)| {
                map.transform_feature_into(x, f_idx, z_f.as_mut());
            });

        // Initialize coefficients
        let mut w = Col::<f32>::zeros(total_dim);

        // IRLS Iteration
        for _ in 0..self.max_iter {
            let base_val = self.base_value;

            // Prediction in logit space: raw_pred = Z * w + base_value
            let mut raw_pred = &z * &w;
            for i in 0..n_samples {
                raw_pred[i] += base_val;
            }

            // Compute probabilities, working weights, and errors
            let mut r_val = Col::<f32>::zeros(n_samples);
            let mut err = Col::<f32>::zeros(n_samples);
            for i in 0..n_samples {
                let prob = Self::sigmoid(raw_pred[i]);
                r_val[i] = (prob * (1.0 - prob)).max(1e-5);
                err[i] = y[i] - prob;
            }

            // Hessian: H = Z^T * Diag(R) * Z
            let mut z_w = z.clone();
            z_w.as_mut()
                .par_row_partition_mut(n_samples)
                .enumerate()
                .for_each(|(i, mut row): (usize, faer::MatMut<'_, f32>)| {
                    let r = r_val[i];
                    for j in 0..total_dim {
                        row[(0, j)] *= r;
                    }
                });

            let mut hessian = z.transpose() * &z_w;

            // Gradient: g = Z^T * (y - prob)
            let rhs = z.transpose() * &err;

            // Add L2 regularization (Ridge)
            for p_idx in 0..total_dim {
                hessian[(p_idx, p_idx)] += self.penalty;
            }

            // Solve the normal equations (H * delta_w = gradient) using LDLT decomposition
            let delta_w = if let Ok(ldlt) = hessian.ldlt(faer::Side::Lower) {
                ldlt.solve(&rhs)
            } else {
                break;
            };

            // Convergence check based on the update magnitude
            let update_mag: f32 = delta_w.iter().map(|&x| x.abs()).sum();
            if update_mag <= 1e-6 {
                break;
            }
            w += delta_w;
        }

        // De-stack the weight vector into per-feature coefficients
        self.coefficients = (0..n_features)
            .map(|f_idx| {
                let start = f_idx * n_bases;
                w.as_ref().subrows(start, n_bases).to_owned()
            })
            .collect();
    }

    /// Predicts class probabilities for the given input matrix X.
    ///
    /// Returns a vector of probabilities for the positive class (1).
    pub fn predict(&self, x: MatRef<'_, f32>) -> Col<f32> {
        let map = self.kernel_feature_map.as_ref().expect("Model not fitted");
        let n_samples = x.nrows();
        let n_features = map.num_features;
        let n_bases = map.num_bases;
        let total_dim = n_features * n_bases;

        if n_features != x.ncols() {
            panic!(
                "Mismatched dimensions: The number of columns in the feature map ({}) must match the number of input columns ({}).",
                n_features,
                x.ncols()
            );
        }

        // Transform features into the kernel space Z
        let mut z = Mat::<f32>::zeros(n_samples, total_dim);
        z.as_mut()
            .par_col_partition_mut(n_features)
            .enumerate()
            .for_each(|(f_idx, mut z_f): (usize, faer::MatMut<'_, f32>)| {
                map.transform_feature_into(x, f_idx, z_f.as_mut());
            });

        // Stack all coefficients into one big column
        let mut all_coeffs = Col::<f32>::zeros(total_dim);
        for f_idx in 0..n_features {
            all_coeffs
                .as_mut()
                .subrows_mut(f_idx * n_bases, n_bases)
                .copy_from(&self.coefficients[f_idx]);
        }

        // Prediction: prob = sigmoid(Z * coefficients + base_value)
        let mut prediction = z * &all_coeffs;
        for r_idx in 0..n_samples {
            prediction[r_idx] = Self::sigmoid(prediction[r_idx] + self.base_value);
        }

        prediction
    }

    /// Explains the model's prediction by decomposing it into individual feature contributions.
    ///
    /// For each feature $i$, it calculates the contribution $C_i = Z_i \cdot \alpha_i$,
    /// resulting in a matrix where each column represents the contribution of a specific feature.
    pub fn explain(&self, x: MatRef<'_, f32>) -> Mat<f32> {
        let map = self.kernel_feature_map.as_ref().expect("Model not fitted");
        let n_samples = x.nrows();
        let n_features = map.num_features;
        let n_bases = map.num_bases;

        if n_features != x.ncols() {
            panic!(
                "Mismatched dimensions: The number of columns in the feature map ({}) must match the number of input columns ({}).",
                n_features,
                x.ncols()
            );
        }

        let mut contributions = Mat::<f32>::zeros(n_samples, n_features);
        contributions
            .as_mut()
            .par_col_partition_mut(n_features)
            .enumerate()
            .for_each(|(f_idx, z_f): (usize, faer::MatMut<'_, f32>)| {
                let mut out_z = Mat::<f32>::zeros(n_samples, n_bases);
                map.transform_feature_into(x, f_idx, out_z.as_mut());
                let col_contrib = out_z * &self.coefficients[f_idx];
                z_f.col_mut(0).copy_from(&col_contrib);
            });
        contributions
    }
}
