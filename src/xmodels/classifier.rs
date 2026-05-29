use std::sync::Arc;

use faer::prelude::Solve;
use faer::{Col, ColRef, Mat, MatRef};
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
    pub fn fit(&mut self, x: MatRef<'_, f32>, y: ColRef<'_, f32>) {
        // Initialize and fit the kernel map
        let mut map = KernelFeatureMap::new();
        map.fit(x);

        // Allocate space for coefficients and compute initial values
        let n_samples = x.nrows();
        let n_features = map.num_features;
        let n_bases = map.num_bases;

        // Validate that the number of rows matches the number of target values
        if n_samples != y.nrows() {
            panic!(
                "Mismatched dimensions: The number of rows in X ({}) must match the number of target values ({}).",
                n_samples,
                y.nrows()
            );
        }

        // Calculate the mean of target 'y' and initialize base_value in logit space
        let mean_y = y.iter().sum::<f32>() / n_samples as f32;
        let eps = 1e-6;
        let p_clamped = mean_y.clamp(eps, 1.0 - eps);
        self.base_value = (p_clamped / (1.0 - p_clamped)).ln();

        let total_dim = n_features * n_bases;

        // Initialize coefficients
        let mut w = Col::<f32>::zeros(total_dim);

        // IRLS Iteration: Streaming approach to save memory
        for _ in 0..self.max_iter {
            let base_val = self.base_value;
            let (mut hessian, rhs, _) = (0..n_samples)
                .into_par_iter()
                .fold(
                    || {
                        (
                            Mat::<f32>::zeros(total_dim, total_dim),
                            Col::<f32>::zeros(total_dim),
                            Col::<f32>::zeros(total_dim),
                        )
                    },
                    |(mut acc_h, mut acc_g, mut z_r), r| {
                        map.transform_row_into(x, r, z_r.as_mut());

                        // Linear prediction: raw_pred = z^T * w + base_value
                        let mut raw_pred = base_val;
                        for i in 0..total_dim {
                            raw_pred += z_r[i] * w[i];
                        }

                        let prob = Self::sigmoid(raw_pred);
                        let r_val = (prob * (1.0 - prob)).max(1e-5);
                        let err = y[r] - prob;

                        for k in 0..total_dim {
                            let z_k = z_r[k];
                            acc_g[k] += z_k * err;
                            let val_k = z_k * r_val;
                            for l in 0..total_dim {
                                acc_h[(k, l)] += val_k * z_r[l];
                            }
                        }
                        (acc_h, acc_g, z_r)
                    },
                )
                .reduce(
                    || {
                        (
                            Mat::<f32>::zeros(total_dim, total_dim),
                            Col::<f32>::zeros(total_dim),
                            Col::<f32>::zeros(0),
                        )
                    },
                    |(mut h1, mut g1, _), (h2, g2, _)| {
                        for j in 0..total_dim {
                            g1[j] += g2[j];
                            for i in 0..total_dim {
                                h1[(i, j)] += h2[(i, j)];
                            }
                        }
                        (h1, g1, Col::<f32>::zeros(0))
                    },
                );

            // Add L2 regularization (Ridge)
            for i in 0..total_dim {
                hessian[(i, i)] += self.penalty;
            }

            // Solve the normal equations (H * delta_w = gradient) using LDLT decomposition
            let delta_w = if let Ok(ldlt) = hessian.ldlt(faer::Side::Lower) {
                ldlt.solve(&rhs)
            } else {
                // If Hessian is singular, stop iterations early
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
            .into_par_iter()
            .map(|f_idx| {
                let start = f_idx * n_bases;
                w.as_ref().subrows(start, n_bases).to_owned()
            })
            .collect();

        // Store the kernel map
        self.kernel_feature_map = Some(Arc::new(map));
    }

    /// Predicts class probabilities for the given input matrix X.
    ///
    /// Returns a vector of probabilities for the positive class (1).
    pub fn predict(&self, x: MatRef<'_, f32>) -> Col<f32> {
        let map = self
            .kernel_feature_map
            .as_ref()
            .expect("Model must be fitted before prediction.");
        let n_samples = x.nrows();
        let n_features = map.num_features;

        if n_features != x.ncols() {
            panic!(
                "Mismatched dimensions: The number of columns in the feature map ({}) must match the number of input columns ({}).",
                n_features,
                x.ncols()
            );
        }

        // Process in chunks to save memory
        let chunk_size = 10000.min(n_samples);
        let mut p_pred = Col::<f32>::zeros(n_samples);

        for start_row in (0..n_samples).step_by(chunk_size) {
            let end_row = (start_row + chunk_size).min(n_samples);
            let n_chunk = end_row - start_row;
            let x_chunk = x.subrows(start_row, n_chunk);

            let z_matrices = map.transform_per_feature(x_chunk);

            let chunk_pred = (0..n_features)
                .into_par_iter()
                .map(|f_idx| &z_matrices[f_idx] * &self.coefficients[f_idx])
                .reduce(
                    || Col::<f32>::zeros(n_chunk),
                    |mut acc, res| {
                        acc += res;
                        acc
                    },
                );

            for i in 0..n_chunk {
                p_pred[start_row + i] = Self::sigmoid(chunk_pred[i] + self.base_value);
            }
        }
        p_pred
    }

    /// Explains the model's prediction by decomposing it into individual feature contributions.
    ///
    /// For each feature $i$, it calculates the contribution $C_i = Z_i \cdot \alpha_i$,
    /// resulting in a matrix where each column represents the contribution of a specific feature.
    pub fn explain(&self, x: MatRef<'_, f32>) -> Mat<f32> {
        let map = self
            .kernel_feature_map
            .as_ref()
            .expect("Model must be fitted before explanation.");
        let n_samples = x.nrows();
        let n_features = map.num_features;

        if n_features != x.ncols() {
            panic!(
                "Mismatched dimensions: The number of columns in the feature map ({}) must match the number of input columns ({}).",
                n_features,
                x.ncols()
            );
        }

        // Process in chunks to save memory
        let chunk_size = 10000.min(n_samples);
        let mut contributions = Mat::<f32>::zeros(n_samples, n_features);

        for start_row in (0..n_samples).step_by(chunk_size) {
            let end_row = (start_row + chunk_size).min(n_samples);
            let n_chunk = end_row - start_row;
            let x_chunk = x.subrows(start_row, n_chunk);

            let z_matrices = map.transform_per_feature(x_chunk);

            let chunk_contributions: Vec<Col<f32>> = (0..n_features)
                .into_par_iter()
                .map(|f_idx| &z_matrices[f_idx] * &self.coefficients[f_idx])
                .collect();

            for j in 0..n_features {
                for i in 0..n_chunk {
                    contributions[(start_row + i, j)] = chunk_contributions[j][i];
                }
            }
        }
        contributions
    }
}
