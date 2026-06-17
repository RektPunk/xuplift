use std::sync::Arc;

use faer::prelude::Solve;
use faer::{Col, ColRef, Mat, MatRef};
use rayon::iter::IndexedParallelIterator;
use rayon::prelude::*;

use crate::xmodels::feature_map::KernelFeatureMap;

/// A Ridge Regressor that uses transformed non-linear features.
///
/// It solves the Global Ridge Regression problem in the transformed feature space $Z$:
/// $$\min_{\alpha} ||y - Z\alpha - b||^2 + \lambda ||\alpha||^2$$
/// where $Z$ is the $n \times (d \cdot m)$ matrix of kernel-mapped features,
/// $\alpha$ are the coefficients, $b$ is the intercept, and $\lambda$ is the penalty.
pub struct Regressor {
    /// The kernel_feature_map responsible for kernel-based feature mapping.
    pub kernel_feature_map: Option<Arc<KernelFeatureMap>>,
    /// The Ridge regularization penalty factor.
    pub penalty: f32,
    /// The global mean of the target variable (used for centering).
    pub base_value: f32,
    /// Learned coefficients for each feature block.
    pub coefficients: Vec<Col<f32>>,
}

impl Regressor {
    /// Creates a new Regressor instance.
    pub fn new(penalty: f32) -> Self {
        Self {
            kernel_feature_map: None,
            penalty,
            base_value: 0.0,
            coefficients: Vec::new(),
        }
    }

    /// Fits the model using Global Ridge Regression.
    ///
    /// This method solves the system: $(Z^T Z + \lambda I) \alpha = Z^T (y - b)$
    pub fn fit(&mut self, x: MatRef<'_, f32>, y: ColRef<'_, f32>) {
        let weights = Col::<f32>::full(x.nrows(), 1.0);
        self.fit_weighted(x, y, &weights);
    }

    /// Fits the model using Weighted Global Ridge Regression.
    ///
    /// This method solves the system: $(Z^T W Z + \lambda I) \alpha = Z^T W (y - b)$
    /// where $W$ is a diagonal weight matrix.
    pub fn fit_weighted(&mut self, x: MatRef<'_, f32>, y: ColRef<'_, f32>, weights: &Col<f32>) {
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
        if n_samples != y.nrows() || n_samples != weights.nrows() {
            panic!(
                "Mismatched dimensions: The number of rows in X ({}) must match the number of target values ({}) and weights ({}).",
                n_samples,
                y.nrows(),
                weights.nrows()
            );
        }

        // The base_value $b$ is the weighted mean of the target $y$
        let total_weight: f32 = weights.iter().sum();
        self.base_value = if total_weight > 1e-6 {
            weights
                .iter()
                .zip(y.iter())
                .map(|(&w, &yi)| w * yi)
                .sum::<f32>()
                / total_weight
        } else {
            y.iter().sum::<f32>() / n_samples as f32
        };

        // Transform features into the kernel space Z
        let mut z = Mat::<f32>::zeros(n_samples, total_dim);

        z.as_mut()
            .par_col_partition_mut(n_features)
            .enumerate()
            .for_each(|(f_idx, mut z_f): (usize, faer::MatMut<'_, f32>)| {
                map.transform_feature_into(x, f_idx, z_f.as_mut());
            });

        // Compute y_centered = y - base_value
        let mut y_w = Col::<f32>::from_fn(n_samples, |i| y[i] - self.base_value);

        // Apply weights: H = Z^T * W * Z  =>  H = (Z * sqrt(W))^T * (Z * sqrt(W))
        z.as_mut()
            .par_row_partition_mut(n_samples)
            .enumerate()
            .for_each(|(i, mut row): (usize, faer::MatMut<'_, f32>)| {
                let w = weights[i];
                if (w - 1.0).abs() > 1e-6 {
                    let sqrt_w = w.sqrt();
                    for j in 0..total_dim {
                        row[(0, j)] *= sqrt_w;
                    }
                }
            });

        for i in 0..n_samples {
            y_w[i] *= weights[i];
        }

        // Hessian: H = Z^T * W * Z
        let mut ridge_lhs = z.transpose() * &z;

        // Gradient: g = Z^T * (W * (y - b) / sqrt(W)) = Z^T * (sqrt(W) * (y - b))
        for i in 0..n_samples {
            let w = weights[i];
            let factor = if w > 1e-6 { w.sqrt() } else { 0.0 };
            y_w[i] = (y[i] - self.base_value) * factor;
        }
        let rhs = z.transpose() * &y_w;

        // Add L2 regularization (Ridge) to the diagonal
        for p_idx in 0..total_dim {
            ridge_lhs[(p_idx, p_idx)] += self.penalty;
        }

        // Solve the linear system using LDLT decomposition
        let alpha_total = ridge_lhs.ldlt(faer::Side::Lower).unwrap().solve(&rhs);

        self.coefficients = (0..n_features)
            .map(|f_idx| {
                let start = f_idx * n_bases;
                alpha_total.as_ref().subrows(start, n_bases).to_owned()
            })
            .collect();
    }

    /// Predicts target values for the given feature matrix X.
    ///
    /// The prediction is: $\hat{y} = Z \alpha + b = \sum_{j} (Z_j \alpha_j) + b$.
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

        // Prediction: y_hat = Z * coefficients + base_value
        let mut prediction = z * &all_coeffs;
        for r_idx in 0..n_samples {
            prediction[r_idx] += self.base_value;
        }

        prediction
    }

    /// Explains the model's prediction by decomposing it into individual feature contributions.
    ///
    /// For each feature $i$, it calculates the contribution $C_i = Z_i \cdot \alpha_i$,
    /// such that $\sum C_i + b = \hat{y}$.
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
