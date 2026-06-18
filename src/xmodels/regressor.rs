use std::sync::Arc;

use faer::prelude::Solve;
use faer::{Col, ColRef, Mat, MatMut, MatRef};
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
    /// The maximum number of landmark points used in the kernel feature map.
    pub max_bases: usize,
    /// The Ridge regularization penalty factor.
    pub penalty: f32,
    /// The global mean of the target variable (used for centering).
    pub base_value: f32,
    /// Learned coefficients for each feature block.
    pub coefficients: Vec<Col<f32>>,
}

impl Regressor {
    /// Creates a new Regressor instance.
    pub fn new(max_bases: usize, penalty: f32) -> Self {
        Self {
            kernel_feature_map: None,
            max_bases,
            penalty,
            base_value: 0.0,
            coefficients: Vec::new(),
        }
    }

    /// Fits the model using Weighted Global Ridge Regression using a pre-computed Z matrix.
    ///
    /// This method solves the system: $(Z^T W Z + \lambda I) \alpha = Z^T W (y - b)$
    /// where $W$ is a diagonal weight matrix.
    pub fn fit_with_z(&mut self, z: MatRef<'_, f32>, y: ColRef<'_, f32>, weights: &Col<f32>) {
        let n_samples = z.nrows();
        let total_dim = z.ncols();

        if n_samples != y.nrows() || n_samples != weights.nrows() {
            panic!(
                "Mismatched dimensions: The number of rows in Z ({}) must match the number of target values ({}) and weights ({}).",
                n_samples,
                y.nrows(),
                weights.nrows()
            );
        }

        // The base_value $b$ is the weighted mean of the target $y$
        let total_weight: f32 = weights.iter().sum();
        self.base_value = if total_weight > 1e-5 {
            weights
                .iter()
                .zip(y.iter())
                .map(|(&w, &yi)| w * yi)
                .sum::<f32>()
                / total_weight
        } else {
            y.iter().sum::<f32>() / n_samples as f32
        };

        // Z is scaled by sqrt(W) for Hessian: H = (Z * sqrt(W))^T * (Z * sqrt(W))
        let mut z_w = Mat::<f32>::zeros(n_samples, total_dim);
        z_w.as_mut()
            .par_row_partition_mut(n_samples)
            .enumerate()
            .for_each(|(i, mut row): (usize, MatMut<'_, f32>)| {
                let w = weights[i];
                let factor = if w > 1e-5 { w.sqrt() } else { 0.0 };
                let z_row = z.row(i);
                for j in 0..total_dim {
                    row[(0, j)] = z_row[j] * factor;
                }
            });

        // And y centered and scaled: y_w = (y - b) * sqrt(W)
        let base_val = self.base_value;
        let y_w = Col::<f32>::from_fn(n_samples, |i| {
            let w = weights[i];
            let factor = if w > 1e-5 { w.sqrt() } else { 0.0 };
            (y[i] - base_val) * factor
        });

        // Hessian: H = Z_w^T * Z_w
        let mut ridge_lhs = z_w.transpose() * &z_w;
        let rhs = z_w.transpose() * &y_w;

        // Add L2 regularization (Ridge) to the diagonal
        for p_idx in 0..total_dim {
            ridge_lhs[(p_idx, p_idx)] += self.penalty;
        }

        // Solve the linear system using LDLT decomposition
        let alpha_total = ridge_lhs.ldlt(faer::Side::Lower).unwrap().solve(&rhs);

        let n_features = self.kernel_feature_map.as_ref().unwrap().num_features;
        let n_bases = self.kernel_feature_map.as_ref().unwrap().num_bases;

        // De-stack the weight vector into per-feature coefficients
        self.coefficients = (0..n_features)
            .map(|f_idx| {
                let start = f_idx * n_bases;
                alpha_total.as_ref().subrows(start, n_bases).to_owned()
            })
            .collect();
    }

    /// Fits the model using Weighted Global Ridge Regression.
    pub fn fit_weighted(
        &mut self,
        x: MatRef<'_, f32>,
        y: ColRef<'_, f32>,
        weights: &Col<f32>,
        is_categorical: &[bool],
    ) {
        if self.kernel_feature_map.is_none() {
            let mut map = KernelFeatureMap::new(self.max_bases);
            map.fit(x, is_categorical);
            self.kernel_feature_map = Some(Arc::new(map));
        }

        let map = self.kernel_feature_map.as_ref().unwrap();
        let z = map.transform(x);

        self.fit_with_z(z.as_ref(), y, weights);
    }

    /// Fits the model using Global Ridge Regression.
    pub fn fit(&mut self, x: MatRef<'_, f32>, y: ColRef<'_, f32>, is_categorical: &[bool]) {
        let weights = Col::<f32>::full(x.nrows(), 1.0);
        self.fit_weighted(x, y, &weights, is_categorical);
    }

    /// Predicts target values using a pre-computed Z matrix.
    ///
    /// The prediction is: $\hat{y} = Z \alpha + b = \sum_{j} (Z_j \alpha_j) + b$.
    pub fn predict_with_z(&self, z: MatRef<'_, f32>) -> Col<f32> {
        let n_samples = z.nrows();
        let total_dim = z.ncols();

        let n_features = self.kernel_feature_map.as_ref().unwrap().num_features;
        let n_bases = self.kernel_feature_map.as_ref().unwrap().num_bases;

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

    /// Predicts target values for the given feature matrix X.
    pub fn predict(&self, x: MatRef<'_, f32>) -> Col<f32> {
        let map = self.kernel_feature_map.as_ref().expect("Model not fitted");
        let z = map.transform(x);
        self.predict_with_z(z.as_ref())
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
            .for_each(|(f_idx, z_f): (usize, MatMut<'_, f32>)| {
                let mut out_z = Mat::<f32>::zeros(n_samples, n_bases);
                map.transform_feature_into(x, f_idx, out_z.as_mut());
                let col_contrib = out_z * &self.coefficients[f_idx];
                z_f.col_mut(0).copy_from(&col_contrib);
            });
        contributions
    }
}
