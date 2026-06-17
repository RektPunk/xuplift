use std::sync::Arc;

use faer::prelude::Solve;
use faer::{Col, ColRef, Mat, MatRef};
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
        let mut map = KernelFeatureMap::new();
        map.fit(x);

        // Allocate space for coefficients and compute initial values
        let n_samples = x.nrows();
        let n_features = map.num_features;
        let n_bases = map.num_bases;

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

        let total_dim = n_features * n_bases;

        // Transform features into the kernel space Z
        let z = Mat::<f32>::zeros(n_samples, total_dim);
        (0..n_features).into_par_iter().for_each(|f_idx| {
            let start = f_idx * n_bases;
            let mut z_f = unsafe {
                // Safe because each thread accesses a disjoint submatrix
                let z_ptr = z.as_ptr() as *mut f32;
                let row_stride = z.row_stride();
                let col_stride = z.col_stride();
                faer::MatMut::from_raw_parts_mut(
                    z_ptr.offset((start as isize) * col_stride),
                    n_samples,
                    n_bases,
                    row_stride,
                    col_stride,
                )
            };
            map.transform_feature_into(x, f_idx, z_f.as_mut());
        });

        // Compute y_centered = y - base_value
        let y_centered = Col::<f32>::from_fn(n_samples, |i| y[i] - self.base_value);

        // Apply weights: Z_w = diag(W) * Z and y_w = diag(W) * y_centered
        let z_w = z.clone();
        let mut y_w = y_centered.clone();

        let z_w_ptr = z_w.as_ptr() as usize;
        let row_stride = z_w.row_stride();
        let col_stride = z_w.col_stride();

        (0..n_samples).into_par_iter().for_each(|i| {
            let w = weights[i];
            unsafe {
                let ptr = (z_w_ptr as *mut f32).offset(i as isize * row_stride);
                for j in 0..total_dim {
                    let val_ptr = ptr.offset(j as isize * col_stride);
                    *val_ptr *= w;
                }
            }
        });

        for i in 0..n_samples {
            y_w[i] *= weights[i];
        }

        // Hessian: H = Z^T * Z_w
        let mut ridge_lhs = z.transpose() * &z_w;

        // Gradient: g = Z^T * y_w (which is Z^T * W * (y - b))
        let rhs = z.transpose() * &y_w;

        // Add L2 regularization (Ridge) to the diagonal
        for p_idx in 0..total_dim {
            ridge_lhs[(p_idx, p_idx)] += self.penalty;
        }

        // Solve the linear system using LDLT decomposition
        let alpha_total = ridge_lhs.ldlt(faer::Side::Lower).unwrap().solve(&rhs);

        self.coefficients = (0..n_features)
            .into_par_iter()
            .map(|f_idx| {
                let start = f_idx * n_bases;
                alpha_total.as_ref().subrows(start, n_bases).to_owned()
            })
            .collect();

        // Store the kernel map
        self.kernel_feature_map = Some(Arc::new(map));
    }

    /// Predicts target values for the given feature matrix X.
    ///
    /// The prediction is: $\hat{y} = Z \alpha + b = \sum_{j} (Z_j \alpha_j) + b$.
    pub fn predict(&self, x: MatRef<'_, f32>) -> Col<f32> {
        let map = self
            .kernel_feature_map
            .as_ref()
            .expect("Model must be fitted before prediction.");
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
        let z = Mat::<f32>::zeros(n_samples, total_dim);
        (0..n_features).into_par_iter().for_each(|f_idx| {
            let start = f_idx * n_bases;
            let mut z_f = unsafe {
                let z_ptr = z.as_ptr() as *mut f32;
                let row_stride = z.row_stride();
                let col_stride = z.col_stride();
                faer::MatMut::from_raw_parts_mut(
                    z_ptr.offset((start as isize) * col_stride),
                    n_samples,
                    n_bases,
                    row_stride,
                    col_stride,
                )
            };
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
        let map = self
            .kernel_feature_map
            .as_ref()
            .expect("Model must be fitted before explanation.");
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

        let contributions = Mat::<f32>::zeros(n_samples, n_features);

        (0..n_features).into_par_iter().for_each(|f_idx| {
            let mut z_f = Mat::<f32>::zeros(n_samples, n_bases);
            map.transform_feature_into(x, f_idx, z_f.as_mut());
            let col_contrib = z_f * &self.coefficients[f_idx];

            let mut out_col = unsafe {
                let out_ptr = contributions.as_ptr() as *mut f32;
                let row_stride = contributions.row_stride();
                let col_stride = contributions.col_stride();
                faer::ColMut::from_raw_parts_mut(
                    out_ptr.offset((f_idx as isize) * col_stride),
                    n_samples,
                    row_stride,
                )
            };
            out_col.copy_from(&col_contrib);
        });

        contributions
    }
}
