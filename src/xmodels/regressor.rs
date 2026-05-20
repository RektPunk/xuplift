use std::sync::Arc;

use faer::prelude::Solve;
use faer::{Col, Mat};
use rayon::prelude::*;

use crate::feature_map::KernelFeatureMap;

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
    pub fn fit(&mut self, x: &Mat<f32>, y: &Col<f32>) {
        let weights = Col::<f32>::full(x.nrows(), 1.0);
        self.fit_weighted(x, y, &weights);
    }

    /// Fits the model using Weighted Global Ridge Regression.
    ///
    /// This method solves the system: $(Z^T W Z + \lambda I) \alpha = Z^T W (y - b)$
    /// where $W$ is a diagonal weight matrix.
    ///
    /// The system is solved efficiently using LDLT decomposition of the Hessian.
    pub fn fit_weighted(&mut self, x: &Mat<f32>, y: &Col<f32>, weights: &Col<f32>) {
        // Initialize and fit the kernel map if it hasn't been set yet
        if self.kernel_feature_map.is_none() {
            let mut map = KernelFeatureMap::new();
            map.fit(x);
            self.kernel_feature_map = Some(Arc::new(map));
        }
        let map = self.kernel_feature_map.as_ref().unwrap();

        let num_rows = x.nrows();
        let num_features = map.num_features;
        let num_bases = map.num_bases;

        if num_rows != y.nrows() || num_rows != weights.nrows() {
            panic!(
                "Mismatched dimensions: The number of rows in X ({}) must match the number of target values ({}) and weights ({}).",
                num_rows,
                y.nrows(),
                weights.nrows()
            );
        }

        let total_weight: f32 = weights.iter().sum();
        // The base_value $b$ is the weighted mean of the target $y$.
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

        // Initialize the Hessian $H = Z^T W Z$ and Gradient $g = Z^T W (y-b)$
        let mut ridge_lhs = Mat::<f32>::zeros(total_dim, total_dim);
        let mut rhs = Col::<f32>::zeros(total_dim);

        // Row-based accumulation to save memory:
        // We compute $H$ and $g$ by iterating over rows $z_r$:
        // $H = \sum_r w_r z_r z_r^T$
        // $g = \sum_r w_r z_r (y_r - b)$
        for r in 0..num_rows {
            let z_r = map.transform_row(x, r);
            let w = weights[r];
            let y_c = y_centered[r];

            // Accumulate RHS: Z_r^T * W * y_c
            for i in 0..total_dim {
                rhs[i] += z_r[i] * w * y_c;
            }

            // Accumulate Hessian: Z_r^T * W * Z_r
            for i in 0..total_dim {
                let val_i = z_r[i] * w;
                for j in 0..total_dim {
                    ridge_lhs[(i, j)] += val_i * z_r[j];
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
    /// The prediction is: $\hat{y} = Z \alpha + b = \sum_{j} (Z_j \alpha_j) + b$.
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
    /// such that $\sum C_i + b = \hat{y}$.
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
        let contributions_vec: Vec<Col<f32>> = (0..num_features)
            .into_par_iter()
            .map(|f_idx| &z_matrices[f_idx] * &self.coefficients[f_idx])
            .collect();
        Mat::from_fn(x.nrows(), num_features, |i, j| contributions_vec[j][i])
    }
}
