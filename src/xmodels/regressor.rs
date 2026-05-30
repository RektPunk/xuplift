use std::sync::Arc;

use faer::prelude::Solve;
use faer::{Col, ColRef, Mat, MatRef};
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
    pub fn fit(&mut self, x: MatRef<'_, f32>, y: ColRef<'_, f32>) {
        let weights = Col::<f32>::full(x.nrows(), 1.0);
        self.fit_weighted(x, y, &weights);
    }

    /// Fits the model using Weighted Global Ridge Regression.
    ///
    /// This method solves the system: $(Z^T W Z + \lambda I) \alpha = Z^T W (y - b)$
    /// where $W$ is a diagonal weight matrix.
    pub fn fit_weighted(&mut self, x: MatRef<'_, f32>, y: ColRef<'_, f32>, weights: &Col<f32>) {
        // Initialize and fit the kernel map if it hasn't been set yet
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
            y.iter().sum::<f32>() / n_samples as f32
        };

        let total_dim = n_features * n_bases;

        // Accumulate Hessian and Gradient in parallel
        let (mut ridge_lhs, rhs, _) = (0..n_samples)
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
                    let w = weights[r];
                    let y_c = y[r] - self.base_value;

                    for i in 0..total_dim {
                        let z_i = z_r[i];
                        acc_g[i] += z_i * w * y_c;
                        let val_i = z_i * w;
                        for j in 0..total_dim {
                            acc_h[(i, j)] += val_i * z_r[j];
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

        // Add L2 regularization (Ridge) to the diagonal
        for i in 0..total_dim {
            ridge_lhs[(i, i)] += self.penalty;
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

        if n_features != x.ncols() {
            panic!(
                "Mismatched dimensions: The number of columns in the feature map ({}) must match the number of input columns ({}).",
                n_features,
                x.ncols()
            );
        }

        // Feature-wise streaming prediction
        let mut prediction = (0..n_features)
            .into_par_iter()
            .map(|f_idx| {
                let mut z_f = Mat::<f32>::zeros(n_samples, n_bases);
                map.transform_feature_into(x, f_idx, z_f.as_mut());
                z_f * &self.coefficients[f_idx]
            })
            .reduce(
                || Col::<f32>::zeros(n_samples),
                |mut acc, res| {
                    acc += res;
                    acc
                },
            );

        for i in 0..n_samples {
            prediction[i] += self.base_value;
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

        let mut contributions = vec![0.0f32; n_samples * n_features];

        contributions
            .par_chunks_exact_mut(n_samples)
            .enumerate()
            .for_each(|(f_idx, col_slice)| {
                let mut z_f = Mat::<f32>::zeros(n_samples, n_bases);
                map.transform_feature_into(x, f_idx, z_f.as_mut());
                let col_contrib = z_f * &self.coefficients[f_idx];

                for i in 0..n_samples {
                    col_slice[i] = col_contrib[i];
                }
            });

        faer::MatRef::from_column_major_slice(&contributions, n_samples, n_features).to_owned()
    }
}
