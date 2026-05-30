use faer::{Col, ColMut, Mat, MatRef};
use rand::rng;
use rand::seq::SliceRandom;
use rayon::prelude::*;

/// A transformer that approximates kernel feature maps using the Nystrom method.
///
/// It maps input data into a finite-dimensional feature space where linear
/// operations approximate non-linear kernels (e.g., RBF kernel).
///
/// The Nystrom method approximates the kernel matrix $K$ by using a small subset
/// of $m$ landmark points:
/// $$K \approx K_{nm} K_{mm}^{-1} K_{mn}$$
/// where $K_{nm}$ is the kernel between all $n$ samples and $m$ landmarks,
/// and $K_{mm}$ is the kernel between landmarks.
#[derive(Default)]
pub struct KernelFeatureMap {
    /// Number of input features (columns).
    pub num_features: usize,
    /// Number of landmark points (basis functions) per feature.
    pub num_bases: usize,
    /// Selected landmark samples from the training set.
    /// Each entry in the vector contains the landmark values per feature.
    pub feature_bases: Vec<Col<f32>>,

    /// Learned projection matrices to map data into the kernel space
    /// $P = U \Lambda^{-1/2}$, where $U$ and $\Lambda$ are the eigenvectors and eigenvalues of $K_{mm}$
    pub proj_matrices: Vec<Mat<f32>>,
    /// Column-wise means of the transformed features for centering.
    /// $\mu_j = \frac{1}{n} \sum_{i=1}^n z_{ij}$
    pub feature_means: Vec<Col<f32>>,
    /// Inverse of the kernel bandwidth parameter (gamma) for each feature.
    /// Calculated using the Median Heuristic: $\gamma = \frac{1}{2\sigma^2}$
    pub s2_invs: Vec<f32>,
}

impl KernelFeatureMap {
    /// Returns a new KernelFeatureMap instance
    pub fn new() -> Self {
        Self::default()
    }

    /// Fits the transformer to the input matrix X.
    pub fn fit(&mut self, x: MatRef<'_, f32>) {
        let n_samples = x.nrows();
        self.num_features = x.ncols();

        // Calculate feature means (skipping NaNs) to use for imputation during landmark selection
        let raw_feature_means: Vec<f32> = (0..self.num_features)
            .into_par_iter()
            .map(|f_idx| {
                let col = x.col(f_idx);
                let mut sum = 0.0;
                let mut count = 0;
                for i in 0..n_samples {
                    let val = col[i];
                    if !val.is_nan() {
                        sum += val;
                        count += 1;
                    }
                }
                if count > 0 { sum / count as f32 } else { 0.0 }
            })
            .collect();

        // Identify rows that have no NaNs across all features to use as high-quality landmark candidates
        let valid_row_indices: Vec<usize> = (0..n_samples)
            .into_par_iter()
            .filter(|&r_idx| (0..self.num_features).all(|f_idx| !x[(r_idx, f_idx)].is_nan()))
            .collect();
        let n_valid = valid_row_indices.len();

        // Select landmarks (defaults to min(N, 64) for efficiency)
        let landmark_indices = if n_valid >= 32 {
            self.num_bases = n_valid.min(64);
            let mut rng = rng();
            let mut indices = valid_row_indices.clone();
            indices.shuffle(&mut rng);
            indices[..self.num_bases].to_vec()
        } else {
            self.num_bases = n_samples.min(64);
            let mut all_indices: Vec<usize> = (0..n_samples).collect();
            let mut rng = rng();
            all_indices.shuffle(&mut rng);
            all_indices[..self.num_bases].to_vec()
        };

        let feature_params: Vec<_> = (0..self.num_features)
            .into_par_iter()
            .map(|f_idx| {
                let f_mean = raw_feature_means[f_idx];

                // Median Heuristic: sets sigma to the median of pairwise distances between landmarks
                let mut dists = Vec::with_capacity(self.num_bases * self.num_bases / 2);
                for i in 0..self.num_bases {
                    let val_i = {
                        let v = x[(landmark_indices[i], f_idx)];
                        if v.is_nan() { f_mean } else { v }
                    };
                    for j in i + 1..self.num_bases {
                        let val_j = {
                            let v = x[(landmark_indices[j], f_idx)];
                            if v.is_nan() { f_mean } else { v }
                        };
                        dists.push((val_i - val_j).abs());
                    }
                }
                dists.sort_by(|a, b| a.total_cmp(b));
                let median = if !dists.is_empty() {
                    let mid = dists.len() / 2;
                    if dists.len() % 2 == 0 {
                        (dists[mid] + dists[mid - 1]) * 0.5
                    } else {
                        dists[mid]
                    }
                } else {
                    1.0
                };

                // Precision parameter $\gamma = 1 / (2 \cdot \text{median}^2)$.
                let s2_inv = 1.0 / (2.0 * (median.max(1e-4)).powi(2));

                // Store landmark values
                let mut bases = Col::<f32>::zeros(self.num_bases);
                for (j_idx, &row_idx) in landmark_indices.iter().enumerate() {
                    let val = x[(row_idx, f_idx)];
                    bases[j_idx] = if val.is_nan() { f_mean } else { val };
                }

                // Compute Landmark Kernel matrix K_mm: $k_{ij} = \exp(-\gamma ||u_i - u_j||^2)$
                let mut k_mm = Mat::<f32>::zeros(self.num_bases, self.num_bases);
                for i in 0..self.num_bases {
                    for j in i..self.num_bases {
                        let diff = bases[i] - bases[j];
                        let val = (-(diff * diff) * s2_inv).exp();
                        k_mm[(i, j)] = val;
                        if i != j {
                            k_mm[(j, i)] = val;
                        }
                    }
                }

                // Eigen-decomposition for symmetric inverse square root: $K_{mm}^{-1/2} = U \Lambda^{-1/2} U^T$
                // This ensures that the transformed features are approximately orthonormal.
                let eig = k_mm.self_adjoint_eigen(faer::Side::Lower).unwrap();
                let mut inv_s = Mat::<f32>::zeros(self.num_bases, self.num_bases);
                for d in 0..self.num_bases {
                    let val = eig.S()[d];
                    inv_s[(d, d)] = if val > 1e-10 { 1.0 / val.sqrt() } else { 0.0 };
                }

                let proj_matrix = eig.U() * &inv_s;

                // Compute feature means for centering without storing full $Z$ matrix.
                // Since $Z = K_{nm} P$, the column means are:
                // $\text{mean}(Z) = \text{mean}(K_{nm}) P$
                let mut k_col_sums = Col::<f32>::zeros(self.num_bases);
                for i in 0..n_samples {
                    let x_val = x[(i, f_idx)];
                    if !x_val.is_nan() {
                        for j in 0..self.num_bases {
                            let diff = x_val - bases[j];
                            k_col_sums[j] += (-(diff * diff) * s2_inv).exp();
                        }
                    }
                }
                let mut z_col_means = Col::<f32>::zeros(self.num_bases);
                for l in 0..self.num_bases {
                    let mut sum = 0.0;
                    for j in 0..self.num_bases {
                        sum += k_col_sums[j] * proj_matrix[(j, l)];
                    }
                    z_col_means[l] = sum / n_samples as f32;
                }

                (bases, proj_matrix, z_col_means, s2_inv)
            })
            .collect();

        for (b, p, o, s) in feature_params {
            self.feature_bases.push(b);
            self.proj_matrices.push(p);
            self.feature_means.push(o);
            self.s2_invs.push(s);
        }
    }

    /// Transforms an entire row into the joint kernel feature space by concatenating all mapped features.
    ///
    /// For a given $x$, it computes the mapped feature for each feature $f$ and landmark $l$,
    /// and stores it at the corresponding layout offset $f \cdot m + l$:
    /// $$\text{out}[f \cdot m + l] = \left( \sum_{j=1}^m k(x_f, u_{f, j}) P_{f, jl} \right) - \mu_{f, l}$$
    /// where $m$ is `num_bases`, $k$ is the RBF kernel, $P_f$ is the projection matrix, and $\mu_f$ is the centering mean.
    pub fn transform_row_into(&self, x: MatRef<'_, f32>, row_idx: usize, mut out: ColMut<'_, f32>) {
        // Temporary buffer on the stack to store intermediate kernel calculations.
        // Capped at 64 as per the Nystrom landmark selection logic.
        let mut kernel_cache = [0.0f32; 64];
        for f_idx in 0..self.num_features {
            let x_val = x[(row_idx, f_idx)];
            let offset = f_idx * self.num_bases;

            // Handle missing values by zeroing the output for this feature.
            if x_val.is_nan() {
                for j in 0..self.num_bases {
                    out[offset + j] = 0.0;
                }
                continue;
            }

            let bases = &self.feature_bases[f_idx];
            let proj = &self.proj_matrices[f_idx];
            let mean = &self.feature_means[f_idx];
            let s2_inv = self.s2_invs[f_idx];

            // Pre-calculate RBF kernel distances between input and landmarks
            for j in 0..self.num_bases {
                let diff = x_val - bases[j];
                kernel_cache[j] = (-(diff * diff) * s2_inv).exp();
            }

            // Map into the learned Nystrom feature space via linear projection and centering
            for l in 0..self.num_bases {
                let mut projection_sum = 0.0;
                for j in 0..self.num_bases {
                    projection_sum += kernel_cache[j] * proj[(j, l)];
                }
                out[offset + l] = projection_sum - mean[l];
            }
        }
    }

    /// Transforms a specific feature column across all rows into its Nystrom kernel feature space.
    ///
    /// For a given feature index $f$ and all sample rows $i$, it computes the mapped feature
    /// for each landmark $l$ and stores it in the output matrix at position $(i, l)$:
    /// $$\text{out}[i, l] = \left( \sum_{j=1}^m k(x_{i, f}, u_{f, j}) P_{f, jl} \right) - \mu_{f, l}$$
    /// where $m$ is `num_bases`, $k$ is the RBF kernel, $P_f$ is the projection matrix,
    /// and $\mu_f$ is the centering mean for that feature.
    pub fn transform_feature_into(
        &self,
        x: MatRef<'_, f32>,
        f_idx: usize,
        mut out: faer::MatMut<'_, f32>,
    ) {
        let n_samples = x.nrows();
        let bases = &self.feature_bases[f_idx];
        let proj = &self.proj_matrices[f_idx];
        let mean = &self.feature_means[f_idx];
        let s2_inv = self.s2_invs[f_idx];

        for i in 0..n_samples {
            let x_val = x[(i, f_idx)];

            // Handle missing values.
            if x_val.is_nan() {
                for l in 0..self.num_bases {
                    out[(i, l)] = 0.0;
                }
                continue;
            }

            // Temporary buffer on the stack to store intermediate kernel calculations.
            // Capped at 64 as per the Nystrom landmark selection logic.
            let mut kernel_cache = [0.0f32; 64];

            // Pre-calculate RBF kernel distances between input and landmarks
            for j in 0..self.num_bases {
                let diff = x_val - bases[j];
                kernel_cache[j] = (-(diff * diff) * s2_inv).exp();
            }

            // Map into the learned Nystrom feature space via linear projection and centering
            for l in 0..self.num_bases {
                let mut projection_sum = 0.0;
                for j in 0..self.num_bases {
                    projection_sum += kernel_cache[j] * proj[(j, l)];
                }
                out[(i, l)] = projection_sum - mean[l];
            }
        }
    }
}
