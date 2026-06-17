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
    /// Indicates whether each feature is categorical or continuous.
    pub is_categorical: Vec<bool>,

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
    const MAX_BASES: usize = 64;
    const MAX_DIST_PAIRS: usize = Self::MAX_BASES * (Self::MAX_BASES - 1) / 2;

    /// Returns a new KernelFeatureMap instance
    pub fn new() -> Self {
        Self::default()
    }

    /// Fits the transformer to the input matrix X.
    pub fn fit(&mut self, x: MatRef<'_, f32>, is_categorical: &Vec<bool>) {
        let n_samples = x.nrows();
        self.num_features = x.ncols();
        self.is_categorical = is_categorical.clone();

        // Calculate feature means (skipping NaNs) to use for imputation during landmark selection
        let raw_feature_means: Vec<f32> = (0..self.num_features)
            .into_par_iter()
            .map(|f_idx| {
                let col = x.col(f_idx);
                let mut sum = 0.0;
                let mut count = 0;
                for r_idx in 0..n_samples {
                    let val = col[r_idx];
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

        // Select landmarks (defaults to min(N, MAX_BASES) for efficiency)
        let landmark_indices = if n_valid >= Self::MAX_BASES / 2 {
            self.num_bases = n_valid.min(Self::MAX_BASES);
            let mut rng = rng();
            let mut indices = valid_row_indices.clone();
            indices.shuffle(&mut rng);
            indices[..self.num_bases].to_vec()
        } else {
            self.num_bases = n_samples.min(Self::MAX_BASES);
            let mut all_indices: Vec<usize> = (0..n_samples).collect();
            let mut rng = rng();
            all_indices.shuffle(&mut rng);
            all_indices[..self.num_bases].to_vec()
        };

        let feature_params: Vec<_> = (0..self.num_features)
            .into_par_iter()
            .map(|f_idx| {
                let f_mean = raw_feature_means[f_idx];
                let is_categorical_f = is_categorical[f_idx];

                // Median Heuristic: sets sigma to the median of pairwise distances between landmarks
                let mut dists_buf = [0.0f32; Self::MAX_DIST_PAIRS];
                let mut dists_count = 0;
                for b_i_idx in 0..self.num_bases {
                    let val_i = {
                        let v = x[(landmark_indices[b_i_idx], f_idx)];
                        if v.is_nan() { f_mean } else { v }
                    };
                    for b_j_idx in b_i_idx + 1..self.num_bases {
                        let val_j = {
                            let v = x[(landmark_indices[b_j_idx], f_idx)];
                            if v.is_nan() { f_mean } else { v }
                        };
                        dists_buf[dists_count] = (val_i - val_j).abs();
                        dists_count += 1;
                    }
                }
                let dists = &mut dists_buf[..dists_count];
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

                // Precision parameter $\gamma = 1 / (2 \cdot \text{median}^2)$
                let s2_inv = 1.0 / (2.0 * (median.max(1e-4)).powi(2));

                // Store landmark values
                let mut bases = Col::<f32>::zeros(self.num_bases);
                for (b_idx, &r_idx) in landmark_indices.iter().enumerate() {
                    let val = x[(r_idx, f_idx)];
                    bases[b_idx] = if val.is_nan() { f_mean } else { val };
                }

                // Compute Landmark Kernel matrix K_mm: $k_{ij} = \exp(-\gamma ||u_i - u_j||^2)$
                let mut k_mm = Mat::<f32>::zeros(self.num_bases, self.num_bases);
                for b_i_idx in 0..self.num_bases {
                    for b_j_idx in b_i_idx..self.num_bases {
                        let diff = bases[b_i_idx] - bases[b_j_idx];
                        if is_categorical_f {
                            k_mm[(b_i_idx, b_j_idx)] = if diff < 1e-5 { 1.0 } else { 0.0 };
                        } else {
                            k_mm[(b_i_idx, b_j_idx)] = (-(diff * diff) * s2_inv).exp();
                        }

                        // Symmetric: k_ij = k_ji
                        if b_i_idx != b_j_idx {
                            k_mm[(b_j_idx, b_i_idx)] = k_mm[(b_i_idx, b_j_idx)];
                        }
                    }
                }

                // Eigen-decomposition for symmetric inverse square root: $K_{mm}^{-1/2} = U \Lambda^{-1/2} U^T$
                // This ensures that the transformed features are approximately orthonormal
                let eig = k_mm.self_adjoint_eigen(faer::Side::Lower).unwrap();
                let mut proj_matrix = eig.U().to_owned();

                for p_idx in 0..self.num_bases {
                    let val = eig.S()[p_idx];
                    let inv_sqrt_s = if val > 1e-10 { 1.0 / val.sqrt() } else { 0.0 };
                    // Efficiently scale each column by the inverse square root of eigenvalues
                    for b_idx in 0..self.num_bases {
                        proj_matrix[(b_idx, p_idx)] *= inv_sqrt_s;
                    }
                }

                // Compute feature means for centering without storing full $Z$ matrix
                // Since $Z = K_{nm} P$, the column means are:
                // $\text{mean}(Z) = \text{mean}(K_{nm}) P = (\frac{1}{n} \sum k_{i}) P$
                let mut k_col_sums = Col::<f32>::zeros(self.num_bases);
                for r_idx in 0..n_samples {
                    let x_val = x[(r_idx, f_idx)];
                    if !x_val.is_nan() {
                        for b_idx in 0..self.num_bases {
                            let diff = x_val - bases[b_idx];
                            if is_categorical_f {
                                k_col_sums[b_idx] += if diff < 1e-5 { 1.0 } else { 0.0 };
                            } else {
                                k_col_sums[b_idx] += (-(diff * diff) * s2_inv).exp();
                            }
                        }
                    }
                }

                // Compute z_col_means using matrix-vector multiplication: z_col_means = P^T * (k_col_sums / n)
                let mut z_col_means = proj_matrix.transpose() * &k_col_sums;
                let scale = 1.0 / n_samples as f32;
                for b_idx in 0..self.num_bases {
                    z_col_means[b_idx] *= scale;
                }

                (bases, proj_matrix, z_col_means, s2_inv)
            })
            .collect();

        for (bases, proj, means, s2_inv) in feature_params {
            self.feature_bases.push(bases);
            self.proj_matrices.push(proj);
            self.feature_means.push(means);
            self.s2_invs.push(s2_inv);
        }
    }

    /// Transforms an entire row into the joint kernel feature space by concatenating all mapped features.
    ///
    /// For a given row index and input matrix $x$, it computes the mapped feature for each feature $f$ and projection component $p$,
    /// and stores it at the corresponding layout offset $f \cdot m + p$:
    /// $$\text{out}[f \cdot m + p] = \left( \sum_{b=1}^m k(x_f, u_{f, b}) P_{f, bp} \right) - \mu_{f, p}$$
    /// where $m$ is `num_bases`, $k$ is the RBF kernel, $P_f$ is the projection matrix, and $\mu_f$ is the centering mean.
    pub fn transform_row_into(&self, x: MatRef<'_, f32>, r_idx: usize, mut out: ColMut<'_, f32>) {
        // Temporary buffer on the stack to store intermediate kernel calculations
        // Capped at MAX_BASES as per the Nystrom landmark selection logic
        let mut kernel_cache = [0.0f32; Self::MAX_BASES];
        for f_idx in 0..self.num_features {
            let x_val = x[(r_idx, f_idx)];
            let is_categorical_f = self.is_categorical[f_idx];

            let offset = f_idx * self.num_bases;

            // Handle missing values
            if x_val.is_nan() {
                for p_idx in 0..self.num_bases {
                    out[offset + p_idx] = 0.0;
                }
                continue;
            }

            let bases = &self.feature_bases[f_idx];
            let proj = &self.proj_matrices[f_idx];
            let mean = &self.feature_means[f_idx];
            let s2_inv = self.s2_invs[f_idx];

            // Pre-calculate RBF kernel distances between input and landmarks
            for b_idx in 0..self.num_bases {
                let diff = x_val - bases[b_idx];
                if is_categorical_f {
                    kernel_cache[b_idx] = if diff < 1e-5 { 1.0 } else { 0.0 };
                } else {
                    kernel_cache[b_idx] = (-(diff * diff) * s2_inv).exp();
                }
            }

            // Map into the learned Nystrom feature space via linear projection and centering
            for p_idx in 0..self.num_bases {
                let mut projection_sum = 0.0;
                for b_idx in 0..self.num_bases {
                    projection_sum += kernel_cache[b_idx] * proj[(b_idx, p_idx)];
                }
                out[offset + p_idx] = projection_sum - mean[p_idx];
            }
        }
    }

    /// Transforms a specific feature column across all rows into its Nystrom kernel feature space.
    ///
    /// For a given feature index $f$ and input matrix $x$, it computes the mapped feature
    /// for each projection component $p$ and stores it in the output matrix at position $(r, p)$:
    /// $$\text{out}[r, p] = \left( \sum_{b=1}^m k(x_{r, f}, u_{f, b}) P_{f, bp} \right) - \mu_{f, p}$$
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
        let is_categorical_f = self.is_categorical[f_idx];

        for r_idx in 0..n_samples {
            let x_val = x[(r_idx, f_idx)];

            // Handle missing values.
            if x_val.is_nan() {
                for p_idx in 0..self.num_bases {
                    out[(r_idx, p_idx)] = 0.0;
                }
                continue;
            }

            // Temporary buffer on the stack to store intermediate kernel calculations
            // Capped at MAX_BASES as per the Nystrom landmark selection logic
            let mut kernel_cache = [0.0f32; Self::MAX_BASES];

            // Pre-calculate RBF kernel distances between input and landmarks
            for b_idx in 0..self.num_bases {
                let diff = x_val - bases[b_idx];
                if is_categorical_f {
                    kernel_cache[b_idx] = if diff < 1e-5 { 1.0 } else { 0.0 };
                } else {
                    kernel_cache[b_idx] = (-(diff * diff) * s2_inv).exp();
                }
            }

            // Map into the learned Nystrom feature space via linear projection and centering
            for p_idx in 0..self.num_bases {
                let mut projection_sum = 0.0;
                for b_idx in 0..self.num_bases {
                    projection_sum += kernel_cache[b_idx] * proj[(b_idx, p_idx)];
                }
                out[(r_idx, p_idx)] = projection_sum - mean[p_idx];
            }
        }
    }
}
