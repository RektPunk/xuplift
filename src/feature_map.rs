use faer::{Col, Mat, MatRef};
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
    pub feature_bases: Vec<Mat<f32>>,

    /// Learned projection matrices to map data into the kernel space.
    /// The projection matrix is $P = U \Lambda^{-1/2}$, where $U$ and $\Lambda$
    /// are the eigenvectors and eigenvalues of $K_{mm}$.
    pub proj_matrices: Vec<Mat<f32>>,
    /// Column-wise means of the transformed features for centering.
    /// $\mu_j = \frac{1}{n} \sum_{i=1}^n z_{ij}$
    pub feature_means: Vec<Col<f32>>,
    /// Inverse of the kernel bandwidth parameter (gamma) for each feature.
    /// Calculated using the Median Heuristic: $\gamma = \frac{1}{2\sigma^2}$
    pub s2_invs: Vec<f32>,
}

impl KernelFeatureMap {
    /// Returns a new Transformer instance.
    pub fn new() -> Self {
        Self {
            num_features: 0,
            num_bases: 0,
            feature_bases: Vec::new(),
            proj_matrices: Vec::new(),
            feature_means: Vec::new(),
            s2_invs: Vec::new(),
        }
    }

    /// Fits the transformer to the input data X.
    pub fn fit(&mut self, x: &MatRef<f32>) {
        let num_rows = x.nrows();
        self.num_features = x.ncols();

        // Calculate raw feature means (skipping NaNs) to use for imputation during landmark selection
        let raw_feature_means: Vec<f32> = (0..self.num_features)
            .into_par_iter()
            .map(|f_idx| {
                let col = x.col(f_idx);
                let mut sum = 0.0;
                let mut count = 0;
                for i in 0..num_rows {
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
        let valid_row_indices: Vec<usize> = (0..num_rows)
            .into_par_iter()
            .filter(|&r_idx| (0..self.num_features).all(|f_idx| !x[(r_idx, f_idx)].is_nan()))
            .collect();
        let n_valid = valid_row_indices.len();

        // Set the number of basis functions (landmarks).
        // Defaults to min(N, 64) for efficiency.
        let landmark_indices = if n_valid >= 32 {
            self.num_bases = n_valid.min(64);
            let mut rng = rng();
            let mut indices = valid_row_indices.clone();
            indices.shuffle(&mut rng);
            indices[..self.num_bases].to_vec()
        } else {
            self.num_bases = num_rows.min(64);
            let mut all_indices: Vec<usize> = (0..num_rows).collect();
            let mut rng = rng();
            all_indices.shuffle(&mut rng);
            all_indices[..self.num_bases].to_vec()
        };

        let feature_params: Vec<_> = (0..self.num_features)
            .into_par_iter()
            .map(|f_idx| {
                let f_mean = raw_feature_means[f_idx];

                // Median Heuristic: sets $\sigma$ to the median of pairwise distances between landmarks.
                // This provides a data-dependent bandwidth for the RBF kernel.
                let mut dists = Vec::with_capacity(self.num_bases * self.num_bases / 2);
                for i in 0..self.num_bases {
                    let mut val_i = x[(landmark_indices[i], f_idx)];
                    if val_i.is_nan() {
                        val_i = f_mean;
                    }
                    for j in i + 1..self.num_bases {
                        let mut val_j = x[(landmark_indices[j], f_idx)];
                        if val_j.is_nan() {
                            val_j = f_mean;
                        }
                        dists.push((val_i - val_j).abs());
                    }
                }
                dists.sort_by(|a: &f32, b: &f32| a.total_cmp(b));
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
                // Add a small epsilon to median to prevent extremely high gamma
                let s2_inv = 1.0 / (2.0 * (median.max(1e-4)).powi(2));

                // Store landmark values (bases)
                let mut bases = Mat::<f32>::zeros(1, self.num_bases);
                for (j_idx, &row_idx) in landmark_indices.iter().enumerate() {
                    let val = x[(row_idx, f_idx)];
                    bases[(0, j_idx)] = if val.is_nan() { f_mean } else { val };
                }

                // Compute Landmark Kernel matrix $K_{mm}$
                // $k_{ij} = \exp(-\gamma ||u_i - u_j||^2)$
                let mut k_mm = Mat::<f32>::zeros(self.num_bases, self.num_bases);
                for i in 0..self.num_bases {
                    for j in i..self.num_bases {
                        let diff = bases[(0, i)] - bases[(0, j)];
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
                for i in 0..num_rows {
                    let x_val = x[(i, f_idx)];
                    if !x_val.is_nan() {
                        for j in 0..self.num_bases {
                            let diff = x_val - bases[(0, j)];
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
                    z_col_means[l] = sum / num_rows as f32;
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

    /// Transforms a single feature for a given input value.
    ///
    /// Computes $z_l = \left( \sum_{j=1}^m k(x, u_j) P_{jl} \right) - \mu_l$.
    pub fn transform_feature_row(&self, f_idx: usize, x_val: f32) -> Col<f32> {
        let mut z = Col::<f32>::zeros(self.num_bases);
        if !x_val.is_nan() {
            let bases = &self.feature_bases[f_idx];
            let proj = &self.proj_matrices[f_idx];
            let mean = &self.feature_means[f_idx];
            let s2_inv = self.s2_invs[f_idx];

            let mut k_row = Vec::with_capacity(self.num_bases);
            for j in 0..self.num_bases {
                let diff = x_val - bases[(0, j)];
                k_row.push((-(diff * diff) * s2_inv).exp());
            }

            for l in 0..self.num_bases {
                let mut sum = 0.0;
                for j in 0..self.num_bases {
                    sum += k_row[j] * proj[(j, l)];
                }
                z[l] = sum - mean[l];
            }
        }
        z
    }

    /// Transforms an entire row into the kernel feature space.
    ///
    /// Concatenates the transformed features: $Z = [Z_1, Z_2, \dots, Z_d]$.
    pub fn transform_row(&self, x: &MatRef<f32>, row_idx: usize) -> Col<f32> {
        let total_dim = self.num_features * self.num_bases;
        let mut z_row = Col::<f32>::zeros(total_dim);
        for f_idx in 0..self.num_features {
            let x_val = x[(row_idx, f_idx)];
            let z_f = self.transform_feature_row(f_idx, x_val);
            let offset = f_idx * self.num_bases;
            for k in 0..self.num_bases {
                z_row[offset + k] = z_f[k];
            }
        }
        z_row
    }

    /// Transforms a new input matrix X into the learned Nystrom feature space.
    ///
    /// Returns a vector of matrices, one for each feature.
    pub fn transform(&self, x: &MatRef<f32>) -> Vec<Mat<f32>> {
        let n_samples = x.nrows();
        let n_features = x.ncols();
        (0..n_features)
            .into_par_iter()
            .map(|f_idx| {
                let mut k_batch = Mat::<f32>::zeros(n_samples, self.num_bases);
                let bases = &self.feature_bases[f_idx];
                let proj = &self.proj_matrices[f_idx];
                let mean = &self.feature_means[f_idx];
                let s2_inv = self.s2_invs[f_idx];

                for i in 0..n_samples {
                    let x_val = x[(i, f_idx)];
                    if !x_val.is_nan() {
                        for j in 0..self.num_bases {
                            let diff = x_val - bases[(0, j)];
                            k_batch[(i, j)] = (-(diff * diff) * s2_inv).exp();
                        }
                    }
                }

                let mut z_batch = k_batch * proj;
                for i in 0..n_samples {
                    if x[(i, f_idx)].is_nan() {
                        for j in 0..self.num_bases {
                            z_batch[(i, j)] = 0.0;
                        }
                    } else {
                        for j in 0..self.num_bases {
                            z_batch[(i, j)] -= mean[j];
                        }
                    }
                }
                z_batch
            })
            .collect()
    }
}
