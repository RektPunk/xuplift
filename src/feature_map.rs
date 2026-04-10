use faer::{Col, Mat};
use rand::rng;
use rand::seq::SliceRandom;
use rayon::prelude::*;

/// A transformer that approximates kernel feature maps using the Nystrom method.
///
/// It maps input data into a finite-dimensional feature space where linear
/// operations approximate non-linear kernels (e.g., RBF kernel).
pub struct KernelFeatureMap {
    // Learned Parameters
    /// Total number of rows in the training data.
    pub num_rows: usize,
    /// Number of input features (columns).
    pub num_features: usize,
    /// Number of landmark points (basis functions) per feature.
    pub num_bases: usize,
    /// Selected landmark samples from the training set.
    pub feature_bases: Vec<Mat<f32>>,

    /// Learned projection matrices to map data into the kernel space.
    pub proj_matrices: Vec<Mat<f32>>,
    /// Column-wise means of the transformed features for centering.
    pub feature_means: Vec<Col<f32>>,
    /// Pre-computed transformed features for the training set.
    pub z_matrices: Vec<Mat<f32>>,
    /// Inverse of the kernel bandwidth parameter (gamma) for each feature.
    pub s2_invs: Vec<f32>,
}

impl KernelFeatureMap {
    /// Returns a new Transformer instance.
    pub fn new() -> Self {
        Self {
            num_rows: 0,
            num_features: 0,
            num_bases: 0,
            feature_bases: Vec::new(),
            proj_matrices: Vec::new(),
            feature_means: Vec::new(),
            z_matrices: Vec::new(),
            s2_invs: Vec::new(),
        }
    }

    /// Fits the transformer to the input data X.
    pub fn fit(&mut self, x: &Mat<f32>) {
        self.num_rows = x.nrows();
        self.num_features = x.ncols();

        // Ensure every column has at least one valid (non-NaN) value
        let valid_row_indices: Vec<usize> = (0..self.num_rows)
            .into_par_iter()
            .filter(|&r_idx| (0..self.num_features).all(|f_idx| !x[(r_idx, f_idx)].is_nan()))
            .collect();

        let n_valid = valid_row_indices.len();

        if n_valid == 0 {
            panic!("Feature columns must not be empty or contain only NaNs.");
        }

        // Set the number of basis functions (clamped between 1 and 50)
        self.num_bases = n_valid.min(50);
        if n_valid < self.num_bases {
            self.num_bases = n_valid;
        }

        // Randomly select indices for Nystrom landmarks
        let mut rng = rng();
        let mut landmark_indices = valid_row_indices.clone();
        landmark_indices.shuffle(&mut rng);
        let landmark_indices = &landmark_indices[..self.num_bases];

        let feature_params: Vec<_> = (0..self.num_features)
            .into_par_iter()
            .map(|f_idx| {
                // Compute pairwise distances for s2_inv
                // Median Heuristic for Kernel Bandwidth
                let mut dists = Vec::with_capacity(self.num_bases * self.num_bases / 2);
                for i in 0..self.num_bases {
                    let val_i = x[(landmark_indices[i], f_idx)];
                    for j in i + 1..self.num_bases {
                        let val_j = x[(landmark_indices[j], f_idx)];
                        dists.push((val_i - val_j).abs());
                    }
                }
                dists.sort_by(|a, b| a.partial_cmp(b).unwrap());
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
                // Precision parameter (1/2*sigma^2)
                let s2_inv = 1.0 / (2.0 * (median.max(1e-6)).powi(2));

                // Nystrom approximation
                // Store landmark values (bases)
                let mut bases = Mat::<f32>::zeros(1, self.num_bases);
                for (j_idx, &row_idx) in landmark_indices.iter().enumerate() {
                    bases[(0, j_idx)] = x[(row_idx, f_idx)];
                }

                // Compute Kernel matrix K_nm(X x Landmarks)
                let mut k_nm = Mat::<f32>::zeros(self.num_rows, self.num_bases);
                for i in 0..self.num_rows {
                    let x_val = x[(i, f_idx)];
                    if !x_val.is_nan() {
                        for j in 0..self.num_bases {
                            let diff = x_val - bases[(0, j)];
                            k_nm[(i, j)] = (-(diff * diff) * s2_inv).exp();
                        }
                    }
                }

                // Compute Landmark Kernel matrix K_mm(Landmarks x Landmarks)
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

                // Eigen-decomposition to find the symmetric inverse square root: K_mm^(-1/2) = U \Lambda^{-1/2} U^T
                let eig = k_mm.self_adjoint_eigen(faer::Side::Lower).unwrap();
                let mut inv_s = Mat::<f32>::zeros(self.num_bases, self.num_bases);
                for d in 0..self.num_bases {
                    let val = eig.S()[d];
                    inv_s[(d, d)] = if val > 1e-10 { 1.0 / val.sqrt() } else { 0.0 };
                }

                // Projection matrix maps raw kernel values to the orthonormal feature space
                let proj_matrix = eig.U() * &inv_s;
                let mut z_features = &k_nm * &proj_matrix;

                // Centering the features (subtracting the mean; ignoring NaNs)
                let mut z_col_means = Col::<f32>::zeros(self.num_bases);
                for j in 0..self.num_bases {
                    let mean_val = z_features.col(j).iter().sum::<f32>() / self.num_rows as f32;
                    z_col_means[j] = mean_val;
                    for i in 0..self.num_rows {
                        if !x[(i, f_idx)].is_nan() {
                            z_features[(i, j)] -= mean_val;
                        } else {
                            z_features[(i, j)] = 0.0;
                        }
                    }
                }
                (bases, proj_matrix, z_features, z_col_means, s2_inv)
            })
            .collect();

        // Transfer calculated parameters to the struct
        for (b, p, z, o, s) in feature_params {
            self.feature_bases.push(b);
            self.proj_matrices.push(p);
            self.z_matrices.push(z);
            self.feature_means.push(o);
            self.s2_invs.push(s);
        }
    }

    /// Transforms a new input matrix X into the learned Nystrom feature space.
    pub fn transform(&self, x: &Mat<f32>) -> Vec<Mat<f32>> {
        let n_samples = x.nrows();
        let n_features = x.ncols();
        (0..n_features)
            .into_par_iter()
            .map(|f_idx| {
                // Apply the RBF kernel and projection learned during for given X (N x num_bases)
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

                // Center the new data using the means stored from training
                for i in 0..n_samples {
                    if !x[(i, f_idx)].is_nan() {
                        for j in 0..self.num_bases {
                            z_batch[(i, j)] -= mean[j];
                        }
                    } else {
                        for j in 0..self.num_bases {
                            z_batch[(i, j)] = 0.0;
                        }
                    }
                }
                z_batch
            })
            .collect()
    }
}
