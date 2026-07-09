use faer::{Col, Mat, MatMut, MatRef, Scale};
use rand::{RngExt, rng};
use rayon::prelude::*;

pub enum Kernel {
    Categorical,
    Rbf { s2_inv: f32 },
}

impl Kernel {
    #[inline]
    fn eval(&self, x: f32, y: f32) -> f32 {
        match self {
            Kernel::Categorical => {
                let diff = (x - y).abs();
                if diff < 1e-5 { 1.0 } else { 0.0 }
            }

            Kernel::Rbf { s2_inv } => {
                let diff = x - y;
                (-(diff * diff) * s2_inv).exp()
            }
        }
    }
}

/// A transformer that approximates kernel feature maps using the Nystr\"om method.
pub struct KernelFeatureMap {
    /// Number of input features (columns).
    pub num_features: usize,
    /// Number of landmark points (basis functions) per feature.
    pub num_bases: usize,
    /// Landmark values selected for each feature.
    /// Each entry contains the landmark points of a single feature.
    pub feature_bases: Vec<Col<f32>>,

    /// Projection matrices that map kernel evaluations into the feature space.
    /// Each matrix is computed as $P = U\Lambda^{-1/2}$, where
    /// $U$ and $\Lambda$ are the eigenvectors and eigenvalues of $K_{mm}$.
    pub proj_matrices: Vec<Mat<f32>>,
    /// Mean feature vectors used to center the transformed features.
    pub feature_means: Vec<Col<f32>>,
    /// Kernel associated with each feature.
    pub kernels: Vec<Kernel>,
}

impl KernelFeatureMap {
    pub fn new(max_bases: usize) -> Self {
        Self {
            num_features: 0,
            num_bases: max_bases,
            feature_bases: Vec::new(),
            proj_matrices: Vec::new(),
            feature_means: Vec::new(),
            kernels: Vec::new(),
        }
    }

    /// Fits the kernel feature map to the input matrix.
    pub fn fit(&mut self, x: MatRef<'_, f32>, is_categorical: &[bool]) {
        self.feature_bases.clear();
        self.proj_matrices.clear();
        self.feature_means.clear();
        self.kernels.clear();

        let n_samples = x.nrows();
        self.num_features = x.ncols();
        let max_dist_pairs = self.num_bases * (self.num_bases - 1) / 2;
        let feature_params: Vec<_> = (0..self.num_features)
            .into_par_iter()
            .map(|f_idx| {
                let x_col = x.col(f_idx);
                let mut rng = rng();
                let mut valid_indices: Vec<usize> =
                    (0..n_samples).filter(|&i| !x_col[i].is_nan()).collect();
                assert!(
                    valid_indices.len() >= self.num_bases,
                    "Not enough valid samples in feature {}. Expected at least {}, but got {}.",
                    f_idx,
                    self.num_bases,
                    valid_indices.len()
                );

                for i in 0..self.num_bases {
                    let j = rng.random_range(i..valid_indices.len());
                    valid_indices.swap(i, j);
                }
                let landmark_indices = &valid_indices[..self.num_bases];
                let is_categorical_f = is_categorical[f_idx];
                let kernel = if is_categorical_f {
                    Kernel::Categorical
                } else {
                    Kernel::Rbf {
                        s2_inv: {
                            // Estimate the RBF kernel precision parameter using the median heuristic.
                            let mut dists_buf = vec![0.0f32; max_dist_pairs];
                            let mut dists_count = 0;
                            for b_i_idx in 0..self.num_bases {
                                let val_i = x_col[landmark_indices[b_i_idx]];
                                for b_j_idx in b_i_idx + 1..self.num_bases {
                                    dists_buf[dists_count] =
                                        (val_i - x_col[landmark_indices[b_j_idx]]).abs();
                                    dists_count += 1;
                                }
                            }
                            let dists = &mut dists_buf[..dists_count];
                            dists.sort_by(|a, b| a.total_cmp(b));
                            let median = if !dists.is_empty() {
                                let mid = dists.len() / 2;
                                if dists.len().is_multiple_of(2) {
                                    (dists[mid] + dists[mid - 1]) * 0.5
                                } else {
                                    dists[mid]
                                }
                            } else {
                                1.0
                            };

                            1.0 / (2.0 * (median.max(1e-4)).powi(2))
                        },
                    }
                };

                // Store landmark values
                let mut basis = Col::<f32>::zeros(self.num_bases);
                for (b_idx, &r_idx) in landmark_indices.iter().enumerate() {
                    basis[b_idx] = x_col[r_idx];
                }

                // Compute the landmark kernel matrix K_mm
                let mut k_mm = Mat::<f32>::zeros(self.num_bases, self.num_bases);
                for b_i_idx in 0..self.num_bases {
                    for b_j_idx in b_i_idx..self.num_bases {
                        k_mm[(b_i_idx, b_j_idx)] = kernel.eval(basis[b_i_idx], basis[b_j_idx]);
                        if b_i_idx != b_j_idx {
                            k_mm[(b_j_idx, b_i_idx)] = k_mm[(b_i_idx, b_j_idx)];
                        }
                    }
                }

                // Compute the symmetric inverse square root of K_mm
                let eig = k_mm.self_adjoint_eigen(faer::Side::Lower).unwrap();
                let mut proj_matrix = eig.U().to_owned();

                for p_idx in 0..self.num_bases {
                    let val = eig.S()[p_idx];
                    let inv_sqrt_s = if val > 1e-5 { 1.0 / val.sqrt() } else { 0.0 };
                    let mut col = proj_matrix.col_mut(p_idx);
                    col *= Scale(inv_sqrt_s);
                }

                // Compute the feature-space mean without explicitly forming Z
                let mut k_col_sums = Col::<f32>::zeros(self.num_bases);
                for &r_idx in &valid_indices {
                    let x_val = x_col[r_idx];
                    for b_idx in 0..self.num_bases {
                        k_col_sums[b_idx] += kernel.eval(x_val, basis[b_idx]);
                    }
                }
                let scale = 1.0 / n_samples as f32;
                let z_col_means = (proj_matrix.transpose() * &k_col_sums) * Scale(scale);

                (basis, proj_matrix, z_col_means, kernel)
            })
            .collect();

        for (bases, proj, means, kernel) in feature_params {
            self.feature_bases.push(bases);
            self.proj_matrices.push(proj);
            self.feature_means.push(means);
            self.kernels.push(kernel);
        }
    }

    /// Transforms a single feature column into its Nystr\"om kernel feature space.
    pub fn transform_feature_into(
        &self,
        x: MatRef<'_, f32>,
        f_idx: usize,
        mut out: MatMut<'_, f32>,
    ) {
        let x_col = x.col(f_idx);
        let n_samples = x.nrows();
        let bases = &self.feature_bases[f_idx];
        let proj = &self.proj_matrices[f_idx];
        let mean = &self.feature_means[f_idx];
        let kernel = &self.kernels[f_idx];

        let mut k_f = Mat::<f32>::zeros(n_samples, self.num_bases);
        for r_idx in 0..n_samples {
            let x_val = x_col[r_idx];
            if !x_val.is_nan() {
                for b_idx in 0..self.num_bases {
                    k_f[(r_idx, b_idx)] = kernel.eval(x_val, bases[b_idx]);
                }
            }
        }
        out.copy_from(&(k_f.as_ref() * proj.as_ref()));

        for p_idx in 0..self.num_bases {
            let m_val = mean[p_idx];
            for r_idx in 0..n_samples {
                if x_col[r_idx].is_nan() {
                    out[(r_idx, p_idx)] = 0.0;
                } else {
                    out[(r_idx, p_idx)] -= m_val;
                }
            }
        }
    }

    /// Transforms the input matrix into the concatenated feature space.
    pub fn transform(&self, x: MatRef<'_, f32>) -> Mat<f32> {
        let n_samples = x.nrows();
        let total_dim = self.num_features * self.num_bases;
        let mut z = Mat::<f32>::zeros(n_samples, total_dim);

        z.as_mut()
            .par_col_partition_mut(self.num_features)
            .enumerate()
            .for_each(|(f_idx, mut z_f): (usize, faer::MatMut<'_, f32>)| {
                self.transform_feature_into(x, f_idx, z_f.as_mut());
            });

        z
    }
}
