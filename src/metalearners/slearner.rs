use faer::{Col, ColRef, Mat, MatRef};

use crate::xmodels::regressor::Regressor;

/// S-Learner (Single Learner) for Uplift Modeling.
///
/// This learner treats the treatment assignment $T$ as an additional feature in a response surface model:
/// $$\mu(x, t) = E[Y | X=x, T=t]$$
/// The uplift is estimated as:
/// $$\tau(x) = \mu(x, 1) - \mu(x, 0)$$
///
/// # Reference
/// * Künzel, S. R., Sekhon, J. S., Bickel, P. J., & Yu, B. (2019). Metalearners for estimating heterogeneous treatment effects using machine learning. Proceedings of the National Academy of Sciences, 116(10), 4156–4165. https://doi.org/10.1073/pnas.1804597116
pub struct SLearner {
    /// Regressor fitted on combined features (X, T).
    pub mu: Regressor,
}

impl SLearner {
    /// Initializes and fits the SLearner using the provided data.
    ///
    /// # Arguments
    /// * `x` - Feature matrix (n_samples x n_features).
    /// * `t` - Treatment vector (n_samples).
    /// * `y` - Outcome vector (n_samples).
    /// * `mu_penalty` - Regularization penalty for the outcome model.
    pub fn new(
        x: MatRef<'_, f32>,
        t: ColRef<'_, f32>,
        y: ColRef<'_, f32>,
        mu_penalty: f32,
    ) -> Self {
        let num_rows = x.nrows();
        let num_cols = x.ncols();
        let mut x_combined = Mat::<f32>::zeros(num_rows, num_cols + 1);

        // Copy features X into the combined matrix
        x_combined
            .as_mut()
            .submatrix_mut(0, 0, num_rows, num_cols)
            .copy_from(x);

        // Append treatment vector T
        x_combined.as_mut().col_mut(num_cols).copy_from(t);

        // Fit Regressor on combined features (X, T)
        let mut mu = Regressor::new(mu_penalty);
        mu.fit(x_combined.as_ref(), y);

        Self { mu }
    }

    /// Estimates the uplift score $\tau(x)$ for the given features.
    pub fn predict_uplift(&self, x: MatRef<'_, f32>) -> Col<f32> {
        let num_rows = x.nrows();
        let num_cols = x.ncols();

        let (mu_t1_pred, mu_t0_pred) = rayon::join(
            || {
                let mut scratch = Mat::<f32>::zeros(num_rows, num_cols + 1);
                scratch
                    .as_mut()
                    .submatrix_mut(0, 0, num_rows, num_cols)
                    .copy_from(x);
                scratch.as_mut().col_mut(num_cols).fill(1.0);
                self.mu.predict(scratch.as_ref())
            },
            || {
                let mut scratch = Mat::<f32>::zeros(num_rows, num_cols + 1);
                scratch
                    .as_mut()
                    .submatrix_mut(0, 0, num_rows, num_cols)
                    .copy_from(x);
                scratch.as_mut().col_mut(num_cols).fill(0.0);
                self.mu.predict(scratch.as_ref())
            },
        );
        mu_t1_pred - mu_t0_pred
    }

    /// Explains the uplift by decomposing the feature contributions.
    ///
    /// Return a matrix of dimensions (n_samples x (n_features + 1)),
    /// where the last column represents the direct effect of the treatment variable itself.
    pub fn explain_uplift(&self, x: MatRef<'_, f32>) -> Mat<f32> {
        let num_rows = x.nrows();
        let num_cols = x.ncols();

        let (exp_t1, exp_t0) = rayon::join(
            || {
                let mut x_t1 = Mat::<f32>::zeros(num_rows, num_cols + 1);
                x_t1.as_mut()
                    .submatrix_mut(0, 0, num_rows, num_cols)
                    .copy_from(x);
                x_t1.as_mut().col_mut(num_cols).fill(1.0);
                self.mu.explain(x_t1.as_ref())
            },
            || {
                let mut x_t0 = Mat::<f32>::zeros(num_rows, num_cols + 1);
                x_t0.as_mut()
                    .submatrix_mut(0, 0, num_rows, num_cols)
                    .copy_from(x);
                x_t0.as_mut().col_mut(num_cols).fill(0.0);
                self.mu.explain(x_t0.as_ref())
            },
        );

        // The difference reveals the source of the treatment effect
        exp_t1 - exp_t0
    }
}
