use faer::{Col, ColRef, Mat, MatRef};

use crate::xmodels::classifier::Classifier;
use crate::xmodels::regressor::Regressor;

/// Doubly Robust (DR) Learner for Uplift Modeling.
///
/// This learner uses a multi-stage process with a doubly robust score wrapper:
/// 1. Train baseline outcome models $\mu_1(x)$ and $\mu_0(x)$ on treatment/control groups.
/// 2. Train a propensity model $e(x) = E[T|X]$ to estimate treatment assignment probabilities.
/// 3. Construct a doubly robust pseudo-outcome ($Y_{dr}$):
///    $$Y_{dr} = \mu_1(x) - \mu_0(x) + \frac{T(Y - \mu_1(x))}{e(x)} - \frac{(1 - T)(Y - \mu_0(x))}{1 - e(x)}$$
/// 4. Train a single final model $\tau(x)$ on the full feature matrix to predict $Y_{dr}$.
pub struct DRLearner {
    /// Final treatment effect model trained on doubly robust pseudo-outcomes
    pub tau: Regressor,
}

impl DRLearner {
    /// Initializes and fits the DRLearner using the provided data.
    ///
    /// # Arguments
    /// * `x` - The original feature matrix (n_samples x n_features).
    /// * `t` - The treatment assignment vector (n_samples, 0 or 1).
    /// * `y` - The observed outcome vector.
    /// * `mu_penalty` - The regularization penalty for the base outcome regressors.
    /// * `p_penalty` - The regularization penalty for the propensity classifier.
    /// * `p_max_iter` - The maximum number of iterations for the propensity classifier.
    /// * `tau_penalty` - The regularization penalty for the final treatment effect model.
    pub fn new(
        x: MatRef<'_, f32>,
        t: ColRef<'_, f32>,
        y: ColRef<'_, f32>,
        mu_penalty: f32,
        p_penalty: f32,
        p_max_iter: usize,
        tau_penalty: f32,
    ) -> Self {
        let num_rows = x.nrows();

        // Create weights for T=1 and T=0
        let w_t1 = Col::<f32>::from_fn(num_rows, |i| if t[i] > 0.5 { 1.0 } else { 0.0 });
        let w_t0 = Col::<f32>::from_fn(num_rows, |i| if t[i] <= 0.5 { 1.0 } else { 0.0 });

        // Train base outcome models and propensity model concurrently
        let ((mu_1, mu_0), p) = rayon::join(
            || {
                rayon::join(
                    || {
                        let mut mu_1 = Regressor::new(mu_penalty);
                        mu_1.fit_weighted(x, y, &w_t1);
                        mu_1
                    },
                    || {
                        let mut mu_0 = Regressor::new(mu_penalty);
                        mu_0.fit_weighted(x, y, &w_t0);
                        mu_0
                    },
                )
            },
            || {
                let mut p = Classifier::new(p_penalty, p_max_iter);
                p.fit(x, t);
                p
            },
        );

        // Predict intermediate components in parallel
        let (p_pred, (mu_1_pred, mu_0_pred)) = rayon::join(
            || p.predict(x),
            || rayon::join(|| mu_1.predict(x), || mu_0.predict(x)),
        );

        // Construct the Doubly Robust Pseudo-Outcomes
        let y_dr = Col::<f32>::from_fn(num_rows, |i| {
            let gi = p_pred[i].clamp(0.01, 0.99);
            let mu_1_i = mu_1_pred[i];
            let mu_0_i = mu_0_pred[i];

            let base_effect = mu_1_i - mu_0_i;
            if t[i] > 0.5 {
                base_effect + (y[i] - mu_1_i) / gi
            } else {
                base_effect - (y[i] - mu_0_i) / (1.0 - gi)
            }
        });

        // Train the final single tau model on the DR target
        let mut tau = Regressor::new(tau_penalty);
        tau.fit(x, y_dr.as_ref());

        Self { tau }
    }

    /// Estimates the uplift score: $\hat{\tau}(x)$ directly using the single DR model.
    pub fn predict_uplift(&self, x: MatRef<'_, f32>) -> Col<f32> {
        self.tau.predict(x)
    }

    /// Explains the uplift by decomposing the feature contributions of the DR tau model.
    ///
    /// Since the DR pseudo-outcome effectively isolates the unbiased treatment effect
    /// into a single target variable, the feature attributions can be directly derived
    /// from the final model without complex weighting or blending.
    ///
    /// # Returns
    /// A matrix (n_samples x n_features) representing the unblended contribution of
    /// each feature to the final estimated Treatment Effect.
    pub fn explain_uplift(&self, x: MatRef<'_, f32>) -> Mat<f32> {
        self.tau.explain(x)
    }
}
