use faer::{Col, Mat};

use xuplift::metalearners::pwlearner::PWRegressor;

#[test]
fn test_pwregressor() {
    let n_samples = 500;
    let n_features = 3;

    let mut x = Mat::<f32>::zeros(n_samples, n_features);
    let mut t = Col::<f32>::zeros(n_samples);
    let mut y = Col::<f32>::zeros(n_samples);

    // --- Synthetic Data Generation ---
    // Objective: Create a dataset with a known constant treatment effect.
    // Generative Model: y = 1.5*x0 + 0.5*sin(x1) + (5.0 * t) + 10.0
    // Ground Truth Uplift (ITE): 5.0
    for r_idx in 0..n_samples {
        let x0 = r_idx as f32 * 0.01;
        let x1 = (r_idx as f32).sin();
        let x2 = (r_idx as f32).cos();

        x[(r_idx, 0)] = x0;
        x[(r_idx, 1)] = x1;
        x[(r_idx, 2)] = x2;

        // Assign treatment: Even indices = Treatment (1), Odd indices = Control (0)
        let treatment = if r_idx % 2 == 0 { 1.0 } else { 0.0 };
        t[r_idx] = treatment;
        y[r_idx] = 1.5 * x0 + 0.5 * x1.sin() + (5.0 * treatment) + 10.0;
    }

    // --- Model Initialization ---
    // PWRegressor fits a propensity classifier and fits a tau model on the pseudo-outcomes.
    let is_categorical = vec![false; n_features];
    let pwregressor = PWRegressor::new(
        x.as_ref(),
        t.as_ref(),
        y.as_ref(),
        &is_categorical,
        64,
        0.5,
        10,
        0.5,
    );

    // --- Prediction ---
    let uplift_estimate = pwregressor.predict_uplift(x.as_ref());

    // --- Verification: Accuracy ---
    let mut sum_uplift = 0.0;
    for r_idx in 0..n_samples {
        sum_uplift += uplift_estimate[r_idx];
    }
    let avg_uplift = sum_uplift / n_samples as f32;

    println!(
        "True Uplift: 5.0, PWRegressor Estimated Average Uplift: {:.4}",
        avg_uplift
    );

    assert!(
        (avg_uplift - 5.0).abs() < 0.5,
        "Uplift estimation is too far from ground truth. Got: {:.4}",
        avg_uplift
    );

    // --- Verification: Explanation Consistency ---
    // In PWRegressor, the explanation is straightforward because it uses a tau regressor.
    let uplift_explanation = pwregressor.explain_uplift(x.as_ref());

    // PWRegressor's explanation matrix should have n_features columns.
    assert_eq!(uplift_explanation.ncols(), n_features);

    let base_value = pwregressor.tau.base_value;
    for r_idx in 0..x.nrows() {
        let mut explained_total = 0.0;
        for p_idx in 0..uplift_explanation.ncols() {
            explained_total += uplift_explanation[(r_idx, p_idx)];
        }

        // Reconstructed Uplift = sum(feature_contributions) + base_value
        let total_reconstructed_uplift = explained_total + base_value;
        assert!(
            (total_reconstructed_uplift - uplift_estimate[r_idx]).abs() < 1e-4,
            "Uplift explanation delta mismatch at sample {}: Explained {:.4}, Predicted {:.4}",
            r_idx,
            total_reconstructed_uplift,
            uplift_estimate[r_idx]
        );
    }
    println!("PWLearner verification passed!");
}
