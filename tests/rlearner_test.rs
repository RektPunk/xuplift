use faer::{Col, Mat};

use xuplift::metalearners::rlearner::RLearner;

#[test]
fn test_rlearner() {
    let n_samples = 500;
    let n_features = 3;

    let mut x = Mat::<f32>::zeros(n_samples, n_features);
    let mut t = Col::<f32>::zeros(n_samples);
    let mut y = Col::<f32>::zeros(n_samples);

    // --- Synthetic Data Generation ---
    // Objective: Create a dataset with a known constant treatment effect.
    // Generative Model: y = 1.5*x0 + 0.5*sin(x1) + (5.0 * t) + 10.0
    // Ground Truth Uplift (ITE): 5.0
    for i in 0..n_samples {
        let x0 = i as f32 * 0.01;
        let x1 = (i as f32 * 0.1).sin();
        let x2 = (i as f32 * 0.1).cos();

        x[(i, 0)] = x0;
        x[(i, 1)] = x1;
        x[(i, 2)] = x2;

        // Random treatment assignment
        let treatment = if i % 2 == 0 { 1.0 } else { 0.0 };
        t[i] = treatment;

        // Outcome = Baseline + (Treatment * Effect) + Noise
        y[i] = 1.5 * x0 + 0.5 * x1.sin() + (5.0 * treatment) + 10.0;
    }

    // --- Model Initialization ---
    // R-Learner trains: m(x) [Outcome], e(x) [Propensity], and tau(x) [Residual-on-Residual]
    let rlearner = RLearner::new(x.as_ref(), t.as_ref(), y.as_ref(), 0.1, 0.1, 20, 0.1);

    // --- Prediction ---
    let uplift_estimate = rlearner.predict_uplift(x.as_ref());

    // --- Verification: Accuracy ---
    let mut sum_uplift = 0.0;
    for i in 0..n_samples {
        sum_uplift += uplift_estimate[i];
    }
    let avg_uplift = sum_uplift / n_samples as f32;

    println!(
        "True Uplift: 5.0, R-Learner Estimated Avg Uplift: {:.4}",
        avg_uplift
    );

    assert!(
        (avg_uplift - 5.0).abs() < 0.1,
        "R-Learner estimation is too far from ground truth. Got: {:.4}",
        avg_uplift
    );

    // --- Verification: Explanation Consistency ---
    let uplift_explanation = rlearner.explain_uplift(x.as_ref());

    // R-Learner's explanation matrix should have n_features columns.
    assert_eq!(uplift_explanation.ncols(), n_features);

    for i in 0..x.nrows() {
        let mut explained_total = 0.0;
        for j in 0..uplift_explanation.ncols() {
            explained_total += uplift_explanation[(i, j)];
        }

        // Reconstructed Uplift = sum(feature_contributions) + base_value
        let reconstructed_uplift = explained_total + rlearner.tau.base_value;
        assert!(
            (reconstructed_uplift - uplift_estimate[i]).abs() < 1e-4,
            "R-Learner explanation mismatch at sample {}: Explained {:.4}, Predicted {:.4}",
            i,
            reconstructed_uplift,
            uplift_estimate[i]
        );
    }
    println!("RLearner verification passed!");
}
