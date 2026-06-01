use faer::{Col, Mat};

use xuplift::metalearners::slearner::SLearner;

#[test]
fn test_slearner() {
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
    let slearner = SLearner::new(x.as_ref(), t.as_ref(), y.as_ref(), 0.01);

    // --- Prediction ---
    let uplift_estimate = slearner.predict_uplift(x.as_ref());

    // --- Verification: Accuracy ---
    let mut sum_uplift = 0.0;
    for r_idx in 0..n_samples {
        sum_uplift += uplift_estimate[r_idx];
    }
    let avg_uplift = sum_uplift / n_samples as f32;

    println!(
        "True Uplift: 5.0, Estimated Average Uplift: {:.4}",
        avg_uplift
    );

    assert!(
        (avg_uplift - 5.0).abs() < 0.1,
        "Uplift estimation is too far from ground truth. Got: {:.4}",
        avg_uplift
    );

    // --- Verification: Explanation Consistency ---
    let uplift_explanation = slearner.explain_uplift(x.as_ref());

    // S-Learner's explanation matrix should have n_features + 1 columns (X + T).
    assert_eq!(uplift_explanation.ncols(), n_features + 1);

    for r_idx in 0..x.nrows() {
        let mut explained_total = 0.0;
        for p_idx in 0..uplift_explanation.ncols() {
            explained_total += uplift_explanation[(r_idx, p_idx)];
        }

        // For S-Learner, sum(contributions) should equal the predicted uplift.
        assert!(
            (explained_total - uplift_estimate[r_idx]).abs() < 1e-4,
            "Uplift explanation delta mismatch at sample {}: Explained {:.4}, Predicted {:.4}",
            r_idx,
            explained_total,
            uplift_estimate[r_idx]
        );
    }
    println!("SLearner verification passed!");
}
