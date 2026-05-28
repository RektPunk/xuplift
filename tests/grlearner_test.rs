use faer::{Col, Mat};

use xuplift::metalearners::grlearner::GRLearner;

#[test]
fn test_grlearner_continuous() {
    let n_samples = 500;
    let n_features = 3;

    let mut x = Mat::<f32>::zeros(n_samples, n_features);
    let mut t = Col::<f32>::zeros(n_samples);
    let mut y = Col::<f32>::zeros(n_samples);

    // --- Synthetic Data Generation ---
    // Objective: Create a dataset with continuous treatment and known confounding.
    // Generative Model:
    // t = 2.0 * x0 + Noise_T
    // y = 1.5 * x0 + 0.5 * sin(x1) + (5.0 * t) + 10.0 + Noise_Y
    // Ground Truth Uplift (CATE Slope): 5.0
    for i in 0..n_samples {
        let x0 = i as f32 * 0.01;
        let x1 = (i as f32).sin();
        let x2 = (i as f32).cos();

        x[(i, 0)] = x0;
        x[(i, 1)] = x1;
        x[(i, 2)] = x2;

        // Continuous Treatment with confounding bias (dependent on x0)
        // Even/Odd patterns inject clean variation around the confounder line
        let treatment_noise = if i % 2 == 0 { 0.5 } else { -0.5 };
        let treatment = 2.0 * x0 + treatment_noise;
        t[i] = treatment;

        // Outcome y with a true treatment effect multiplier of 5.0
        let outcome_noise = if i % 3 == 0 { 0.1 } else { -0.1 };
        y[i] = 1.5 * x0 + 0.5 * x1.sin() + (5.0 * treatment) + 10.0 + outcome_noise;
    }

    // --- Model Initialization ---
    // GRLearner uses regressors for both outcome and treatment models to support continuous treatment.
    let grlearner = GRLearner::new(x.as_ref(), t.as_ref(), y.as_ref(), 0.001, 0.001, 0.001);

    // --- Prediction ---
    let uplift_estimate = grlearner.predict_uplift(x.as_ref());

    // --- Verification: Accuracy ---
    let mut sum_uplift = 0.0;
    for i in 0..n_samples {
        sum_uplift += uplift_estimate[i];
    }
    let avg_uplift = sum_uplift / n_samples as f32;

    println!(
        "True Uplift: 5.0, GRLearner Estimated Average Uplift: {:.4}",
        avg_uplift
    );

    assert!(
        (avg_uplift - 5.0).abs() < 0.1,
        "Uplift estimation is too far from ground truth. Got: {:.4}",
        avg_uplift
    );

    // --- Verification: Explanation Consistency ---
    let uplift_explanation = grlearner.explain_uplift(x.as_ref());

    // Regressor-based explanation matrix should have n_features columns.
    assert_eq!(uplift_explanation.ncols(), n_features);

    let base_value = grlearner.tau.base_value;
    for i in 0..x.nrows() {
        let mut explained_total = 0.0;
        for j in 0..uplift_explanation.ncols() {
            explained_total += uplift_explanation[(i, j)];
        }

        // Reconstructed Uplift = sum(feature_contributions) + base_value
        let total_reconstructed_uplift = explained_total + base_value;
        assert!(
            (total_reconstructed_uplift - uplift_estimate[i]).abs() < 1e-4,
            "Uplift explanation delta mismatch at sample {}: Explained {:.4}, Predicted {:.4}",
            i,
            total_reconstructed_uplift,
            uplift_estimate[i]
        );
    }
    println!("GRLearner verification passed!");
}
