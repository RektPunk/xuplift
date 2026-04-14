use faer::{Col, Mat};

use xuplift::metalearners::rlearner::RLearner;

#[test]
fn test_rlearner() {
    let n_samples = 1000;
    let n_features = 3;

    let mut x = Mat::<f32>::zeros(n_samples, n_features);
    let mut t = Col::<f32>::zeros(n_samples);
    let mut y = Col::<f32>::zeros(n_samples);

    // 1. Synthetic Data Generation (Robinson's Transformation Scenario)
    // Baseline (m(x)): 1.5 * x0 + sin(x1)
    // Propensity (e(x)): 0.5 (Random assignment)
    // Ground Truth Uplift (tau): 3.0 (Constant for simplicity)
    for i in 0..n_samples {
        let x0 = i as f32 * 0.01;
        let x1 = (i as f32 * 0.1).sin();
        let x2 = (i as f32 * 0.1).cos();

        x[(i, 0)] = x0;
        x[(i, 1)] = x1;
        x[(i, 2)] = x2;

        // Random treatment assignment (50/50)
        let treatment = if i % 2 == 0 { 1.0 } else { 0.0 };
        t[i] = treatment;

        // Outcome = Baseline + (Treatment * Effect) + Noise
        // y = 1.5*x0 + 0.5*sin(x1) + 5.0*t
        y[i] = 1.5 * x0 + 0.5 * x1.sin() + (5.0 * treatment) + 10.0;
    }

    // 2. Model Training
    // R-Learner trains: m(x) [Outcome], e(x) [Propensity], and tau(x) [Residual-on-Residual]
    let rlearner = RLearner::new(&x, &t, &y);

    // 3. Uplift Estimation
    // In R-Learner, the tau model directly estimates the treatment effect.
    let uplift_estimate = rlearner.predict_uplift(&x);

    // 4. Accuracy Verification
    let mut sum_uplift = 0.0;
    for i in 0..n_samples {
        sum_uplift += uplift_estimate[i];
    }
    let avg_uplift = sum_uplift / n_samples as f32;

    println!(
        "True Uplift: 5.0, R-Learner Estimated Avg Uplift: {:.4}",
        avg_uplift
    );

    // Verify if the average estimated uplift is close to 3.0
    assert!(
        (avg_uplift - 5.0).abs() < 0.5,
        "R-Learner estimation is too far from ground truth. Got: {:.4}",
        avg_uplift
    );

    // 5. Mathematical Explanation Consistency Check
    // For R-Learner, the explanation logic is simpler because it's a single Stage-2 model.
    // Predict(x) should be approximately (sum of feature contributions + base_value).
    let uplift_explanation = rlearner.explain_uplift(&x);

    for i in 0..x.nrows() {
        let mut explained_total = 0.0;
        for j in 0..uplift_explanation.ncols() {
            explained_total += uplift_explanation[(i, j)];
        }

        // R-Learner's prediction is derived from its internal tau regressor.
        let reconstructed_uplift = explained_total + rlearner.tau.base_value;

        assert!(
            (reconstructed_uplift - uplift_estimate[i]).abs() < 1e-4,
            "R-Learner explanation mismatch at sample {}: Explained {:.4}, Predicted {:.4}",
            i,
            reconstructed_uplift,
            uplift_estimate[i]
        );
    }
    println!("RLearner Residual Explanation check passed!");
}
