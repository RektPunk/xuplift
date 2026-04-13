use faer::{Col, Mat};

use xuplift::metalearners::xlearner::XLearner;

#[test]
fn test_xlearner_imbalanced_uplift() {
    let n_samples = 400;
    let n_features = 3;

    let mut x = Mat::<f32>::zeros(n_samples, n_features);
    let mut t = Col::<f32>::zeros(n_samples);
    let mut y = Col::<f32>::zeros(n_samples);

    // 1. Synthetic Data Generation with Imbalance
    // Objective: Simulate a scenario where Treatment (T=1) is rare.
    // Generative Model: y = 2.0*x0 + (5.0 * t) + 10.0
    // Ground Truth Uplift: 5.0
    for i in 0..n_samples {
        let x0 = i as f32 * 0.02;
        let x1 = (i as f32 * 0.1).cos();
        let x2 = (i as f32 * 0.1).sin();

        x[(i, 0)] = x0;
        x[(i, 1)] = x1;
        x[(i, 2)] = x2;

        // Intentional Imbalance: Only 20% receive treatment
        let treatment = if i % 5 == 0 { 1.0 } else { 0.0 };
        t[i] = treatment;

        // Outcome with a constant treatment effect of 5.0
        y[i] = 1.5 * x0 + 0.5 * x1 + (5.0 * treatment) + 10.0;
    }

    // 2. Model Training
    // X-Learner internally trains 5 models:
    // Stage 1: mu_1, mu_0 | Stage 2: tau_1, tau_0 | Stage 3: p (propensity)
    let xlearner = XLearner::new(&x, &t, &y);

    // 3. Uplift Estimation
    // The estimate uses the weighted average: g(x)*tau_0 + (1-g(x))*tau_1
    let uplift_estimate = xlearner.predict_uplift(&x);

    // 4. Accuracy Verification
    let mut sum_uplift = 0.0;
    for i in 0..n_samples {
        sum_uplift += uplift_estimate[i];
    }
    let avg_uplift = sum_uplift / n_samples as f32;

    println!(
        "True Uplift: 5.0, X-Learner Estimated Avg Uplift: {:.4}",
        avg_uplift
    );

    // X-Learner should handle the 20/80 imbalance well
    assert!(
        (avg_uplift - 5.0).abs() < 0.2,
        "Uplift estimation error too high. Got: {:.4}",
        avg_uplift
    );

    // 5. Mathematical Explanation Consistency Check
    // In X-Learner, the explanation must account for the dynamic base value
    // caused by the propensity-weighted blending of two models.
    let uplift_explanation = xlearner.explain_uplift(&x);
    let propensity = xlearner.p.predict(&x);

    for i in 0..x.nrows() {
        let mut feature_contribution_sum = 0.0;
        for j in 0..n_features {
            feature_contribution_sum += uplift_explanation[(i, j)];
        }

        // Calculate Dynamic Base Value: g(x)*base_tau0 + (1-g(x))*base_tau1
        let gi = propensity[i].clamp(0.01, 0.99);
        let dynamic_base = gi * xlearner.tau_0.base_value + (1.0 - gi) * xlearner.tau_1.base_value;

        let reconstructed_uplift = feature_contribution_sum + dynamic_base;

        // The sum of weighted contributions + weighted base must equal the prediction
        assert!(
            (reconstructed_uplift - uplift_estimate[i]).abs() < 1e-4,
            "X-Learner explanation mismatch at sample {}: Explained {:.4}, Predicted {:.4}",
            i,
            reconstructed_uplift,
            uplift_estimate[i]
        );
    }
    println!("XLearner Uplift Delta Explanation check passed!");
}
