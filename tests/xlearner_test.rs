use faer::{Col, Mat};

use xuplift::metalearners::xlearner::XLearner;

#[test]
fn test_xlearner() {
    let n_samples = 500;
    let n_features = 3;

    let mut x = Mat::<f32>::zeros(n_samples, n_features);
    let mut t = Col::<f32>::zeros(n_samples);
    let mut y = Col::<f32>::zeros(n_samples);

    // Synthetic Data Generation with Imbalance
    // Objective: Simulate a scenario where Treatment (T=1) is rare.
    // Generative Model: y = 1.5*x0 + 0.5*sin(x1) + (5.0 * t) + 10.0
    // Ground Truth Uplift (ITE): 5.0
    for i in 0..n_samples {
        let x0 = i as f32 * 0.02;
        let x1 = (i as f32 * 0.1).cos();
        let x2 = (i as f32 * 0.1).sin();

        x[(i, 0)] = x0;
        x[(i, 1)] = x1;
        x[(i, 2)] = x2;

        // Intentional Imbalance: Only 20% receive treatment
        // X-Learner should handle the 20/80 imbalance well
        let treatment = if i % 5 == 0 { 1.0 } else { 0.0 };
        t[i] = treatment;

        // Outcome with a constant treatment effect of 5.0
        y[i] = 1.5 * x0 + 0.5 * x1.sin() + (5.0 * treatment) + 10.0;
    }

    // X-Learner internally trains 5 models:
    // Stage 1: mu_1, mu_0 | Stage 2: tau_1, tau_0 | Stage 3: p (propensity)
    let xlearner = XLearner::new(x.as_ref(), t.as_ref(), y.as_ref(), 0.1, 0.1, 20, 0.1);

    // Estimate Individual Treatment Effect (ITE): g(x)*tau_0 + (1-g(x))*tau_1
    let uplift_estimate = xlearner.predict_uplift(x.as_ref());

    // Verify if the average estimated uplift is close to the true effect.
    let mut sum_uplift = 0.0;
    for i in 0..n_samples {
        sum_uplift += uplift_estimate[i];
    }
    let avg_uplift = sum_uplift / n_samples as f32;

    println!(
        "True Uplift: 5.0, X-Learner Estimated Avg Uplift: {:.4}",
        avg_uplift
    );

    assert!(
        (avg_uplift - 5.0).abs() < 0.1,
        "Uplift estimation error too high. Got: {:.4}",
        avg_uplift
    );

    // Verify Mathematical Explanation Consistency
    // In X-Learner, the explanation must account for the dynamic base value
    // caused by the propensity-weighted blending of two models.
    let uplift_explanation = xlearner.explain_uplift(x.as_ref());

    // X-Learner's explanation matrix should have n_features columns.
    assert_eq!(uplift_explanation.ncols(), n_features);

    let propensity = xlearner.p.predict(x.as_ref());

    for i in 0..x.nrows() {
        let mut feature_contribution_sum = 0.0;
        for j in 0..n_features {
            feature_contribution_sum += uplift_explanation[(i, j)];
        }

        // Calculate Dynamic Base Value: g(x)*base_tau0 + (1-g(x))*base_tau1
        let gi = propensity[i].clamp(0.01, 0.99);
        let dynamic_base =
            gi * xlearner.tau_t0.base_value + (1.0 - gi) * xlearner.tau_t1.base_value;

        // The sum of weighted contributions + weighted base must equal the uplift estimate for each sample.
        let reconstructed_uplift = feature_contribution_sum + dynamic_base;
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
