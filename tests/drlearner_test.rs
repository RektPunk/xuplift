use faer::{Col, Mat};

use xuplift::metalearners::drlearner::DRLearner;

#[test]
fn test_drlearner() {
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
        // DR-Learner should handle the 20/80 imbalance well
        let treatment = if i % 5 == 0 { 1.0 } else { 0.0 };
        t[i] = treatment;

        // Outcome with a constant treatment effect of 5.0
        y[i] = 1.5 * x0 + 0.5 * x1.sin() + (5.0 * treatment) + 10.0;
    }

    // DR-Learner internally trains 4 models:
    // Stage 1: mu_1, mu_0 (base outcomes) | Stage 2: p (propensity score)
    // Stage 3: Construct pseudo-outcomes | Stage 4: tau (final CATE regressor)
    let drlearner = DRLearner::new(x.as_ref(), t.as_ref(), y.as_ref(), 0.1, 0.1, 20, 0.1);

    // Estimate Individual Treatment Effect (ITE) using the single DR model
    let uplift_estimate = drlearner.predict_uplift(x.as_ref());

    // Verify if the average estimated uplift is close to the true effect.
    let mut sum_uplift = 0.0;
    for i in 0..n_samples {
        sum_uplift += uplift_estimate[i];
    }
    let avg_uplift = sum_uplift / n_samples as f32;

    println!(
        "True Uplift: 5.0, DRLearner Estimated Avg Uplift: {:.4}",
        avg_uplift
    );

    assert!(
        (avg_uplift - 5.0).abs() < 0.1,
        "Uplift estimation error too high. Got: {:.4}",
        avg_uplift
    );

    // Verify Mathematical Explanation Consistency
    // In DR-Learner, the explanation is straightforward because it uses a single final tau regressor.
    let uplift_explanation = drlearner.explain_uplift(x.as_ref());
    assert_eq!(uplift_explanation.ncols(), n_features);

    let base_value = drlearner.tau.base_value;
    for i in 0..x.nrows() {
        let mut explained_total = 0.0;
        for j in 0..uplift_explanation.ncols() {
            explained_total += uplift_explanation[(i, j)];
        }

        // The sum of contributions + base must equal the explained uplift
        let total_reconstructed_uplift = explained_total + base_value;
        assert!(
            (total_reconstructed_uplift - uplift_estimate[i]).abs() < 1e-4,
            "Uplift explanation delta mismatch at sample {}: Explained {:.4}, Predicted {:.4}",
            i,
            total_reconstructed_uplift,
            uplift_estimate[i]
        );
    }
    println!("DRLearner Uplift Delta Explanation check passed!");
}
