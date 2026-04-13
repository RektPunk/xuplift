use faer::{Col, Mat};

use xuplift::metalearners::slearner::SLearner;

#[test]
fn test_slearner_constant_uplift() {
    let n_samples = 200;
    let n_features = 3;

    let mut x = Mat::<f32>::zeros(n_samples, n_features);
    let mut t = Col::<f32>::zeros(n_samples);
    let mut y = Col::<f32>::zeros(n_samples);

    // 1. Synthetic Data Generation
    // Objective: Create a dataset with a known constant treatment effect.
    // Generative Model: y = 1.5*x0 + 0.5*x1 + (2.0 * t) + 10.0
    // Ground Truth Uplift (ITE): 2.0
    for i in 0..n_samples {
        let x0 = i as f32 * 0.01;
        let x1 = (i as f32).sin();
        let x2 = (i as f32).cos();

        x[(i, 0)] = x0;
        x[(i, 1)] = x1;
        x[(i, 2)] = x2;

        // Assign treatment: Even indices = Treatment (1), Odd indices = Control (0)
        let treatment = if i % 2 == 0 { 1.0 } else { 0.0 };
        t[i] = treatment;
        y[i] = 1.5 * x0 + 0.5 * x1 + (2.0 * treatment) + 10.0;
    }

    // 2. Model Training
    // Initialize SLearner which internally handles feature augmentation and kernel mapping.
    let slearner = SLearner::new(&x, &t, &y);

    // 3. Uplift Estimation
    // Estimate Individual Treatment Effect (ITE) using the counterfactual approach:
    // τ(x) = E[Y|X, T=1] - E[Y|X, T=0]
    let uplift_estimate = slearner.predict_uplift(&x);

    // 4. Accuracy Verification
    // Check if the average estimated uplift converges to the true effect (2.0).
    let mut sum_uplift = 0.0;
    for i in 0..n_samples {
        sum_uplift += uplift_estimate[i];
    }
    let avg_uplift = sum_uplift / n_samples as f32;

    println!(
        "True Uplift: 2.0, Estimated Average Uplift: {:.4}",
        avg_uplift
    );

    assert!(
        (avg_uplift - 2.0).abs() < 0.1,
        "Uplift estimation is too far from ground truth. Got: {:.4}",
        avg_uplift
    );

    // 5. Explanation Consistency Check
    // Verify that the sum of feature contribution deltas matches the predicted uplift.
    // Mathematical Consistency: Σ(Contribution_T1 - Contribution_T0) == Predict_T1 - Predict_T0
    let uplift_explanation = slearner.explain_uplift(&x);
    for i in 0..x.nrows() {
        let mut explained_total = 0.0;
        for j in 0..uplift_explanation.ncols() {
            explained_total += uplift_explanation[(i, j)];
        }

        // The total explained uplift must match the actual prediction score for each sample.
        assert!(
            (explained_total - uplift_estimate[i]).abs() < 1e-4,
            "Uplift explanation delta mismatch at sample {}: Explained {:.4}, Predicted {:.4}",
            i,
            explained_total,
            uplift_estimate[i]
        );
    }
    println!("SLearner Uplift Delta Explanation check passed!");
}
