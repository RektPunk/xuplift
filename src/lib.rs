pub mod feature_map;
pub mod models;

pub use crate::feature_map::KernelFeatureMap;
pub use crate::models::classifier::Classifier;
pub use crate::models::regressor::Regressor;

#[cfg(test)]
mod tests {
    use super::*;
    use faer::{Col, Mat};
    use rand_distr::{Distribution, Normal};

    #[test]
    fn test_gaussian_blobs_classification() {
        let mut rng = rand::rng();
        let n_samples = 300;
        let mut x = Mat::<f32>::zeros(n_samples, 2);
        let mut y = Col::<f32>::zeros(n_samples);

        // 1. Generate Gaussian Blobs (Linearly separable but complex)
        let dist1 = Normal::new(-1.5, 0.7).unwrap();
        let dist2 = Normal::new(1.5, 0.7).unwrap();

        for i in 0..n_samples {
            if i < n_samples / 2 {
                x[(i, 0)] = dist1.sample(&mut rng);
                x[(i, 1)] = dist1.sample(&mut rng);
                y[i] = 0.0;
            } else {
                x[(i, 0)] = dist2.sample(&mut rng);
                x[(i, 1)] = dist2.sample(&mut rng);
                y[i] = 1.0;
            }
        }

        // 2. Setup Kernel Feature Map
        let mut map = KernelFeatureMap::new(); // 10 bases are enough for blobs
        map.fit(&x);

        // 3. Setup Classifier (Explainable IRLS)
        let mut model = Classifier::new(map);
        model.fit(&y, 20);

        // 4. Verify Accuracy
        let probs = model.predict(&x);
        let mut correct = 0;
        for i in 0..n_samples {
            let pred = if probs[i] > 0.5 { 1.0 } else { 0.0 };
            if (pred - y[i]).abs() < 1e-5 {
                correct += 1;
            }
        }

        let accuracy = correct as f32 / n_samples as f32;
        println!("Classification Accuracy: {:.2}%", accuracy * 100.0);
        assert!(
            accuracy > 0.95,
            "Accuracy should be above 95% for Gaussian Blobs"
        );
    }

    #[test]
    fn test_noisy_linear_regression() {
        use super::*;
        use crate::KernelFeatureMap;
        let n_samples = 100;
        let mut x = Mat::<f32>::zeros(n_samples, 1);
        let mut y = Col::<f32>::zeros(n_samples);

        // 1. Generate Noisy Linear Data: y = 3x + 2 + noise
        for i in 0..n_samples {
            let val = i as f32 / 10.0;
            x[(i, 0)] = val;
            y[i] = 3.0 * val + 2.0;
        }

        // 2. Setup Kernel Feature Map
        let mut map = KernelFeatureMap::new();
        map.fit(&x);

        // 3. Setup Regressor (Ridge)
        let mut model = Regressor::new(map);
        model.fit(&y);

        // 4. Verify R-squared or Mean Absolute Error
        let preds = model.predict(&x);
        let mut total_error = 0.0;
        for i in 0..n_samples {
            total_error += (preds[i] - y[i]).abs();
        }

        let mae = total_error / n_samples as f32;
        println!("Regression MAE: {:.4}", mae);
        assert!(mae < 0.5, "Regression MAE is too high");
    }
}
