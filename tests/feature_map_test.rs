use faer::{Col, Mat};
use xuplift::xmodels::feature_map::KernelFeatureMap;

#[test]
fn test_feature_map_basic_functionality() {
    // --- Model Initialization ---
    // Verify that the KernelFeatureMap initializes and fits correctly on a standard linear dataset.
    let data = Mat::from_fn(10, 2, |r, c| (r as f32) + (c as f32));
    let mut map = KernelFeatureMap::new();
    let is_categorical = vec![false; 2];
    map.fit(data.as_ref(), &is_categorical);

    // --- Verification ---
    assert_eq!(map.num_features, 2);
    assert!(map.num_bases > 0);
    assert_eq!(map.feature_bases.len(), 2);
    assert_eq!(map.proj_matrices.len(), 2);
    assert_eq!(map.feature_means.len(), 2);
    assert_eq!(map.s2_invs.len(), 2);
}

#[test]
fn test_transform_feature_into() {
    // --- Model Initialization ---
    // Verify that the transform operation produces matrices with the correct dimensions.
    let data = Mat::from_fn(5, 2, |r, c| (r as f32) * (c as f32 + 1.0));
    let mut map = KernelFeatureMap::new();
    let is_categorical = vec![false; 2];
    map.fit(data.as_ref(), &is_categorical);

    // --- Verification ---
    for f_idx in 0..map.num_features {
        let mut transformed = Mat::<f32>::zeros(5, map.num_bases);
        map.transform_feature_into(data.as_ref(), f_idx, transformed.as_mut());
        assert_eq!(transformed.nrows(), 5);
        assert_eq!(transformed.ncols(), map.num_bases);
    }
}

#[test]
fn test_transform_row_matches_feature_transform() {
    // --- Model Initialization ---
    // Verify that transforming rows individually yields identical results to feature-wise transformation.
    let data = Mat::from_fn(5, 2, |r, c| (r as f32) + (c as f32));
    let mut map = KernelFeatureMap::new();

    let is_categorical = vec![false; 2];
    map.fit(data.as_ref(), &is_categorical);

    // --- Verification ---
    for f_idx in 0..map.num_features {
        let mut transformed_feat = Mat::<f32>::zeros(5, map.num_bases);
        map.transform_feature_into(data.as_ref(), f_idx, transformed_feat.as_mut());

        for r_idx in 0..5 {
            let total_dim = map.num_features * map.num_bases;
            let mut transformed_row = Col::<f32>::zeros(total_dim);
            map.transform_row_into(data.as_ref(), r_idx, transformed_row.as_mut());

            let offset = f_idx * map.num_bases;
            for b in 0..map.num_bases {
                let feat_val = transformed_feat[(r_idx, b)];
                let row_val = transformed_row[offset + b];
                assert!(
                    (feat_val - row_val).abs() < 1e-5,
                    "Mismatch at row {} feat {} base {}",
                    r_idx,
                    f_idx,
                    b
                );
            }
        }
    }
}

#[test]
fn test_nan_handling_in_feature_map() {
    // --- Synthetic Data Generation ---
    // Create a matrix with some NaNs to verify robust handling
    let data = Mat::from_fn(4, 1, |r, _| match r {
        0 => 1.0,
        1 => f32::NAN,
        2 => 3.0,
        3 => 4.0,
        _ => 0.0,
    });

    // --- Model Initialization ---
    let mut map = KernelFeatureMap::new();
    let is_categorical = vec![false; 1];
    map.fit(data.as_ref(), &is_categorical);

    // --- Verification ---
    // Test transform_row with a NaN row
    // Verify that the transformer correctly masks NaNs by outputting a zero vector.
    let x_nan = Mat::from_fn(1, 1, |_, _| f32::NAN);

    let total_dim = map.num_features * map.num_bases;
    let mut z_nan = Col::<f32>::zeros(total_dim);
    map.transform_row_into(x_nan.as_ref(), 0, z_nan.as_mut());
    assert!(
        z_nan.iter().all(|&val| val == 0.0),
        "Output for NaN should be zero vector"
    );

    // Test transform_row with a valid row
    let x_valid = Mat::from_fn(1, 1, |_, _| 2.0);
    let total_dim = map.num_features * map.num_bases;
    let mut z_valid = Col::<f32>::zeros(total_dim);
    map.transform_row_into(x_valid.as_ref(), 0, z_valid.as_mut());
    assert!(
        z_valid.iter().any(|&val| val != 0.0),
        "Output for valid value should not be zero vector"
    );
}
