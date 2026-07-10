use faer::Mat;
use xuplift::xmodels::feature_map::KernelFeatureMap;

#[test]
fn test_feature_map_basic_functionality() {
    let data = Mat::from_fn(10, 2, |r, c| (r as f32) + (c as f32));
    let mut map = KernelFeatureMap::new(5);
    let is_categorical = vec![false; 2];
    map.fit(data.as_ref(), &is_categorical);

    assert_eq!(map.num_features, 2);
    assert!(map.num_bases > 0);
    assert_eq!(map.feature_params.len(), 2);
}

#[test]
fn test_transform_feature_into() {
    let data = Mat::from_fn(10, 2, |r, c| (r as f32) * (c as f32 + 1.0));
    let mut map = KernelFeatureMap::new(5);
    let is_categorical = vec![false; 2];
    map.fit(data.as_ref(), &is_categorical);

    for f_idx in 0..map.num_features {
        let mut transformed = Mat::<f32>::zeros(10, map.num_bases);
        map.transform_feature_into(data.as_ref(), f_idx, transformed.as_mut());
        assert_eq!(transformed.nrows(), 10);
        assert_eq!(transformed.ncols(), map.num_bases);
    }
}

#[test]
fn test_full_transform_matches_feature_transform() {
    let data = Mat::from_fn(10, 2, |r, c| (r as f32) + (c as f32));
    let mut map = KernelFeatureMap::new(5);

    let is_categorical = vec![false; 2];
    map.fit(data.as_ref(), &is_categorical);

    let full_transformed = map.transform(data.as_ref());

    for f_idx in 0..map.num_features {
        let mut transformed_feat = Mat::<f32>::zeros(10, map.num_bases);
        map.transform_feature_into(data.as_ref(), f_idx, transformed_feat.as_mut());
        let offset = f_idx * map.num_bases;
        for r_idx in 0..10 {
            for b in 0..map.num_bases {
                let feat_val = transformed_feat[(r_idx, b)];
                let full_val = full_transformed[(r_idx, offset + b)];

                assert!(
                    (feat_val - full_val).abs() < 1e-5,
                    "Mismatch at row {} feat {} base {}. Feat val: {}, Full val: {}",
                    r_idx,
                    f_idx,
                    b,
                    feat_val,
                    full_val
                );
            }
        }
    }
}

#[test]
fn test_nan_handling_in_feature_map() {
    let data = Mat::from_fn(4, 1, |r, _| match r {
        0 => 1.0,
        1 => f32::NAN,
        2 => 3.0,
        3 => 4.0,
        _ => 0.0,
    });

    let mut map = KernelFeatureMap::new(3);
    let is_categorical = vec![false; 1];
    map.fit(data.as_ref(), &is_categorical);

    let x_nan = Mat::from_fn(1, 1, |_, _| f32::NAN);
    let z_nan = map.transform(x_nan.as_ref());

    for r in 0..z_nan.nrows() {
        for c in 0..z_nan.ncols() {
            assert_eq!(
                z_nan[(r, c)],
                0.0,
                "Output for NaN at ({}, {}) should be exactly 0.0",
                r,
                c
            );
        }
    }

    let x_valid = Mat::from_fn(1, 1, |_, _| 2.0);
    let z_valid = map.transform(x_valid.as_ref());

    let mut has_non_zero = false;
    for r in 0..z_valid.nrows() {
        for c in 0..z_valid.ncols() {
            if z_valid[(r, c)] != 0.0 {
                has_non_zero = true;
            }
        }
    }
    assert!(
        has_non_zero,
        "Output for valid value should contain non-zero elements"
    );
}
