use num::complex::Complex64;

use rand::prelude::*;

use itertools::Itertools;

// mul_mv multiplies a Matrix by a Vector
pub fn mul_mv(m: &[Vec<Complex64>], v: &[Complex64]) -> Vec<Complex64> {
    assert_eq!(m[0].len(), m.len());
    assert_eq!(m.len(), v.len());

    (0..m.len())
        .map(|i| (0..m.len()).map(|j| m[i][j] * v[j]).sum())
        .collect()
}

pub fn add_vv(a: &[Complex64], b: &[Complex64]) -> Vec<Complex64> {
    a.iter().zip_eq(b.iter()).map(|(x, y)| x + y).collect()
}

// mul_vv_el multiplies elements of one vector by the elements of another vector
pub fn mul_vv_el(a: &[Complex64], b: &[Complex64]) -> Vec<Complex64> {
    a.iter().zip_eq(b.iter()).map(|(x, y)| x * y).collect()
}

pub fn generate_random_values() -> Vec<f64> {
    let mut rng = rand::rng();

    // Generate 1024 random f64 values (uniformly distributed in [0, 1))
    (0..1024).map(|_| rng.random::<f64>()).collect::<Vec<f64>>()
}

#[cfg(test)]
mod tests {
    use super::*;
    use num::complex::Complex64;

    #[test]
    fn test_mul_mv() {
        // 2x2 matrix times a 2-element vector
        let m = vec![
            vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)],
            vec![Complex64::new(3.0, 0.0), Complex64::new(4.0, 0.0)],
        ];
        let v = vec![Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0)];

        // Expected result:
        // row 0: (1*1) + (2*1) = 3
        // row 1: (3*1) + (4*1) = 7
        let expected = vec![Complex64::new(3.0, 0.0), Complex64::new(7.0, 0.0)];

        let result = mul_mv(&m, &v);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_add_vv() {
        // Simple vector addition
        let a = vec![Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)];
        let b = vec![Complex64::new(5.0, 6.0), Complex64::new(7.0, 8.0)];

        // Expected result:
        // (1+2i) + (5+6i) = (6 + 8i)
        // (3+4i) + (7+8i) = (10 + 12i)
        let expected = vec![Complex64::new(6.0, 8.0), Complex64::new(10.0, 12.0)];

        let result = add_vv(&a, &b);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_mul_vv_el() {
        // Element-wise multiplication
        let a = vec![Complex64::new(1.0, 1.0), Complex64::new(2.0, 0.0)];
        let b = vec![Complex64::new(3.0, 2.0), Complex64::new(4.0, 1.0)];

        // Expected result:
        // (1+1i)*(3+2i) = 1+5i
        // (2+0i)*(4+1i) = 8+2i
        let expected = vec![Complex64::new(1.0, 5.0), Complex64::new(8.0, 2.0)];

        let result = mul_vv_el(&a, &b);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_generate_random_values() {
        let random_values = generate_random_values();
        // The spec says we generate 1024 random floats in [0,1)
        assert_eq!(random_values.len(), 1024);

        // Check that each value is within [0, 1)
        for &val in &random_values {
            assert!(val >= 0.0 && val < 1.0);
        }
    }
}
