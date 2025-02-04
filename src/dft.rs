use num::complex::{Complex, Complex64};
use std::f64::consts::PI;

use crate::{error::FftError, utils::mul_mv};

// dft computes the Discrete Fourier Transform
pub fn dft(x: &[f64]) -> Result<Vec<Complex64>, FftError> {
    let x_complex: Vec<Complex64> = (0..x.len()).map(|i| Complex::new(x[i], 0_f64)).collect();
    dft_complex(&x_complex)
}

fn compute_dft_matrix(len: usize, w: Complex64) -> Vec<Vec<Complex64>> {
    (0..len)
        .map(|i| {
            (0..len)
                .map(|j| {
                    let i_compl = Complex::new(0_f64, i as f64);
                    let j_compl = Complex::new(0_f64, j as f64);
                    (w * i_compl * j_compl).exp()
                })
                .collect()
        })
        .collect()
}

pub fn dft_complex(x: &[Complex64]) -> Result<Vec<Complex64>, FftError> {
    let w = Complex::new(0_f64, -2_f64 * PI / x.len() as f64);

    // https://en.wikipedia.org/wiki/Discrete_Fourier_transform
    let dft_matrix: Vec<Vec<Complex64>> = compute_dft_matrix(x.len(), w);

    let r = mul_mv(&dft_matrix, x);
    Ok(r)
}

// idft computes the Inverse Discrete Fourier Transform
pub fn idft(x: &[Complex64]) -> Vec<f64> {
    let w = Complex::new(0_f64, 2_f64 * PI / x.len() as f64);

    // f_k (dft_matrix) = (SUM{n=0, N-1} f_n * e^(j2pi*k*n)/N)/N
    let dft_matrix: Vec<Vec<Complex64>> = compute_dft_matrix(x.len(), w);
    let r = mul_mv(&dft_matrix, x);
    let n = x.len() as f64;
    (0..r.len())
        .map(|i| (r[i] / Complex::new(n, 0_f64)).re)
        .collect()
}

#[test]
fn test_dft_simple_values() {
    let values: Vec<f64> = vec![0.2, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
    let r = dft(&values).unwrap();
    assert_eq!(r.len(), 8);

    assert_eq!(format!("{:.2}", r[0]), "3.70+0.00i");
    assert_eq!(format!("{:.2}", r[1]), "-0.30-0.97i");
    assert_eq!(format!("{:.2}", r[2]), "-0.30-0.40i");
    assert_eq!(format!("{:.2}", r[3]), "-0.30-0.17i");
    assert_eq!(format!("{:.2}", r[4]), "-0.30+0.00i");

    // expect result similar to initial values
    let o = idft(&r);
    assert_eq!(format!("{:.1}", o[0]), "0.2");
    assert_eq!(format!("{:.1}", o[1]), "0.2");
    assert_eq!(format!("{:.1}", o[2]), "0.3");
    assert_eq!(format!("{:.1}", o[3]), "0.4");
    assert_eq!(format!("{:.1}", o[4]), "0.5");
    assert_eq!(format!("{:.1}", o[5]), "0.6");
    assert_eq!(format!("{:.1}", o[6]), "0.7");
    assert_eq!(format!("{:.1}", o[7]), "0.8");
}

#[test]
fn test_dft_random_values() {
    let values = crate::utils::generate_random_values();
    let r = dft(&values).unwrap();
    println!("{:?}", r.len());
    let o = idft(&r);
    assert_eq!(values.len(), o.len());
    // Compare each index with a tolerance of 1e-5
    for i in 0..r.len() {
        let diff = (values[i] - o[i]).abs();
        // If diff < 1e-5, they match up to ~5 decimal places.
        assert!(
            diff < 1e-5,
            "Mismatch at index {}: {} vs {}, difference {}",
            i,
            values[i],
            o[i],
            diff
        );
    }
}
