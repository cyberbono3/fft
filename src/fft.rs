use num::complex::{Complex, Complex64};
use std::f64::consts::PI;

use crate::dft::dft_complex;
use crate::error::FftError;
use crate::utils::{add_vv, mul_vv_el};

// fft computes the Fast Fourier Transform
pub fn fft(x: &[f64]) -> Result<Vec<Complex64>, FftError> {
    let x_complex: Vec<Complex64> = (0..x.len()).map(|i| Complex::new(x[i], 0_f64)).collect();
    fft_complex(&x_complex)
}

fn fft_complex(x: &[Complex64]) -> Result<Vec<Complex64>, FftError> {
    let N = x.len();
    if !N.is_power_of_two() {
        return Err(FftError::NotAPowerOfTwo(N));
    } else if N <= 2 {
        return dft_complex(x);
    }

    let mut x_even: Vec<Complex64> = Vec::with_capacity(x.len() / 2);
    let mut x_odd: Vec<Complex64> = Vec::with_capacity(x.len() / 2);
    for i in 0..N {
        if i % 2 == 0 {
            x_even.push(x[i]);
        } else {
            x_odd.push(x[i]);
        }
    }
    let x_even_cmplx = fft_complex(&x_even)?;
    let x_odd_cmplx = fft_complex(&x_odd)?;

    let w = Complex::new(0_f64, 2_f64 * PI / N as f64);
    let mut complex = Complex64::default();
    let f_i: Vec<Complex64> = (0..N)
        .map(|i| {
            complex.re = i as f64;
            (w * complex).exp()
        })
        .collect();

    let mut r: Vec<Complex64> = Vec::new();
    let mut aa = add_vv(
        &x_even_cmplx.clone(),
        &mul_vv_el(&x_odd_cmplx, &f_i[0..N / 2]),
    );
    let mut bb = add_vv(&x_even_cmplx, &mul_vv_el(&x_odd_cmplx, &f_i[N / 2..]));
    r.append(&mut aa);
    r.append(&mut bb);

    Ok(r)
}

// ifft computes the Inverse Fast Fourier Transform
pub fn ifft(x: &[Complex64]) -> Result<Vec<f64>, FftError> {
    // use the IFFT method of computing conjugates, then FFT, then conjugate again, and then divide
    // by N
    let x_conj: Vec<Complex64> = (0..x.len()).map(|i| x[i].conj()).collect();
    let x_res = fft_complex(&x_conj)?;
    let r: Vec<Complex64> = (0..x.len()).map(|i| x_res[i].conj()).collect();
    let divisor = Complex::<f64>::new(x.len() as f64, 0_f64);
    let v: Vec<f64> = (0..r.len()).map(|i| (r[i] / divisor).re).collect();
    Ok(v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fft_complex_not_power_of_two() {
        let input = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0), // Length is 3 (not a power of two)
        ];

        let result = fft_complex(&input);
        assert!(result.is_err());

        if let Err(FftError::NotAPowerOfTwo(n)) = result {
            assert_eq!(n, 3);
        }
    }

    #[test]
    fn test_fft_simple_values() {
        let values: Vec<f64> = vec![0.2, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let r = fft(&values).unwrap();

        assert_eq!(r.len(), 8);

        assert_eq!(format!("{:.2}", r[0]), "3.70+0.00i");
        assert_eq!(format!("{:.2}", r[1]), "-0.30-0.97i");
        assert_eq!(format!("{:.2}", r[2]), "-0.30-0.40i");
        assert_eq!(format!("{:.2}", r[3]), "-0.30-0.17i");
        assert_eq!(format!("{:.2}", r[4]), "-0.30+0.00i");

        // expect result similar to initial values
        let o = ifft(&r).unwrap();
        println!("{:?}", o);
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
    fn test_fft_random_values() {
        let values = crate::utils::generate_random_values();
        let r = fft(&values).unwrap();
        println!("{:?}", r.len());
        let o = ifft(&r).unwrap();
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
}
