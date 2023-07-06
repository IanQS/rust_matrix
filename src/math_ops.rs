/// Operations on the matrices; typically things that are mapped element-wise
///     These are operations that aren't exclusive to matrices e.g. you can take the
///     sin of a number, or take the exp of it, etc.
use rayon::prelude::*;
use crate::matrix::Matrix;
use crate::matrix::TensorType::{MatrixT, ScalarT, VectorT};

const E: f64 = std::f64::consts::E;

fn map_across_row_out_scalar(
    m1: &Matrix,
    f: fn(Vec<f64>) -> f64,
    is_strict: bool,
) -> Matrix
{
    let cloned_mat = m1.clone();
    match cloned_mat.data {
        ScalarT(v) => {
            if is_strict {
                panic!("Attempting to apply row-vector reducing op op on a singular scalar. This is probably not what you intended");
            } else {
                Matrix::from_scalar(f(vec![v]))
            }
        }
        VectorT(v, as_row) => {
            if as_row {
                Matrix::from_scalar(f(v))
            } else if is_strict {
                panic!("Attempting to do a row-vec operation on a col-vec")
            } else {
                Matrix::from_scalar(f(v))
            }
        }
        MatrixT(v) => {
            let final_res: Vec<f64> = v.into_par_iter().map(f).collect();
            Matrix::from_vector(&final_res, true)
        }
    }
}

/// Maps a function across an entire row of elements and spits out another row
fn map_across_row_out_vec(
    m1: &Matrix,
    f: fn(Vec<f64>) -> Vec<f64>,
    is_strict: bool,
) -> Matrix
{
    let cloned_mat = m1.clone();
    match cloned_mat.data {
        ScalarT(v) => {
            if is_strict {
                panic!("Attempting to apply vector-wise op on a singular scalar. This is probably not what you intended");
            } else {
                Matrix::from_vector(&f(vec![v]), true)
            }
        }
        VectorT(v, as_row) => {
            if as_row {
                Matrix::from_vector(&f(v), true)
            } else if is_strict {
                panic!("Attempting to do a row-vec operation on a col-vec")
            } else {
                Matrix::from_vector(&f(v), true)
            }
        }
        MatrixT(v) => {
            let final_res = v.into_par_iter().map(f).collect();
            Matrix::from_mat(&final_res)
        }
    }
}

// Maps a float64 onto another float64.
fn map_elementwise(m1: &Matrix, f: fn(f64) -> f64) -> Matrix {
    return match m1.data.clone() {
        ScalarT(v) => {
            Matrix::from_scalar(f(v))
        }
        VectorT(v, as_row) => {
            let tmp_res = v.par_iter()
                .map(|s| f(*s))
                .collect();

            Matrix::from_vector(&tmp_res, as_row)
        }
        MatrixT(m) => {
            let tmp_res = m.par_iter()
                .map(
                    |v| v.par_iter()
                        .map(|s| f(*s)).collect()
                ).collect();

            Matrix::from_mat(&tmp_res)
        }
    };
}

#[must_use]
pub fn sin(m1: &Matrix) -> Matrix {
    map_elementwise(m1, f64::sin)
}


#[must_use]
pub fn cos(m1: &Matrix) -> Matrix {
    map_elementwise(m1, f64::cos)
}

#[must_use]
pub fn exp(m1: &Matrix) -> Matrix {
    map_elementwise(m1, f64::exp)
}

#[must_use]
pub fn pow(m1: &Matrix, raise_to: i32) -> Matrix {
    match m1.data.clone() {
        ScalarT(s) => {
            Matrix::from_scalar(f64::powi(s, raise_to))
        }
        VectorT(v, as_row) => {
            Matrix::from_vector(
                &v.par_iter()
                    .map(|s| f64::powi(*s, raise_to))
                    .collect(),
                as_row,
            )
        }
        MatrixT(m) => {
            Matrix::from_mat(
                &m.par_iter()
                    .map(
                        |v| v.par_iter()
                            .map(|s| f64::powi(*s, raise_to)).collect()
                    ).collect(),
            )
        }
    }
}

/// Take element-wise log of the current function
/// Note: Tensorflow doesn't seem to allow us to take the log at bases other
/// than e.
#[must_use]
pub fn log(m1: &Matrix) -> Matrix {
    match m1.data.clone() {
        ScalarT(s) => {
            Matrix::from_scalar(f64::log(s, E))
        }
        VectorT(v, as_row) => {
            Matrix::from_vector(
                &v.par_iter()
                    .map(|s| f64::log(*s, E))
                    .collect(),
                as_row,
            )
        }
        MatrixT(m) => {
            Matrix::from_mat(
                &m.par_iter()
                    .map(
                        |v| v.par_iter()
                            .map(|s| f64::log(*s, E)).collect()
                    ).collect(),
            )
        }
    }
}

#[must_use]
pub fn sigmoid(m1: &Matrix) -> Matrix {
    map_elementwise(
        m1,
        |x| -> f64 {
            1.0 / (1.0 + f64::exp(-x))
        })
}

#[must_use]
pub fn softmax(m1: &Matrix) -> Matrix {
    todo!()
}

#[must_use]
pub fn relu(m1: &Matrix) -> Matrix {
    map_elementwise(m1, |x| f64::max(x, 0.0))
}


#[must_use]
pub fn clip(m1: &Matrix, min_val: f64, max_val: f64) -> Matrix {
    match m1.data.clone() {
        ScalarT(s) => {
            Matrix::from_scalar(s.clamp(min_val, max_val))
        }
        VectorT(v, as_row) => {
            Matrix::from_vector(
                &v.par_iter()
                    .map(|s| s.clamp(min_val, max_val))
                    .collect(),
                as_row,
            )
        }
        MatrixT(m) => {
            Matrix::from_mat(
                &m.par_iter()
                    .map(
                        |v| v.par_iter()
                            .map(|s| s.clamp(min_val, max_val)).collect()
                    ).collect(),
            )
        }
    }
}


#[cfg(test)]
mod tests {
    use crate::matrix::{Axis, Matrix};
    use crate::math_ops::*;

    fn setup_matrix() -> Matrix {
        Matrix::from_mat(
            &vec![
                vec![-1.0, -2.0, -3.0],
                vec![-0.1, -0.2, -0.3],
                vec![0.1, 0.2, 0.3],
                vec![1.0, 2.0, 3.0],
            ],
        )
    }

    #[test]
    fn test_sin() {
        let mat = setup_matrix();
        let res = sin(&mat);


        let expected_eval = &vec![
            vec![-0.8414709848078965, -0.9092974268256817, -0.1411200080598672],
            vec![-0.09983341664682815, -0.19866933079506122, -0.29552020666133955],
            vec![0.09983341664682815, 0.19866933079506122, 0.29552020666133955],
            vec![0.8414709848078965, 0.9092974268256817, 0.1411200080598672],
        ];
        assert!(Matrix::almost_equal(
            &res,
            &Matrix::from_mat(expected_eval),
            Some(0.001),
        ), "Failed sin-eval");
    }


    #[test]
    fn test_exp() {
        let mat = setup_matrix();
        let res = exp(&mat);

        // Result is the same for exp and its derivative
        let expected = &vec![
            vec![0.3678794411714423, 0.1353352832366127, 0.049787068367863944],
            vec![0.9048374180359595, 0.8187307530779818, 0.7408182206817179],
            vec![1.1051709180756477, 1.2214027581601699, 1.3498588075760032],
            vec![2.7182818284590455, 7.38905609893065, 20.085536923187668],
        ];
        assert!(Matrix::almost_equal(
            &res,
            &Matrix::from_mat(expected),
            Some(0.001),
        ), "Failed exp-eval");
    }


    #[test]
    fn test_pow() {
        let mat = setup_matrix();
        let res = pow(&mat, 4);

        let expected_eval = &vec![
            vec![1.0, 16.0, 81.0],
            vec![0.00010000000000000002, 0.0016000000000000003, 0.0081],
            vec![0.00010000000000000002, 0.0016000000000000003, 0.0081],
            vec![1.0, 16.0, 81.0],
        ];
        assert!(Matrix::almost_equal(
            &res,
            &Matrix::from_mat(expected_eval),
            Some(0.001),
        ), "Failed pow-eval");
    }


    #[test]
    fn test_log() {
        let mat = setup_matrix();
        let res = log(&mat);

        let expected_eval = &vec![
            vec![f64::NAN, f64::NAN, f64::NAN],
            vec![f64::NAN, f64::NAN, f64::NAN],
            vec![-2.3025850929940455, -1.6094379124341003, -1.203972804325936],
            vec![0.0, 0.6931471805599453, 1.0986122886681096],
        ];

        assert!(Matrix::almost_equal(
            &res,
            &Matrix::from_mat(expected_eval),
            Some(0.001),
        ), "Failed log-eval");
    }


    #[test]
    fn test_sigmoid() {
        let mat = setup_matrix();
        let res = sigmoid(&mat);

        let expected_eval = &vec![
            vec![0.2689414213699951, 0.11920292202211755, 0.04742587317756679],
            vec![0.47502081252106, 0.4501660026875221, 0.425557483188341],
            vec![0.52497918747894, 0.549833997312478, 0.574442516811659],
            vec![0.7310585786300049, 0.8807970779778824, 0.9525741268224333],
        ];
        assert!(Matrix::almost_equal(
            &res,
            &Matrix::from_mat(expected_eval),
            Some(0.001),
        ), "Failed sigmoid-eval");
    }

    #[test]
    fn test_relu() {
        let mat = setup_matrix();
        let res = relu(&mat);
        let expected_eval = &vec![
            vec![0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0],
            vec![0.1, 0.2, 0.3],
            vec![1.0, 2.0, 3.0],
        ];

        assert!(Matrix::almost_equal(
            &res,
            &Matrix::from_mat(expected_eval),
            Some(0.001),
        ), "Failed relu-eval");
    }

    #[test]
    fn test_clip() {
        let mat = setup_matrix();
        let res = clip(&mat, -0.5, 5.0);

        let expected_eval = &vec![
            vec![-0.5, -0.5, -0.5],
            vec![-0.1, -0.2, -0.3],
            vec![0.1, 0.2, 0.3],
            vec![1.0, 2.0, 3.0],
        ];
        assert!(Matrix::almost_equal(
            &res,
            &Matrix::from_mat(expected_eval),
            Some(0.001),
        ), "Failed clip-eval");
    }
}