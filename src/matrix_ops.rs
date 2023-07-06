//////////////////////////////
// Code related to the primitive operations:
//      +, -, *, /
//      and things like transpose, dot product, sum, axis_sum
//////////////////////////////
use crate::raw_broadcasting::broadcasting_ops::{
    _matrix_matrix, _matrix_scalar, _matrix_vector,
    _scalar_matrix, _scalar_scalar, _scalar_vector,
    _vector_matrix, _vector_scalar, _vector_vector,
};

use std::ops;

use rayon::prelude::*;
use crate::matrix::{Axis, Matrix};
use crate::matrix::TensorType::{ScalarT, VectorT, MatrixT};
use crate::raw_broadcasting::Op;

impl ops::Sub<Self> for Matrix {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        assert!(Self::is_broadcastable(&self, &rhs), "Could not broadcast sub between: {self:?} and {rhs:?}");
        _broadcast_op(&self, &rhs, Op::Sub)
    }
}

impl ops::Add<Self> for Matrix {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        assert!(Self::is_broadcastable(&self, &rhs), "Could not broadcast addition between: {self:?} and {rhs:?}");
        _broadcast_op(&self, &rhs, Op::Add)
    }
}

impl ops::Mul<Self> for Matrix {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        assert!(Self::is_broadcastable(&self, &rhs), "Could not broadcast multiplication between: {self:?} and {rhs:?}");
        _broadcast_op(&self, &rhs, Op::Mul)
    }
}

impl ops::Div<Self> for Matrix {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        assert!(Self::is_broadcastable(&self, &rhs), "Could not broadcast div between: {self:?} and {rhs:?}");
        _broadcast_op(&self, &rhs, Op::Div)
    }
}

pub(crate) fn _broadcast_op(lhs: &Matrix, rhs: &Matrix, op: Op) -> Matrix {
    return match (lhs.data.clone(), rhs.data.clone()) {
        // All the scalars
        (ScalarT(s1), ScalarT(s2)) => {
            Matrix::from_scalar(_scalar_scalar(s1, s2, op))
        }
        (ScalarT(s), VectorT(v, as_row)) => {
            Matrix::from_vector(
                &_scalar_vector(s, &v, op),
                as_row,
            )
        }
        (ScalarT(s), MatrixT(m)) => {
            Matrix::from_mat(&_scalar_matrix(s, &m, op))
        }
        // All the vector ops
        (VectorT(v, as_row), ScalarT(s)) => {
            Matrix::from_vector(&_vector_scalar(&v, s, op),
                                as_row,
            )
        }
        (VectorT(v1, _), VectorT(v2, _)) => {
            match (lhs.shape, rhs.shape) {
                // row-vec && col-vec
                // We iterate over the
                ((1, a), (b, 1)) => {
                    let inner_res = match op {
                        Op::Div => {
                            let transposed_res: Vec<Vec<f64>> = v1.par_iter()
                                .map(|x| {
                                    _scalar_vector(*x, &v2, Op::Div)
                                })
                                .collect();
                            Matrix::transpose_raw_matrix(&transposed_res, b, a)
                        }
                        _ => {
                            v2.par_iter()
                                .map(|x| {
                                    match op {
                                        Op::Sub => _scalar_vector(-1.0 * (*x), &v1, Op::Add),
                                        _ => _scalar_vector(*x, &v1, op)
                                    }
                                })
                                .collect()
                        }
                    };
                    Matrix::from_mat(
                        &inner_res
                    )
                }
                // col-vec && row-vec
                ((_a, 1), (1, _b)) => {
                    let res = v1.par_iter()
                        .map(|x| _scalar_vector(*x, &v2, op))
                        .collect();
                    Matrix::from_mat(&res)
                }
                _ => {
                    Matrix::from_vector(
                        &_vector_vector(&v1, &v2, op),
                        lhs.shape.0 == 1,
                    )
                }
            }
        }
        (VectorT(v, _), MatrixT(m)) => {
            Matrix::from_mat(&_vector_matrix(&v, &m, op))
        }
        (MatrixT(m), ScalarT(s)) => {
            Matrix::from_mat(&_matrix_scalar(&m, s, op))
        }
        (MatrixT(m), VectorT(v, _)) => {
            Matrix::from_mat(&_matrix_vector(&m, &v, op))
        }
        (MatrixT(m1), MatrixT(m2)) => {
            Matrix::from_mat(&_matrix_matrix(&m1, &m2, op))
        }
    };
}

fn _scalar_dot(lhs: &Matrix, rhs: &Matrix) -> Matrix {
    let ScalarT(self_data) = lhs.data.clone() else { panic!("Should not be here") };
    match rhs.data.clone() {
        // Scalar * Scalar
        ScalarT(other_s) => {
            Matrix::from_scalar(self_data * other_s)
        }
        // Scalar * Vector -> Vector. Standard broadcasted multiplication
        VectorT(other_v, as_row) => {
            Matrix::from_vector(&_scalar_vector(self_data, &other_v, Op::Mul),
                                as_row,
            )
        }
        // Scalar * Matrix -> Matrix. Standard multiplication
        MatrixT(other_v) => {
            Matrix::from_mat(&_scalar_matrix(self_data, &other_v, Op::Mul))
        }
    }
}

fn _vector_dot(lhs: &Matrix, rhs: &Matrix) -> Matrix {
    let VectorT(self_data, lhs_as_row) = lhs.data.clone() else { panic!("Should not be here") };
    return match rhs.data.clone() {
        // Vector-Scalar -> Vector. Broadcasted mult
        ScalarT(other_data) => {
            Matrix::from_vector(
                &_vector_scalar(&self_data, other_data, Op::Mul),
                lhs_as_row,
            )
        }
        // Vector * Vector -> {Scalar, Matrix}
        VectorT(other_data, _) => {
            assert_eq!(lhs.shape.1, rhs.shape.0, "Taking inner product of incompatibly-shaped vectors: {lhs:?} and {rhs:?}");

            // Reduction to a scalar
            if lhs_as_row {
                // Take the elementwise mult, then sum
                Matrix::from_scalar(
                    _vector_vector(&self_data, &other_data, Op::Mul)
                        .iter()
                        .sum::<f64>(),
                )
            } else {
                // Outer product
                let num_rows = lhs.shape.0;
                let num_cols = rhs.shape.1;
                let mut container: Vec<Vec<f64>> = vec![vec![0.0; num_cols]; num_rows];
                for i in 0..num_rows {
                    for j in 0..num_cols {
                        container[i][j] = self_data[i] * other_data[j];
                    }
                }
                Matrix::from_mat(
                    &container,
                )
            }
        }
        // Vector * Matrix -> Vector. (1, X) * (X, Y) -> (1, Y) another row-vector
        MatrixT(other_data) => {
            assert_ne!(lhs.shape.1, 1, "Attempting to take dot-product of col-vector and a matrix: {self_data:?} and {other_data:?}");

            let rotated = Matrix::transpose_raw_matrix(
                &other_data,
                rhs.shape.1, rhs.shape.0);

            let inner_producted: Vec<f64> = rotated
                .par_iter()
                .map(
                    |x| _vector_vector(x, &self_data, Op::Mul).iter().sum()
                )
                .collect();

            Matrix::from_vector(&inner_producted, true)
        }
    };
}

fn _matrix_dot(lhs: &Matrix, rhs: &Matrix) -> Matrix {
    let MatrixT(self_data) = lhs.data.clone() else { panic!("Should not be here") };
    return match rhs.data.clone() {
        // Matrix-Scalar -> Matrix. Broadcasted mult
        ScalarT(other_data) => {
            Matrix::from_mat(&_matrix_scalar(&self_data, other_data, Op::Mul))
        }
        // Matrix * Vector -> Vector
        VectorT(other_data, _) => {
            assert_eq!(lhs.shape.1, rhs.shape.0, "Taking inner product of incompatibly-shaped matrix-vector: {lhs:?} and {rhs:?}");

            let res: Vec<f64> = self_data
                .par_iter()
                .map(|x| _vector_vector(x, &other_data, Op::Mul).iter().sum())
                .collect();

            Matrix::from_vector(
                &res,
                false,
            )
        }
        // Matrix * Matrix -> Matrix . (1, X) * (X, Y) -> (1, Y) another row-vector
        MatrixT(other_data) => {
            assert_ne!(lhs.shape.1, 1, "Attempting to take dot-product of col-vector and a matrix: {self_data:?} and {other_data:?}");
            let mut result_mat = vec![vec![0.0; lhs.shape.0]; rhs.shape.1];
            let num_cols = other_data[0].len();
            let num_rows = other_data.len();

            let transposed_other = Matrix::transpose_raw_matrix(
                &other_data,
                num_cols,
                num_rows,
            );
            for row_idx in 0..result_mat.len() {
                for col_idx in 0..result_mat[0].len() {
                    result_mat[row_idx][col_idx] = _vector_vector(
                        &self_data[row_idx],
                        &transposed_other[col_idx],
                        Op::Mul,
                    ).iter().sum();
                }
            }

            Matrix::from_mat(
                &result_mat,
            )
        }
    };
}

pub(crate) fn _dot(this: &Matrix, other: &Matrix) -> Matrix {
    assert_eq!(this.shape.1, other.shape.0, "Attempting to take dot-product of incompatibly shaped matrices");
    match (this.data.clone(), other.data.clone()) {
        (ScalarT(_), _) => {
            _scalar_dot(this, other)
        }
        (VectorT(_, _), _) => {
            _vector_dot(this, other)
        }
        (MatrixT(_), _) => {
            _matrix_dot(this, other)
        }
    }
}


#[must_use]
pub(crate) fn _sum(m1: &Matrix) -> Matrix {
    match m1.data.clone() {
        ScalarT(v) => { Matrix::from_scalar(v) }
        VectorT(v, _) => {
            Matrix::from_scalar(v.iter().sum::<f64>())
        }
        MatrixT(mat) => {
            let accum: f64 = mat.par_iter().map(
                |vec| -> f64{ vec.par_iter().sum() }
            ).sum();
            Matrix::from_scalar(accum)
        }
    }
}


#[must_use]
pub fn _sum_axis(m1: &Matrix, axis: &Axis) -> Matrix {
    match m1.data.clone() {
        ScalarT(s) => { Matrix::from_scalar(s) }
        VectorT(v, as_row) => {
            match (axis, as_row) {
                (Axis::First, false) | (Axis::Second, true) => {
                    // Sum along the non-sized-1 axis, so there is something to sum over!
                    // Matrix::from_scalar(v.iter().sum()::<f64>())
                    Matrix::from_scalar(v.iter().sum::<f64>())
                }
                _ => {
                    Matrix::from_vector(&v, as_row)
                }
            }
        }
        MatrixT(mat) => {
            // Consider a matrix of size (m, n)
            match axis {
                Axis::First => { // reduces the matrix to a vector of size (1, n)
                    // First transpose to transform initial matrix size to (n, m)
                    let transposed = Matrix::transpose_raw_matrix(
                        &mat, m1.shape.1, m1.shape.0);
                    // Then do a reduce, as we did in the other block, which transforms us into (n, 1)
                    let res = transposed.iter().map(|x| x.iter().sum::<f64>()).collect();

                    // BUT because our col-vecs and row-vecs are stored as vectors we don't need to reshape. All
                    // we have to do is specify that it should be a row vec
                    Matrix::from_vector(&res, true)
                }
                Axis::Second => { // reduces the matrix to a vector of size (m, 1)
                    let res = mat.par_iter().map(
                        |x| (*x).par_iter().sum::<f64>()
                    ).collect();
                    Matrix::from_vector(&res, false)
                }
            }
        }
    }
}


#[cfg(test)]
mod tests {
    use crate::matrix::{Axis, Matrix};

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
    fn test_axis_sum() {
        let lhs = setup_matrix();
        let rhs = setup_matrix();

        let addition = lhs + rhs;
        let sum1 = addition.sum_axis(&Axis::First);
        let sum2 = addition.sum_axis(&Axis::Second);

        let expected_sum1 = &vec![0, 0, 0];
        assert!(Matrix::almost_equal(
            &sum1,
            &Matrix::from_vector(expected_sum1, true),
            Some(0.001),
        ), "Failed axis_sum_1-eval");


        let expected_sum2 = &vec![-12.0, -1.2, 1.2, 12.0];
        assert!(Matrix::almost_equal(
            &sum2,
            &Matrix::from_vector(expected_sum2, false),
            Some(0.001),
        ), "Failed axis_sum_2-eval");
    }

    #[test]
    fn test_add() {
        let a = Matrix::from_seed(3, 4, 1.5);
        let b = Matrix::zeros(3, 4);
        let c = a + b;


        let expected_eval = &vec![
            vec![1.5, 1.5, 1.5, 1.5, ],
            vec![1.5, 1.5, 1.5, 1.5, ],
            vec![1.5, 1.5, 1.5, 1.5, ],
        ];

        assert!(Matrix::almost_equal(
            &c,
            &Matrix::from_mat(expected_eval),
            Some(0.001),
        ), "Failed add-eval");

    }


    #[test]
    fn test_mult() {
        let a = Matrix::from_seed(3, 4, 2);
        let c = a.clone() * a;

        let expected_eval = &vec![
            vec![4.0, 4.0, 4.0, 4.0],
            vec![4.0, 4.0, 4.0, 4.0],
            vec![4.0, 4.0, 4.0, 4.0],
        ];

        assert!(Matrix::almost_equal(
            &c,
            &Matrix::from_mat(expected_eval),
            Some(0.001),
        ), "Failed mult-eval");
    }


    #[test]
    fn test_div() {
        let a = Matrix::from_seed(3, 4, 1.5);
        let c = a.clone() / a;

        let expected_eval = &vec![
            vec![1.0, 1.0, 1.0, 1.0],
            vec![1.0, 1.0, 1.0, 1.0],
            vec![1.0, 1.0, 1.0, 1.0],
        ];

        assert!(Matrix::almost_equal(
            &c,
            &Matrix::from_mat(expected_eval),
            Some(0.001),
        ), "Failed div-eval");
    }


    #[test]
    fn test_sub() {
        let a = Matrix::from_seed(3, 4, 1.5);
        let b = Matrix::ones(3, 4);
        let d = a.clone();
        let c = a - b;

        let expected_eval = &vec![
            vec![0.5,0.5,0.5,0.5 ],
            vec![0.5, 0.5, 0.5, 0.5],
            vec![0.5, 0.5, 0.5, 0.5],
        ];

        assert!(Matrix::almost_equal(
            &c,
            &Matrix::from_mat(expected_eval),
            Some(0.001),
        ), "Failed sub-eval");
    }


    #[test]
    fn test_transpose() {
        let mat = setup_matrix();
        let _res = mat.T();

        let expected_eval = &vec![
            vec![-1.0, -0.1, 0.1, 1.0, ],
            vec![-2.0, -0.2, 0.2, 2.0, ],
            vec![-3.0, -0.3, 0.3, 3.0, ],
        ];
        assert!(Matrix::almost_equal(
            &_res,
            &Matrix::from_mat(expected_eval),
            Some(0.001),
        ), "Failed transpose-eval");
    }

}
