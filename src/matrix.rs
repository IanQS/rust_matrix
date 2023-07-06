/// Functions pertaining to matrix instantiation and top-level matrix operations e.g.
///     transpose, sum, sum_axis, dot, etc.
///
use crate::matrix::TensorType::{MatrixT, ScalarT, VectorT};
use crate::matrix_ops;

#[derive(Debug, Eq, PartialEq)]
pub enum Axis {
    First,
    Second,
}


#[derive(Debug, PartialEq)]
pub enum TensorType {
    ScalarT(f64),
    VectorT(Vec<f64>, bool),
    MatrixT(Vec<Vec<f64>>),
}

impl Clone for TensorType {
    fn clone(&self) -> Self {
        match self {
            ScalarT(data) => ScalarT(*data),
            VectorT(data, as_row) => {
                let mut cloned = vec![0.0; data.len()];
                cloned[..data.len()].copy_from_slice(&data[..]);
                VectorT(cloned, *as_row)
            }
            MatrixT(data) => {
                let num_rows = data.len();
                let num_cols = data[0].len();

                let mut cloned = vec![vec![0.0; num_cols]; num_rows];
                for i in 0..num_rows {
                    for j in 0..num_cols {
                        cloned[i][j] = data[i][j];
                    }
                }
                MatrixT(cloned)
            }
        }
    }
}


#[derive(Debug, Clone)]
pub struct Matrix {
    pub data: TensorType,
    pub shape: (usize, usize),
}

#[allow(non_snake_case)]
impl Matrix {
    pub fn from_scalar<T>(v: T) -> Self
        where T: Into<f64>
    {
        Self {
            data: ScalarT(v.into()),
            shape: (1, 1),
        }
    }

    #[must_use]
    pub fn from_vector<T>(v: &Vec<T>, is_row_vec: bool) -> Self
        where T: Into<f64> + Copy + Default
    {
        let mut container = vec![0.0; v.len()];
        for (i, &el) in v.iter().enumerate() {
            container[i] = el.into();
        }

        if is_row_vec {
            Self {
                data: VectorT(container, is_row_vec),
                shape: (1, v.len()),
            }
        } else {
            Self {
                data: VectorT(container, is_row_vec),
                shape: (v.len(), 1),
            }
        }
    }

    #[must_use]
    pub fn from_mat<T>(m: &Vec<Vec<T>>) -> Self
        where T: Into<f64> + Copy + Default
    {
        let s1 = m.len();
        let s2 = m[0].len();
        assert_ne!(s1, 1, "Called from_mat on something that should be row-vec. Use that instead");
        assert_ne!(s2, 1, "Called from_mat on something that should be col-vec. Use that instead");

        let mut container: Vec<Vec<f64>> = vec![vec![0.0; s2]; s1];
        for i in 0..s1 {
            for j in 0..s2 {
                container[i][j] = m[i][j].into();
            }
        }

        Self {
            data: MatrixT(container),
            shape: (s1, s2),
        }
    }

    #[must_use]
    pub fn from_seed<T>(rows: usize, cols: usize, fill: T) -> Self
        where T: Into<f64> + Default + Copy
    {
        if rows == 1 && cols == 1 {
            Self::from_scalar(fill)
        } else if rows > 1 && cols == 1 { // col-vec
            Self::from_vector(&vec![fill; rows], false)
        } else if rows == 1 && cols > 1 { // row-vec
            Self::from_vector(&vec![fill; cols], true)
        } else if rows > 1 && cols > 1 {
            let container = vec![vec![fill; cols]; rows];

            Self::from_mat(&container)
        } else {
            panic!("Invalid matrix dimensions: ({rows}, {cols})")
        }
    }


    #[must_use]
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self::from_seed(rows, cols, 0.0)
    }

    #[must_use]
    pub fn ones(rows: usize, cols: usize) -> Self {
        Self::from_seed(rows, cols, 1.0)
    }

    #[allow(non_snake_case)]
    // The in-place transpose
    pub fn _T(&mut self) {
        match self.data.clone() {
            ScalarT(_) => (), // Do nothing if is scalar
            VectorT(data, as_row) => {
                self.data = VectorT(data, !as_row);
                self.shape = (self.shape.1, self.shape.0);
            }
            MatrixT(data) => {
                self.data = MatrixT(Self::transpose_raw_matrix(&data, self.shape.1, self.shape.0));
                self.shape = (self.shape.1, self.shape.0);
            }
        }
    }

    pub(crate) fn transpose_raw_matrix<T>(incoming_mat: &[Vec<T>], curr_num_cols: usize, curr_num_rows: usize) -> Vec<Vec<f64>>
        where T: Into<f64> + Copy + Default
    {
        let mut mat = Vec::new();
        for col_i in 0..curr_num_cols {
            let mut vec: Vec<f64> = Vec::new();
            for row in incoming_mat.iter().take(curr_num_rows) {
                vec.push(row[col_i].into());
            }
            mat.push(vec);
        };
        mat
    }

    #[must_use]
    pub fn T(&self) -> Self{
        let mut to_ret = self.clone();
        to_ret._T();
        to_ret
    }

    #[must_use]
    pub fn dot(&self, other: &Self) -> Self{
        matrix_ops::_dot(self, other)
    }

    #[must_use]
    pub fn sum(&self) -> Self{
        matrix_ops::_sum(self)
    }

    #[must_use]
    pub fn sum_axis(&self, axis: &Axis) -> Self{
        matrix_ops::_sum_axis(self, axis)
    }


    /**
    From Numpy:
        https://numpy.org/doc/stable/user/basics.broadcasting.html#general-broadcasting-rules

    When operating on two arrays, NumPy compares their shapes element-wise. It starts with the
    trailing (i.e. rightmost) dimension and works its way left. Two dimensions are compatible when.
    Two dimensions are compatible when:
            - they are equal, or
            - one of them is 1.

    1) We check the trailing first.
    **/

    #[must_use]
    pub const fn is_broadcastable(lhs: &Self, rhs: &Self) -> bool {
        let is_eq_shape0 = lhs.shape.0 == rhs.shape.0;
        let is_eq_shape1 = lhs.shape.1 == rhs.shape.1;
        let is_broadcastable0 = lhs.shape.0 == 1 || rhs.shape.0 == 1;
        let is_broadcastable1 = lhs.shape.1 == 1 || rhs.shape.1 == 1;

        (is_eq_shape1 || is_broadcastable1) && (is_eq_shape0 || is_broadcastable0)
    }

    #[must_use]
    pub fn almost_equal(lhs: &Self, rhs: &Self, abs_tol: Option<f64>) -> bool {
        // We do NOT do broadcast equality checks.... that would be madness
        if lhs.shape != rhs.shape {
            return false;
        }
        let inner_abs_tol = abs_tol.unwrap_or(1e-3);
        match (lhs.data.clone(), rhs.data.clone()) {
            (ScalarT(l), ScalarT(r)) => {
                if f64::abs(f64::abs(l) - f64::abs(r)) > inner_abs_tol {
                    return false;
                }
            }
            (VectorT(l_vec, _), VectorT(r_vec, _)) => {
                for (l_scalar, r_scalar) in l_vec.iter().zip(r_vec.iter()) {
                    let diff = f64::abs(*l_scalar) - f64::abs(*r_scalar);
                    if f64::abs(diff) > inner_abs_tol {
                        return false;
                    }
                }
            }
            (MatrixT(m1), MatrixT(m2)) => {
                for (l_vec, r_vec) in m1.iter().zip(m2.iter()) {
                    for (l_scalar, r_scalar) in l_vec.iter().zip(r_vec.iter()) {
                        let diff = f64::abs(*l_scalar) - f64::abs(*r_scalar);
                        if f64::abs(diff) > inner_abs_tol {
                            return false;
                        }
                    }
                }
            }
            _ => {
                return false;
            }
        }
        true
    }
}
impl PartialEq<Self> for Matrix {
    fn eq(&self, other: &Self) -> bool {
        if self.shape != other.shape {
            return false;
        }
        match (self.data.clone(), other.data.clone()) {
            (ScalarT(s1), ScalarT(s2)) => {
                s1 == s2
            }
            (VectorT(v1, _), VectorT(v2, _)) => {
                if v1.len() != v2.len() {
                    return false;
                }
                for i in 0..v1.len() {
                    if v1[i] != v2[i] {
                        return false;
                    }
                }
                true
            }
            (MatrixT(m1), MatrixT(m2)) => {
                if m1.len() != m2.len() {
                    return false;
                }
                for i in 0..m1.len() {
                    let v1 = m1[i].clone();
                    let v2 = m2[i].clone();
                    if v1.len() != v2.len() {
                        return false;
                    }
                    for i in 0..v1.len() {
                        if v1[i] != v2[i] {
                            return false;
                        }
                    }
                }
                true
            }
            _ => false
        }
    }
}

impl Eq for Matrix {}

