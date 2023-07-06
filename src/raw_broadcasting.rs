//////////////////////////////
// Raw Broadcasting code i.e. code that is external to our matrix.
//      If you create a new matrix interface, you can just use this as
//      the base of your code to implement broadcasting.
//////////////////////////////

#[derive(Copy, Clone)]
pub enum Op {
    Sub,
    Add,
    Mul,
    Div,
}


pub(crate) mod broadcasting_ops {
    use rayon::prelude::*;
    use rayon::iter::IntoParallelRefIterator;
    use crate::raw_broadcasting::Op;


    //////////////////////////////
    // Same-Cardinality Ops
    //////////////////////////////

    // Scalar-X
    pub fn _scalar_scalar(lhs: f64, rhs: f64, op: Op) -> f64 {
        match op {
            Op::Add => lhs + rhs,
            Op::Sub => lhs - rhs,
            Op::Mul => lhs * rhs,
            Op::Div => lhs / rhs
        }
    }

    pub fn _vector_vector(lhs: &Vec<f64>, rhs: &Vec<f64>, op: Op) -> Vec<f64> {
        lhs.into_par_iter()
            .zip(rhs.into_par_iter())
            .map(
                |(l_val, r_val)|
                    _scalar_scalar(*l_val, *r_val, op)
            )
            .collect()
    }


    pub fn _matrix_matrix(
        lhs: &Vec<Vec<f64>>,
        rhs: &Vec<Vec<f64>>,
        op: Op,
    ) -> Vec<Vec<f64>> {
        lhs.into_par_iter()
            .zip(rhs.into_par_iter())
            .map(|(l_vec, r_vec)| _vector_vector(l_vec, r_vec, op))
            .collect()
    }

    //////////////////////////////
    // Broadcasts
    //////////////////////////////

    pub fn _vector_scalar(lhs: &Vec<f64>, rhs: f64, op: Op) -> Vec<f64> {
        lhs
            .par_iter()
            .map(|x| _scalar_scalar(*x, rhs, op))
            .collect()
    }

    pub fn _scalar_vector(lhs: f64, rhs: &Vec<f64>, op: Op) -> Vec<f64> {
        return match op {
            Op::Sub => {
                let distributed = rhs.par_iter().map(
                    |x| -x
                ).collect();
                _vector_scalar(&distributed, lhs, Op::Add)
            }
            Op::Div => {
                let distributed = rhs.par_iter().map(
                    |x| lhs / x
                ).collect();
                distributed
            }
            _ => { _vector_scalar(rhs, lhs, op) }
        };
    }

    pub fn _scalar_matrix(lhs: f64, rhs: &Vec<Vec<f64>>, op: Op) -> Vec<Vec<f64>> {
        rhs.into_par_iter()
            .map(
                |r_vec| _scalar_vector(lhs, r_vec, op)
            )
            .collect()
    }

    pub fn _matrix_scalar(lhs: &Vec<Vec<f64>>, rhs: f64, op: Op) -> Vec<Vec<f64>> {
        lhs.into_par_iter()
            .map(
                |l_vec| _vector_scalar(l_vec, rhs, op)
            )
            .collect()
    }


    pub fn _vector_matrix(lhs: &Vec<f64>, rhs: &Vec<Vec<f64>>, op: Op) -> Vec<Vec<f64>> {
        rhs.into_par_iter()
            .map(|r_vec| _vector_vector(lhs, r_vec, op))
            .collect()
    }

    pub fn _matrix_vector(lhs: &Vec<Vec<f64>>, rhs: &Vec<f64>, op: Op) -> Vec<Vec<f64>> {
        lhs.into_par_iter()
            .map(|l_vec| _vector_vector(l_vec, rhs, op))
            .collect()
    }

    pub fn almost_equal(
        lhs: &Vec<Vec<f64>>,
        rhs: &Vec<Vec<f64>>,
        tol: f64,
    ) -> bool {
        if lhs.len() != rhs.len() {
            return false;
        }
        for row_idx in 0..lhs.len() {
            if lhs[row_idx].len() != rhs[row_idx].len() {
                return false;
            }
            for col_idx in 0..lhs[row_idx].len() {
                if f64::abs(lhs[row_idx][col_idx] - rhs[row_idx][col_idx]) > tol {
                    return false;
                }
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use std::vec;
    use crate::raw_broadcasting::broadcasting_ops::*;
    use crate::raw_broadcasting::Op;

    fn setup_vector1() -> Vec<f64> {
        vec![1.0, 2.0, 3.0]
    }

    fn setup_vector2() -> Vec<f64> {
        vec![4.0, 5.0, 6.0]
    }

    fn setup_mat1() -> Vec<Vec<f64>> {
        vec![setup_vector1(), setup_vector2()]
    }

    fn setup_mat2() -> Vec<Vec<f64>> {
        vec![setup_vector2(), setup_vector1()]
    }

    #[test]
    fn test_scalar() {
        let scalar = 3.0;

        // test scalar-scalar
        assert_eq!(
            _scalar_scalar(scalar, scalar, Op::Sub),
            0.0
        );
        assert_eq!(
            _scalar_scalar(scalar, scalar, Op::Add),
            6.0
        );
        assert_eq!(
            _scalar_scalar(scalar, scalar, Op::Mul),
            9.0
        );

        // test scalar-vec

        let vec = setup_vector1(); // 1, 2, 3

        assert_eq!(
            _scalar_vector(scalar, &vec, Op::Sub),
            vec![2.0, 1.0, 0.0]
        );
        assert_eq!(
            _scalar_vector(scalar, &vec, Op::Add),
            vec![4.0, 5.0, 6.0]
        );
        assert_eq!(
            _scalar_vector(scalar, &vec, Op::Mul),
            vec![3.0, 6.0, 9.0]
        );

        // test scalar-mat
        let mat = setup_mat1(); // [1,2,3], [4,5,6]

        assert_eq!(
            _scalar_matrix(scalar, &mat, Op::Sub),
            vec![vec![2.0, 1.0, 0.0], vec![-1.0, -2.0, -3.0]]
        );

        assert_eq!(
            _scalar_matrix(scalar, &mat, Op::Add),
            vec![vec![4.0, 5.0, 6.0], vec![7.0, 8.0, 9.0]]
        );

        assert_eq!(
            _scalar_matrix(scalar, &mat, Op::Mul),
            vec![vec![3.0, 6.0, 9.0], vec![12.0, 15.0, 18.0]]
        );
    }

    #[test]
    fn test_vector() {
        let vec = setup_vector2(); // [4,5,6]
        let scalar = 3.0;

        // test vec-scalar
        assert_eq!(
            _vector_scalar(&vec, scalar, Op::Sub),
            vec![1.0, 2.0, 3.0]
        );
        assert_eq!(
            _vector_scalar(&vec, scalar, Op::Add),
            vec![7.0, 8.0, 9.0]
        );
        assert_eq!(
            _vector_scalar(&vec, scalar, Op::Mul),
            vec![12.0, 15.0, 18.0]
        );

        // test vec-vec
        let vec2 = setup_vector1(); // [1,2,3]
        assert_eq!(
            _vector_vector(&vec, &vec2, Op::Sub),
            vec![3.0, 3.0, 3.0]
        );
        assert_eq!(
            _vector_vector(&vec, &vec2, Op::Add),
            vec![5.0, 7.0, 9.0]
        );
        assert_eq!(
            _vector_vector(&vec, &vec2, Op::Mul),
            vec![4.0, 10.0, 18.0]
        );

        let mat = setup_mat2(); //[4,5,6] [1,2,3]
        assert_eq!(
            _vector_matrix(&vec, &mat, Op::Sub),
            vec![vec![0.0, 0.0, 0.0], vec![3.0, 3.0, 3.0]]
        );
        assert_eq!(
            _vector_matrix(&vec, &mat, Op::Add),
            vec![vec![8.0, 10.0, 12.0], vec![5.0, 7.0, 9.0]]
        );
        assert_eq!(
            _vector_matrix(&vec, &mat, Op::Mul),
            vec![vec![16.0, 25.0, 36.0], vec![4.0, 10.0, 18.0]]
        );
    }

    #[test]
    fn test_matrix() {
        let mat = setup_mat2();  // [4,5,6] [1,2,3]
        let scalar = 5.0;

        // Mat-scalar
        assert_eq!(
            _matrix_scalar(&mat, scalar, Op::Sub),
            vec![vec![-1.0, 0.0, 1.0], vec![-4.0, -3.0, -2.0]]
        );
        assert_eq!(
            _matrix_scalar(&mat, scalar, Op::Add),
            vec![vec![9.0, 10.0, 11.0], vec![6.0, 7.0, 8.0]]
        );
        assert_eq!(
            _matrix_scalar(&mat, scalar, Op::Mul),
            vec![vec![20.0, 25.0, 30.0], vec![5.0, 10.0, 15.0]]
        );

        // Mat-vector

        let vec = setup_vector1(); //[1,2,3]
        assert_eq!(
            _matrix_vector(&mat, &vec, Op::Sub),
            vec![vec![3.0, 3.0, 3.0], vec![0.0, 0.0, 0.0]]
        );
        assert_eq!(
            _matrix_vector(&mat, &vec, Op::Add),
            vec![vec![5.0, 7.0, 9.0], vec![2.0, 4.0, 6.0]]
        );
        assert_eq!(
            _matrix_vector(&mat, &vec, Op::Mul),
            vec![vec![4.0, 10.0, 18.0], vec![1.0, 4.0, 9.0]]
        );

        //Mat - Mat
        let mat2 = setup_mat2();
        assert_eq!(
            _matrix_matrix(&mat, &mat2, Op::Sub),
            vec![vec![0.0, 0.0, 0.0], vec![0.0, 0.0, 0.0]]
        );
        assert_eq!(
            _matrix_matrix(&mat, &mat2, Op::Add),
            vec![vec![8.0, 10.0, 12.0], vec![2.0, 4.0, 6.0]]
        );
        assert_eq!(
            _matrix_matrix(&mat, &mat2, Op::Mul),
            vec![vec![16.0, 25.0, 36.0], vec![1.0, 4.0, 9.0]]
        );
    }
}