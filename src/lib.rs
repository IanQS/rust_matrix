pub mod matrix;
pub mod raw_broadcasting;
pub mod matrix_ops;
pub mod math_ops;

#[cfg(test)]
mod tests {
    use crate::matrix::Matrix;

    #[test]
    fn it_works() {
        let a = Matrix::from_scalar(4);

        let b = Matrix::from_seed(5, 3, 5);
        println!("{:?}", b);
    }
}
