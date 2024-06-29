#![allow(dead_code)]
use crate::core::{Tensor, Value};

pub struct Tanh {}

impl Tanh {
    pub fn new() -> Tanh {
        Tanh {}
    }

    pub fn forward(&self, inputs: &Tensor) -> Tensor {
        // let o = &(&(2 * &n).exp() - 1) / &(&(2 * &n).exp() + 1);
        let exp = (inputs * 2).exp();
        let out = &(&exp - 1) / &(&exp + 1);
        out
    }

    pub fn parameters(&self) -> Vec<&Value> {
        vec![]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Value;
    fn float_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < 0.0001
    }

    #[test]
    fn test_tanh_shape() {
        let tanh = Tanh::new();
        let inputs = Tensor::randn(vec![3]);
        let outputs = tanh.forward(&inputs);
        assert_eq!(outputs.shape, vec![3]);
    }

    #[test]
    fn test_tanh() {
        let tanh = Tanh::new();
        let inputs = Tensor {
            items: vec![Value::new(10.0), Value::new(-11.0), Value::new(2.0)],
            shape: vec![3],
        };
        let outputs = tanh.forward(&inputs);
        let ans = vec![1.0, -1.0, 0.9640];
        for i in 0..3 {
            assert!(float_eq(outputs.get_item(vec![i]).item(), ans[i]));
        }
    }
}
