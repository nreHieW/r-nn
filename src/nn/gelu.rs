use std::f32::consts::PI;

use crate::core::{Tensor, Value};
use crate::nn::tanh::Tanh;

pub struct GELU {
    tanh: Tanh,
}

impl GELU {
    pub fn new() -> Self {
        GELU { tanh: Tanh::new() }
    }

    pub fn forward(&self, inputs: &Tensor) -> Tensor {
        0.5 * &(inputs
            * &(1.0
                + &self
                    .tanh
                    .forward(&((2.0 / PI).sqrt() * &(inputs + &(0.044715 * &inputs.pow(3.0)))))))
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
    fn test_gelu_shape() {
        let gelu = GELU::new();
        let inputs = Tensor::randn(vec![3]);
        let outputs = gelu.forward(&inputs);
        assert_eq!(outputs.shape, vec![3]);
    }

    #[test]
    fn test_gelu() {
        let gelu = GELU::new();
        let inputs = Tensor {
            items: vec![
                Value::new(10.0),
                Value::new(-1.0),
                Value::new(2.0),
                Value::new(1.28),
            ],
            shape: vec![3],
        };
        let outputs = gelu.forward(&inputs);
        let ans = vec![10.0, -0.1588, 1.9546, 1.1514]; // The approximiate values are taken from PyTorch
        for i in 0..3 {
            assert!(float_eq(outputs.get_item(vec![i]).item(), ans[i]));
        }
    }
}
