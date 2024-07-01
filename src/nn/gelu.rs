use crate::core::{Tensor, Value};

pub struct GELU {}

impl GELU {
    pub fn new() -> Self {
        GELU {}
    }

    pub fn forward(&self, inputs: &Tensor) -> Tensor {
        let inner_term = &(&(0.044715 * inputs) * inputs);
        let tanh_arg = &(&(inputs * 0.7978845608) * &(1.0 + inner_term));
        let tanh_result = &(1.0 + &tanh_arg.tanh());
        let result = &(inputs * 0.5) * tanh_result;

        result
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
        let inputs = Tensor::randn(vec![3], true);
        let outputs = gelu.forward(&inputs);
        assert_eq!(outputs.shape, vec![3]);
    }

    #[test]
    fn test_gelu() {
        let gelu = GELU::new();
        let inputs = Tensor {
            items: vec![
                Value::new(10.0, true),
                Value::new(-1.0, true),
                Value::new(2.0, true),
                Value::new(1.28, true),
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
