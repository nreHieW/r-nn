#![allow(dead_code)]
use crate::core::{Tensor, Value};

pub struct ReLU {}

impl ReLU {
    pub fn new() -> ReLU {
        ReLU {}
    }

    pub fn forward(&self, inputs: &Tensor) -> Tensor {
        inputs.relu()
    }

    pub fn parameters(&self) -> Vec<&Value> {
        vec![]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{Tensor, Value};

    #[test]
    fn test_relu_shape() {
        let relu = ReLU::new();
        let inputs = Tensor::randn(vec![3], false);
        let outputs = relu.forward(&inputs);
        assert_eq!(outputs.shape, vec![3]);
    }

    #[test]
    fn test_relu() {
        let relu = ReLU::new();
        let inputs = Tensor {
            items: vec![
                Value::new(1.0, true),
                Value::new(-1.0, true),
                Value::new(0.0, true),
            ],
            shape: vec![3],
        };
        let outputs = relu.forward(&inputs);
        let ans = vec![1.0, 0.0, 0.0];
        for i in 0..3 {
            assert_eq!(outputs.get_item(vec![i]).item(), ans[i]);
        }
    }
}
