#![allow(dead_code)]
use crate::core::{Tensor, Value};

pub struct Linear {
    pub weights: Tensor,
    pub bias: Tensor,
    pub n_dim_in: usize,
    pub n_dim_out: usize,
    pub use_bias: bool,
}

impl Linear {
    pub fn new(n_dim_in: usize, n_dim_out: usize, use_bias: bool) -> Self {
        let weights = Tensor::randn(vec![n_dim_in, n_dim_out], true);
        if use_bias {
            let bias = Tensor::randn(vec![n_dim_out], true);
            Self {
                weights,
                bias,
                n_dim_in,
                n_dim_out,
                use_bias,
            }
        } else {
            Self {
                weights,
                bias: Tensor::zeros(vec![1], true),
                n_dim_in,
                n_dim_out,
                use_bias,
            }
        }
    }

    pub fn forward(&self, inputs: &Tensor) -> Tensor {
        if self.use_bias {
            &inputs.matmul(&self.weights) + &self.bias
        } else {
            inputs.matmul(&self.weights)
        }
    }

    pub fn parameters(&self) -> Vec<&Value> {
        let mut params = self.weights.items.iter().collect::<Vec<_>>();
        if self.use_bias {
            params.extend(self.bias.items.iter());
        }
        params
    }

    pub fn no_grad(&mut self) {
        self.weights.items.iter_mut().for_each(|p| p.no_grad());
        if self.use_bias {
            self.bias.items.iter_mut().for_each(|p| p.no_grad());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Tensor;

    #[test]
    fn test_linear_shape() {
        let linear = Linear::new(3, 2, true);
        let inputs = Tensor::randn(vec![3], true);
        let outputs = linear.forward(&inputs);
        assert_eq!(outputs.shape, vec![2]);
    }

    #[test]
    fn test_linear_computational_graph() {
        let linear = Linear::new(3, 2, true);
        let inputs = Tensor::randn(vec![3], true);
        let outputs = linear.forward(&inputs);
        let loss = outputs.sum(0);
        loss.backward();
        let params = linear.parameters();
        assert_eq!(params.len(), 3 * 2 + 2);
        for p in params {
            assert!(p.data.borrow().grad.is_some());
        }
    }
}
