use crate::core::{Tensor, Value};

// // 1D Layer Normalization
// pub struct LayerNorm {
//     weight: Tensor,
//     bias: Tensor,
// }

pub struct LayerNorm {
    weight: Tensor,
    bias: Tensor,
    dim: usize,
}

impl LayerNorm {
    pub fn new(dim: usize) -> Self {
        let weight = &Tensor::zeros(vec![dim]) + 1.0;
        let bias = Tensor::zeros(vec![dim]);
        Self { weight, bias, dim }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let num_dims = x.shape.len();
        assert!(
            num_dims >= 2,
            "Input tensor must have at least 2 dimensions"
        );

        let last_dim = num_dims - 1;
        let s = x.sum(last_dim as i32);
        let denom = x.shape[last_dim] as f32;
        let mut mean_shape = x.shape.clone();
        mean_shape[last_dim] = 1;
        let mean = (&s / denom).reshape(mean_shape.clone());
        let variance = (&(x - &mean).pow(2.0).sum(last_dim as i32) / denom).reshape(mean_shape);
        let normalized = &(x - &mean) / &(&variance + 1e-5).pow(0.5);
        let mut broadcast_shape = vec![1; num_dims];
        broadcast_shape[last_dim] = self.dim;
        let weight_broadcast = self.weight.reshape(broadcast_shape.clone());
        let bias_broadcast = self.bias.reshape(broadcast_shape);

        &(&normalized * &weight_broadcast) + &bias_broadcast
    }

    pub fn parameters(&self) -> Vec<&Value> {
        let mut params = self.weight.items.iter().collect::<Vec<_>>();
        params.extend(self.bias.items.iter());
        params
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Tensor;
    fn float_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-4
    }

    #[test]
    fn test_layernorm() {
        let layernorm = LayerNorm::new(3);
        let inputs = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let outputs = layernorm.forward(&inputs);
        assert_eq!(outputs.shape, vec![2, 3]);
        let ans = Tensor::from_vec(vec![-1.2247, 0.0, 1.2247, -1.2247, 0.0, 1.2247], vec![2, 3]);
        for i in 0..outputs.shape[0] {
            for j in 0..outputs.shape[1] {
                assert!(float_eq(
                    outputs.get_item(vec![i, j]).item(),
                    ans.get_item(vec![i, j]).item()
                ));
            }
        }
    }
}
