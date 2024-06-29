use r_nn::{
    core::{Tensor, Value},
    nn::*,
};

struct MLP {
    linear_layers: Vec<Linear>,
}

impl MLP {
    fn new(input_dim: usize, hidden_dims: Vec<usize>, output_dim: usize, use_bias: bool) -> Self {
        let mut layers = vec![];
        let mut in_features = input_dim;
        for &out_features in hidden_dims.iter() {
            layers.push(Linear::new(in_features, out_features, use_bias));
            in_features = out_features;
        }
        layers.push(Linear::new(in_features, output_dim, use_bias));
        Self {
            linear_layers: layers,
        }
    }

    fn forward(&self, inputs: &Tensor) -> Tensor {
        // No new Value object is created, the pointers are cloned
        let mut x = inputs.clone();

        for layer in self.linear_layers.iter() {
            x = layer.forward(&x);
            // x = x.relu(); This causes a problem if the parameters are unlucky, everything is zeroed out
            // tanh
            x = &(&(2 * &x).exp() - 1) / &(&(2 * &x).exp() + 1);
        }
        x
    }

    fn parameters(&self) -> Vec<&Value> {
        let mut params = vec![];
        for layer in self.linear_layers.iter() {
            params.extend(layer.parameters());
        }
        params
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    #[test]
    fn simple_mlp_nn() {
        let x = Tensor::randn(vec![3]);
        let mlp = MLP::new(3, vec![4, 4], 1, false);
        let y = mlp.forward(&x);
        assert_eq!(y.shape, vec![1]);
    }

    #[test]
    fn learning_mlp_nn() {
        let mut start_loss = None;
        let mut end_loss = None;
        let xs = vec![
            2.0, 3.0, -1.0, 3.0, -1.0, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, -1.0,
        ];
        let xs = Tensor::new(xs, vec![4, 3]);
        let ys = Tensor::new(vec![1.0, -1.0, -1.0, 1.0], vec![4, 1]);
        let mlp = MLP::new(3, vec![], 1, false);
        let lr = -0.001;
        for epoch in 0..20 {
            let y_pred = mlp.forward(&xs);
            // println!("y_pred: {}", y_pred);
            let loss = &ys - &y_pred;
            let loss = loss.pow(2.0).sum(-1);
            for param in mlp.parameters() {
                param.zero_grad();
            }
            loss.backward();
            for param in mlp.parameters() {
                assert!(param.data.borrow().grad.is_some());
                param.update(lr);
            }
            assert_eq!(loss.shape, vec![1]);
            println!("Epoch: {}, Loss: {}", epoch, loss.get_item(vec![0]).item());
            if start_loss.is_none() {
                start_loss = Some(loss.get_item(vec![0]).item());
            }
            end_loss = Some(loss.get_item(vec![0]).item());
        }
        println!(
            "Start loss: {}, End loss: {}",
            start_loss.unwrap(),
            end_loss.unwrap()
        );
        assert!(start_loss.unwrap() > end_loss.unwrap());
    }
}
