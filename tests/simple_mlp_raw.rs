use r_nn::core::Value;
use rand::distributions::{Distribution, Uniform};

struct Neuron {
    weights: Vec<Value>,
    bias: Value,
}

impl Neuron {
    fn new(n_in: usize) -> Self {
        let mut rng = rand::thread_rng();
        let between = Uniform::from(-1.0..1.0);
        let weights = (0..n_in)
            .map(|_| Value::new(between.sample(&mut rng)))
            .collect();
        let bias = Value::new(between.sample(&mut rng));
        Self { weights, bias }
    }

    fn forward(&self, inputs: &Vec<Value>) -> Value {
        let out: Value = &self
            .weights
            .iter()
            .zip(inputs.iter())
            .map(|(w, x)| w * x)
            .sum::<Value>()
            + &self.bias;
        &(&(2 * &out).exp() - 1) / &(&(2 * &out).exp() + 1) // tanh
    }

    fn parameters(&self) -> Vec<&Value> {
        let mut params = self.weights.iter().collect::<Vec<_>>();
        params.push(&self.bias);
        params
    }
}

struct Layer {
    neurons: Vec<Neuron>,
}

impl Layer {
    fn new(n_in: usize, n_out: usize) -> Self {
        let neurons = (0..n_out).map(|_| Neuron::new(n_in)).collect();
        Self { neurons }
    }

    fn forward(&self, inputs: &Vec<Value>) -> Vec<Value> {
        self.neurons.iter().map(|n| n.forward(inputs)).collect()
    }

    fn parameters(&self) -> Vec<&Value> {
        self.neurons.iter().flat_map(|n| n.parameters()).collect()
    }
}

struct MLP {
    layers: Vec<Layer>,
}

impl MLP {
    fn new(n_in: usize, n_out: Vec<usize>) -> Self {
        let n_layers = n_out.len();
        let layers = (0..n_layers)
            .map(|i| {
                let n_in = if i == 0 { n_in } else { n_out[i - 1] };
                let n_out = n_out[i];
                Layer::new(n_in, n_out)
            })
            .collect();
        Self { layers }
    }

    fn forward(&self, inputs: &Vec<Value>) -> Vec<Value> {
        let mut outputs = inputs.to_vec();
        for layer in &self.layers {
            outputs = layer.forward(&outputs);
        }
        outputs
    }

    fn parameters(&self) -> Vec<&Value> {
        self.layers.iter().flat_map(|l| l.parameters()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_mlp() {
        let x = vec![2.0, 3.0, -1.0]
            .iter()
            .map(|&v| Value::new(v))
            .collect::<Vec<Value>>();
        let mlp = MLP::new(3, vec![4, 4, 1]);
        let y = mlp.forward(&x);
        assert_eq!(y.len(), 1);
        dbg!(y[0].item());
    }

    #[test]
    fn computational_graph() {
        let n = Neuron::new(3);
        let x = vec![2.0, 3.0, -1.0]
            .iter()
            .map(|&v| Value::new(v))
            .collect::<Vec<Value>>();
        let y = n.forward(&x);
        y.backward();
        for param in n.parameters() {
            assert_ne!(param.data.borrow().grad, 0.0);
        }

        let l = Layer::new(3, 1);
        let y = &l.forward(&x)[0];
        y.backward();
        for param in l.parameters() {
            assert_ne!(param.data.borrow().grad, 0.0);
        }

        let mlp = MLP::new(3, vec![1]);
        let y = mlp.forward(&x)[0].clone();
        y.backward();
        for param in mlp.parameters() {
            assert_ne!(param.data.borrow().grad, 0.0);
        }
    }

    #[test]
    fn learning() {
        let mut start_loss = None;
        let mut end_loss = None;
        let xs = vec![
            vec![2.0, 3.0, -1.0],
            vec![3.0, -1.0, 0.5],
            vec![0.5, 1.0, 1.0],
            vec![1.0, 1.0, -1.0],
        ]
        .iter()
        .map(|v| v.iter().map(|&v| Value::new(v)).collect::<Vec<Value>>())
        .collect::<Vec<Vec<Value>>>();
        let ys = vec![1.0, -1.0, -1.0, 1.0]
            .iter()
            .map(|&v| Value::new(v))
            .collect::<Vec<Value>>();
        let mlp = MLP::new(3, vec![4, 4, 1]);
        let lr = Value::new(-0.1);
        for epoch in 0..20 {
            let y_pred = xs
                .iter()
                .map(|x| mlp.forward(x)[0].clone())
                .collect::<Vec<Value>>();

            let losses = ys
                .iter()
                .zip(y_pred.iter())
                .map(|(ygt, yout)| (ygt - yout).pow(&Value::new(2.0)))
                .collect::<Vec<Value>>();
            let loss = losses.iter().sum::<Value>();
            for param in mlp.parameters() {
                param.zero_grad();
            }

            loss.backward();

            for param in mlp.parameters() {
                assert_ne!(param.data.borrow().grad, 0.0);
                param.update(&lr);
            }

            println!("Epoch: {}, Loss: {}", epoch, loss.item());

            if start_loss.is_none() {
                start_loss = Some(loss.item());
            }

            end_loss = Some(loss.item());
        }
        println!(
            "Start loss: {}, End loss: {}",
            start_loss.unwrap(),
            end_loss.unwrap()
        );
        assert!(start_loss.unwrap() > end_loss.unwrap());
    }
}
