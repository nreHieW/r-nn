// https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html
use r_nn::{
    core::{Tensor, Value},
    nn::{Embedding, Linear, Tanh},
};
use rand::Rng;

struct RNN {
    _input_size: usize,
    hidden_size: usize,
    _output_size: usize,
    act: Tanh,
    i2h: Embedding,
    h2h: Linear,
    h2o: Linear,
}

impl RNN {
    fn new(input_size: usize, hidden_size: usize, output_size: usize) -> Self {
        let i2h = Embedding::new(input_size, hidden_size);
        let h2h = Linear::new(hidden_size, hidden_size, false);
        let h2o = Linear::new(hidden_size, output_size, false);
        Self {
            _input_size: input_size,
            hidden_size,
            _output_size: output_size,
            act: Tanh::new(),
            i2h,
            h2h,
            h2o,
        }
    }

    fn forward(&self, input_idx: usize, hidden: &Tensor) -> (Tensor, Tensor) {
        let input = self.i2h.forward(&Tensor {
            items: vec![Value::new(input_idx as f32)],
            shape: vec![1],
        });
        let x1 = self.i2h.forward(&input);
        let x2 = self.h2h.forward(hidden);
        let hidden = self.act.forward(&(&x1 + &x2));
        let output = self.h2o.forward(&hidden).softmax(1).ln(); // LogSoftmax
        (output, hidden)
    }
    fn parameters(&self) -> Vec<&Value> {
        let mut params = self.i2h.parameters();
        params.extend(self.h2h.parameters());
        params.extend(self.h2o.parameters());
        params
    }
    fn init_hidden(&self) -> Tensor {
        Tensor::zeros(vec![1, self.hidden_size])
    }
}
fn create_one_hot_label(label: u8) -> Vec<u8> {
    let mut one_hot = vec![0; 10];
    one_hot[label as usize] = 1;
    one_hot
}

fn get_dataset() -> (Vec<Vec<i32>>, Vec<Tensor>) {
    let mut rng = rand::thread_rng();
    let ds_size = 100;
    let mut data = Vec::with_capacity(ds_size);
    let mut labels = Vec::with_capacity(ds_size);
    let seq_len = 5;
    let num_classes = 3;
    for _ in 0..ds_size {
        let item: Vec<i32> = (0..seq_len).map(|_| rng.gen_range(0..10)).collect();
        let label = rng.gen_range(0..num_classes);
        data.push(item);
        labels.push(Tensor::from_vec(
            create_one_hot_label(label),
            vec![1, num_classes as usize],
        ));
    }
    (data, labels)
}
fn nll_loss(y_pred: &Tensor, y_true: &Tensor) -> Tensor {
    // batch size 1
    let n = y_pred.shape[0] as f32; // 1
    let y_true = y_true.to_flattened_vec()[0] as usize;
    let items = &y_pred.slice(&vec![0..1, y_true..y_true + 1]) * -1.0;

    &items.sum(-1) / n
}

fn main() {
    let rnn = RNN::new(10, 20, 3);
    let (data, labels) = get_dataset();
    let optim = r_nn::nn::optim::base_sgd::BaseSGD::new(rnn.parameters(), 0.01);

    for epoch in 0..100 {
        let mut epoch_loss = 0.0;
        for (seq, label) in data.iter().zip(labels.iter()) {
            let mut hidden = rnn.init_hidden();

            let mut output_tensor = Tensor::zeros(vec![1, 3]);
            for item in seq {
                let output = rnn.forward(*item as usize, &hidden);
                output_tensor = output.0;
                hidden = output.1;
            }
            let loss = nll_loss(&output_tensor, label);
            optim.zero_grad();
            loss.backward();
            optim.step();
            epoch_loss += loss.get_item(vec![0]).item();
        }
        println!("Epoch: {}, Loss: {}", epoch, epoch_loss);
    }
}

#[cfg(test)]
mod test {
    use super::*;

    fn float_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-5
    }

    #[test]
    fn test_nll_loss() {
        {
            let y_pred = Tensor::new(vec![0.8, 0.1, 0.1], vec![1, 3]);
            let y_true = Tensor::new(vec![0.0], vec![1]);
            let loss = nll_loss(&y_pred.ln(), &y_true);
            assert!(float_eq(loss.get_item(vec![0]).item(), 0.223143));
        }
    }
}
