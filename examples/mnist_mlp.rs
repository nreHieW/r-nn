// #[!allow(unused)]
use image::{self};
use r_nn::core::{Tensor, Value};
use r_nn::nn::{LayerNorm, Linear, ReLU};
use std::path::Path;
use std::time::Instant;
use std::{fs, vec};

#[derive(Debug, Clone)]
struct MnistDataset {
    images: Vec<Vec<u8>>,
    labels: Vec<u8>,
    curr_idx: usize,
    batch_size: usize,
}

fn create_one_hot_label(label: u8) -> Vec<u8> {
    let mut one_hot = vec![0; 10];
    one_hot[label as usize] = 1;
    one_hot
}

impl Iterator for MnistDataset {
    type Item = (Tensor, Tensor);
    fn next(&mut self) -> Option<Self::Item> {
        let mut images = Vec::new();
        let mut labels = Vec::with_capacity(self.batch_size);
        for _ in 0..self.batch_size {
            if self.curr_idx < self.images.len() {
                images.extend(self.images[self.curr_idx].clone());
                let l = self.labels[self.curr_idx];

                labels.push(create_one_hot_label(l));
                self.curr_idx += 1;
            } else {
                break;
            }
        }
        if images.is_empty() {
            None
        } else {
            Some((
                Tensor::from_vec(images, vec![self.batch_size, 28 * 28], true),
                Tensor::from_vec(labels.concat(), vec![self.batch_size, 10], true),
            ))
        }
    }
}

impl MnistDataset {
    fn new(images: Vec<Vec<u8>>, labels: Vec<u8>, batch_size: usize) -> Self {
        assert_eq!(images.len(), labels.len());
        Self {
            images,
            labels,
            curr_idx: 0,
            batch_size: batch_size,
        }
    }
    fn len(&self) -> usize {
        self.images.len()
    }

    fn num_batches(&self) -> usize {
        self.images.len() / self.batch_size
    }

    fn _display_image(&self, idx: usize) {
        let img = &self.images[idx];
        for i in 0..28 {
            for j in 0..28 {
                let idx = i * 28 + j;
                if img[idx] > 128 {
                    print!("X");
                } else {
                    print!(" ");
                }
            }
            println!();
        }
        println!("Label: {:?}", create_one_hot_label(self.labels[idx]))
    }
}

fn load_image(path: &Path) -> Vec<u8> {
    let img = image::open(path).expect("Failed to open image");
    img.to_luma8().into_raw()
}

fn load_dataset() -> (MnistDataset, MnistDataset) {
    let train_paths = fs::read_dir("data/mnist_train").unwrap().take(100);
    let test_paths = fs::read_dir("data/mnist_test").unwrap().take(1);

    let mut train_image = Vec::new();
    let mut train_label = Vec::new();

    for p in train_paths {
        let path = p.unwrap().path();
        let label = path
            .to_str()
            .expect("Failed to convert path to string")
            .split("/")
            .last()
            .unwrap()
            .split("_")
            .next()
            .unwrap()
            .parse::<u8>()
            .expect("Failed to parse label");
        let img = load_image(&path);
        train_image.push(img);
        train_label.push(label);
    }

    let mut test_image = Vec::new();
    let mut test_label = Vec::new();

    for p in test_paths {
        let path = p.unwrap().path();
        let label = path
            .to_str()
            .expect("Failed to convert path to string")
            .split("/")
            .last()
            .unwrap()
            .split("_")
            .next()
            .unwrap()
            .parse::<u8>()
            .expect("Failed to parse label");
        let img = load_image(&path);
        test_image.push(img);
        test_label.push(label);
    }

    (
        MnistDataset::new(train_image, train_label, 4),
        MnistDataset::new(test_image, test_label, 1),
    )
}

// MODEL
struct Layer {
    linear1: Linear,
    act: ReLU,
    norm1: LayerNorm,
    linear2: Linear,
}

impl Layer {
    fn new(in_features: usize, out_features: usize) -> Self {
        let linear1 = Linear::new(in_features, in_features / 2, false);
        let norm1 = LayerNorm::new(in_features / 2);
        let linear2 = Linear::new(in_features / 2, out_features, false);
        let act = ReLU::new();
        Self {
            linear1,
            norm1,
            act,
            linear2,
        }
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let x = self.linear1.forward(x);
        let x = self.norm1.forward(&x);
        let x = self.act.forward(&x);
        let x = self.linear2.forward(&x);
        x
    }

    fn parameters(&self) -> Vec<&Value> {
        let mut params = self.linear1.parameters();
        params.extend(self.norm1.parameters());
        params.extend(self.linear2.parameters());
        params
    }
}

struct MLP {
    layers: Vec<Layer>,
}

impl MLP {
    fn new(in_features: usize, hidden_features: Vec<usize>, out_features: usize) -> Self {
        let mut in_feat = in_features;
        let mut layers: Vec<Layer> = hidden_features
            .iter()
            .map(|&out_feat| {
                let layer = Layer::new(in_feat, out_feat);
                in_feat = out_feat;
                layer
            })
            .collect();
        let output_layer = Layer::new(in_feat, out_features);
        layers.push(output_layer);
        Self { layers }
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let mut x = x.clone();
        for layer in &self.layers {
            x = layer.forward(&x);
        }
        x // Softmax done in loss
    }

    fn parameters(&self) -> Vec<&Value> {
        self.layers.iter().flat_map(|l| l.parameters()).collect()
    }
}

fn cross_entropy(y_pred: &Tensor, y_true: &Tensor) -> Tensor {
    let n = y_pred.shape[0] as f32;
    let loss = &-y_true * &y_pred.softmax(1).ln();
    &loss.sum(-1) / n
}

fn accuracy(y_pred: &Tensor, y_true: &Tensor) -> i32 {
    // Single batch
    let y_pred = y_pred.argmax();
    let y_true = y_true.get_item(vec![0]).item() as usize;
    (y_pred == y_true) as i32
}

fn main() {
    let start = Instant::now();
    let (train_ds, test_ds) = load_dataset();
    let duration = start.elapsed();
    println!("Time elapsed in loading dataset: {:?}", duration);
    println!("Train dataset size: {}", train_ds.len());
    println!("Test dataset size: {}", test_ds.len());
    let net = MLP::new(28 * 28, vec![256, 128], 10);
    println!("Created MLP with {} parameters", net.parameters().len());
    let mut optim = r_nn::nn::optim::base_sgd::BaseSGD::new(net.parameters(), 0.01);
    let mut buffer_time = 0;
    let mut prev_loss = None;
    for epoch in 0..100 {
        let mut curr_epoch_loss = 0.0;
        let epoch_start = Instant::now();
        for (_, (images, labels)) in train_ds.clone().enumerate() {
            let y_pred = net.forward(&images);
            let loss = cross_entropy(&y_pred, &labels);
            optim.zero_grad();
            loss.backward();
            optim.step();
            curr_epoch_loss += loss.get_item(vec![0]).item();
        }
        let epoch_duration = epoch_start.elapsed();
        curr_epoch_loss /= train_ds.num_batches() as f32;
        if let Some(pre) = prev_loss {
            if curr_epoch_loss >= pre {
                buffer_time += 1;
                if buffer_time > 1 {
                    println!("Decreasing learning rate");
                    optim._update_lr(optim.lr * 0.5);
                    buffer_time = 0;
                }
            } else {
                buffer_time = 0;
            }
        }
        prev_loss = Some(curr_epoch_loss);

        println!(
            "Epoch: {}, Loss: {}, Time: {:?}",
            epoch, curr_epoch_loss, epoch_duration
        );

        let mut acc = 0;
        for (images, labels) in test_ds.clone() {
            let y_pred = net.forward(&images);
            acc += accuracy(&y_pred, &labels);
        }
        println!("Accuracy: {}", acc as f32 / test_ds.len() as f32);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn float_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-4
    }
    #[test]
    fn test_cross_entropy() {
        {
            let y_pred = Tensor::new(vec![0.1, 0.2, 0.7], vec![1, 3]);
            let y_true = Tensor::new(vec![0.0, 1.0, 0.0], vec![1, 3]);
            let loss = cross_entropy(&y_pred, &y_true);
            assert!(float_eq(loss.get_item(vec![0]).item(), 1.2679));
        }
        {
            let y_pred = Tensor::new(
                vec![0.1000, 0.2000, 0.7000, 0.8000, 0.1000, 0.1000],
                vec![2, 3],
            );
            let y_true = Tensor::new(vec![0., 0., 1., 1., 0., 0.], vec![2, 3]);
            let loss = cross_entropy(&y_pred, &y_true);
            assert_eq!(loss.shape, vec![1]);
            assert!(float_eq(loss.get_item(vec![0]).item(), 0.7288));
        }
    }

    #[test]
    fn test_computational_graph_mnist() {
        let dim = 28;
        let images = Tensor::new(vec![0.1; dim * dim * 1], vec![1, dim * dim]);
        let net = MLP::new(28 * 28, vec![256, 128], 10);
        let y_labels = Tensor::from_vec(create_one_hot_label(2), vec![1, 10], true);

        let y_pred = net.forward(&images);
        let loss = cross_entropy(&y_pred, &y_labels);
        loss.backward();
        for param in net.parameters() {
            assert!(param.data.borrow().grad.is_some());
        }
    }
}
