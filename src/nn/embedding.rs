#![allow(dead_code)]
use crate::core::{Tensor, Value};

pub struct Embedding {
    weights: Tensor,
    num_embeddings: usize,
    embedding_dim: usize,
}

impl Embedding {
    pub fn new(num_embeddings: usize, embedding_dim: usize) -> Self {
        let weights = Tensor::randn(vec![num_embeddings, embedding_dim]);
        Embedding {
            weights,
            num_embeddings,
            embedding_dim,
        }
    }

    pub fn forward(&self, indices: &Tensor) -> Tensor {
        let flat_indices = indices.to_flattened_vec();
        let mut embeddings = Vec::new();

        for &index in flat_indices.iter() {
            let index = index as usize;

            let embedding = self
                .weights
                .slice(&vec![index..index + 1, 0..self.embedding_dim])
                .squeeze(0);
            embeddings.push(embedding);
        }

        if embeddings.len() == 1 {
            return embeddings[0].clone();
        }
        let mut result = embeddings[0].clone();
        for i in 1..embeddings.len() {
            result = result.cat(&embeddings[i], result.shape.len() - 1);
        }
        let mut result_shape = indices.shape.clone();
        result_shape.push(self.embedding_dim);
        result.reshape(result_shape)
    }
    pub fn parameters(&self) -> Vec<&Value> {
        self.weights.items.iter().collect::<Vec<_>>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_shape() {
        let embedding = Embedding::new(10, 3);
        let outputs = embedding.forward(&Tensor {
            items: vec![Value::new(2.0)],
            shape: vec![1],
        });
        println!("{:?}", outputs.shape);
        assert_eq!(outputs.shape, vec![3]);
    }

    #[test]
    fn test_embedding_computational_graph() {
        let embedding = Embedding::new(10, 3);
        let outputs = embedding.forward(&Tensor {
            items: vec![Value::new(2.0)],
            shape: vec![1],
        });
        let y = Tensor::zeros(vec![3]);
        let loss = (&outputs - &y).pow(2.0).sum(-1);
        loss.backward();
        assert_eq!(outputs.shape, vec![3]);

        let params = embedding.parameters()[6..9].to_vec();
        for p in params {
            assert!(p.data.borrow().grad.is_some());
        }
    }
}
