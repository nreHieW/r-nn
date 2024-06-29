#![allow(dead_code)]
use npy::NpyData;
use r_nn::{
    core::Tensor,
    nn::{self, Linear},
};
use std::io::Read;

pub fn read_from_npy(path: &str, shape: Vec<usize>) -> Tensor {
    let mut buf = vec![];
    std::fs::File::open(path)
        .unwrap()
        .read_to_end(&mut buf)
        .unwrap();

    let data = NpyData::from_bytes(&buf)
        .unwrap()
        .into_iter()
        .collect::<Vec<f32>>();
    Tensor::from_vec(data, shape)
}

pub struct GPT2Config {
    pub block_size: usize, // Context length
    pub vocab_size: usize, // Number of tokens in the vocabulary
    pub n_layer: usize,    // Number of hidden layers
    pub n_head: usize,     // Number of attention heads
    pub n_embd: usize,     // Dimension of the embeddings and hidden states
}

impl GPT2Config {
    pub fn get_gpt_config() -> Self {
        // Self {
        //     block_size: 1024,
        //     vocab_size: 50257,
        //     n_layer: 12,
        //     n_head: 12,
        //     n_embd: 768,
        // }
        Self {
            block_size: 10,
            vocab_size: 50257,
            n_layer: 2,
            n_head: 10,
            n_embd: 100,
        }
    }
}

// Model Stuff Ref: https://github.com/karpathy/ng-video-lecture/blob/master/gpt.py and https://github.com/karpathy/build-nanogpt/blob/master/train_gpt2.py
struct MLP {
    c_fc: nn::Linear,
    act: nn::GELU,
    c_proj: nn::Linear,
}

impl MLP {
    pub fn new(config: &GPT2Config) -> Self {
        Self {
            c_fc: nn::Linear::new(config.n_embd, config.n_embd * 4, true),
            act: nn::GELU::new(),
            c_proj: nn::Linear::new(config.n_embd * 4, config.n_embd, true),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        self.c_proj
            .forward(&self.act.forward(&self.c_fc.forward(x)))
    }
}

fn generate_tril(size: usize) -> Tensor {
    let mut items = vec![];
    for i in 0..size {
        for j in 0..size {
            items.push(if i < j { 0.0 } else { 1.0 });
        }
    }
    Tensor::from_vec(items, vec![size, size])
}

pub struct CausalSelfAttention {
    c_attn: Linear,
    c_proj: Linear,
    n_head: usize,
    n_embd: usize,
}

impl CausalSelfAttention {
    pub fn new(config: &GPT2Config) -> Self {
        assert_eq!(config.n_embd % config.n_head, 0);
        Self {
            c_attn: Linear::new(config.n_embd, config.n_embd * 3, true),
            c_proj: Linear::new(config.n_embd, config.n_embd, true),
            n_head: config.n_head,
            n_embd: config.n_embd,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let (b, t, c) = (x.shape[0], x.shape[1], x.shape[2]);
        let qkv = self.c_attn.forward(x);
        let splitted = qkv.split(self.n_embd, 2);
        let q = splitted[0]
            .reshape(vec![b, t, self.n_head, c / self.n_head])
            .transpose(1, 2);
        let k = splitted[1]
            .reshape(vec![b, t, self.n_head, c / self.n_head])
            .transpose(1, 2);
        let v = splitted[2]
            .reshape(vec![b, t, self.n_head, c / self.n_head])
            .transpose(1, 2);

        // attention
        let attn_logits = &q.matmul(&k.transpose(2, 3)) / (self.n_embd as f32).sqrt();
        let mask = &generate_tril(t);
        let attn_logits = &(&attn_logits * mask)
            + &(&(&(&Tensor::zeros(mask.shape.clone()) + 1.0) - mask) * f32::NEG_INFINITY);

        let o = attn_logits
            .softmax(3)
            .matmul(&v)
            .transpose(1, 2)
            .reshape(vec![b, t, c]);
        self.c_proj.forward(&o)
    }
}

pub struct Block {
    ln_1: nn::LayerNorm,
    attn: CausalSelfAttention,
    ln_2: nn::LayerNorm,
    mlp: MLP,
}

impl Block {
    pub fn new(config: &GPT2Config) -> Self {
        Self {
            ln_1: nn::LayerNorm::new(config.n_embd),
            attn: CausalSelfAttention::new(config),
            ln_2: nn::LayerNorm::new(config.n_embd),
            mlp: MLP::new(config),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let x = x + &self.attn.forward(&self.ln_1.forward(x));
        &x + &self.mlp.forward(&self.ln_2.forward(&x))
    }
}

pub struct GPT {
    wte: nn::Embedding,
    wpe: nn::Embedding,
    h: Vec<Block>,
    ln_f: nn::LayerNorm,
    lm_head: Linear,
}

impl GPT {
    pub fn new(config: &GPT2Config) -> Self {
        let wte = nn::Embedding::new(config.vocab_size, config.n_embd);
        let wpe = nn::Embedding::new(config.block_size, config.n_embd);
        let h = (0..config.n_layer)
            .map(|_| Block::new(config))
            .collect::<Vec<Block>>();
        let ln_f = nn::LayerNorm::new(config.n_embd);
        let lm_head = Linear::new(config.n_embd, config.vocab_size, true);
        Self {
            wte,
            wpe,
            h,
            ln_f,
            lm_head,
        }
    }

    pub fn forward(&self, idx: &Tensor) -> Tensor {
        let (b, t) = (idx.shape[0], idx.shape[1]);
        assert_eq!(b, 1, "Batch size must be 1");
        let pos = Tensor::from_vec((0..t).map(|x| x as f32).collect::<Vec<f32>>(), vec![t]);
        let pos_emb = self.wpe.forward(&pos);
        let tok_emb = self.wte.forward(&idx.squeeze(0));
        let mut x = (&tok_emb + &pos_emb).unsqueeze(0);
        for block in &self.h {
            println!("In {:?}", x.shape);
            x = block.forward(&x);
            println!("Out {:?}", x.shape);
        }
        x = self.ln_f.forward(&x);
        let logits = self.lm_head.forward(&x);

        logits
    }
}

#[cfg(test)]
mod tests {
    use r_nn::core::{
        tensor::{self, Tensor},
        Value,
    };
    use std::vec;

    use nn::Linear;

    use super::*;

    #[test]
    fn test_load() {
        let mut layer = Linear::new(10, 68, false);
        let tensor = read_from_npy("pysrc/test.npy", vec![10, 68]);
        let ref_out = read_from_npy("pysrc/out.npy", vec![2, 68]);
        let a = Tensor {
            items: (0..20).map(|x| Value::new(x as f32)).collect(),
            shape: vec![2, 10],
        };
        let orig = layer.forward(&a);
        layer.weights = tensor.clone();
        let res = layer.forward(&a);
        assert_ne!(orig, res);
        assert_eq!(res, ref_out);
    }

    #[test]
    fn test_attn_shapes() {
        let mut config = GPT2Config {
            block_size: 10,
            vocab_size: 50,
            n_layer: 12,
            n_head: 10,
            n_embd: 100,
        };
        let attn = CausalSelfAttention::new(&config);
        let a = Tensor {
            items: (0..(1 * 10 * config.n_embd))
                .map(|x| Value::new(x as f32))
                .collect(),
            shape: vec![1, 10, config.n_embd],
        };
        let res = attn.forward(&a);
        assert_eq!(res.shape, vec![1, 10, config.n_embd]);
    }

    #[test]
    fn test_gpt_shapes() {
        let config = GPT2Config {
            block_size: 10,
            vocab_size: 50,
            n_layer: 2,
            n_head: 10,
            n_embd: 100,
        };
        let gpt = GPT::new(&config);
        let a = Tensor {
            items: (0..(1 * 10)).map(|x| Value::new(x as f32)).collect(),
            shape: vec![1, 10],
        };
        let res = gpt.forward(&a);
        assert_eq!(res.shape, vec![1, 10, config.vocab_size]);
    }
}
