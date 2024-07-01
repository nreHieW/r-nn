#![allow(dead_code)]
use npy::NpyData;
use r_nn::{
    core::{Tensor, Value},
    nn::{self, Linear},
};
use rand_distr::{Distribution, WeightedIndex};

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
    Tensor::from_vec(data, shape, false)
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
        Self {
            block_size: 1024,
            vocab_size: 50257,
            n_layer: 12,
            n_head: 12,
            n_embd: 768,
        }
        // Self {
        //     block_size: 10,
        //     vocab_size: 50257,
        //     n_layer: 2,
        //     n_head: 10,
        //     n_embd: 100,
        // }
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
        let x = self.c_fc.forward(x);
        let x = self.act.forward(&x);
        self.c_proj.forward(&x)
    }

    pub fn parameters(&self) -> Vec<&Value> {
        let mut params = self.c_fc.parameters();
        params.extend(self.c_proj.parameters());
        params
    }

    pub fn no_grad(&mut self) {
        self.c_fc.no_grad();
        self.c_proj.no_grad();
    }

    pub fn from_pretrained(&mut self, index: usize) {
        let cfc_weight_fpath =
            "pysrc/weights/h.{}.mlp.c_fc.weight_3072_768.npy".replace("{}", &index.to_string());
        let cfc_bias_fpath =
            "pysrc/weights/h.{}.mlp.c_fc.bias_3072.npy".replace("{}", &index.to_string());
        let cproj_weight_fpath =
            "pysrc/weights/h.{}.mlp.c_proj.weight_768_3072.npy".replace("{}", &index.to_string());
        let cproj_bias_fpath =
            "pysrc/weights/h.{}.mlp.c_proj.bias_768.npy".replace("{}", &index.to_string());
        self.c_fc.weights = read_from_npy(&cfc_weight_fpath, vec![768, 3072]);
        self.c_fc.bias = read_from_npy(&cfc_bias_fpath, vec![3072]);
        self.c_proj.weights = read_from_npy(&cproj_weight_fpath, vec![3072, 768]);
        self.c_proj.bias = read_from_npy(&cproj_bias_fpath, vec![768]);
        println!("Loaded weights for MLP layer {}", index);
    }
}

fn generate_tril(size: usize) -> Tensor {
    let mut items = vec![];
    for i in 0..size {
        for j in 0..size {
            items.push(if i < j { 0.0 } else { 1.0 });
        }
    }
    Tensor::from_vec(items, vec![size, size], false)
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
        // println!("QKV {:?}", qkv.shape);
        // println!("QKV {}", qkv.get_item(vec![0, 1, 21]));
        // println!("QKV {}", qkv.get_item(vec![0, 1, 400]));
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
        let attn_logits =
            &q.matmul(&k.transpose(2, 3)) / (self.n_embd as f32 / self.n_head as f32).sqrt();
        // CORRECT

        let mask = &generate_tril(t);
        let attn_logits = &(&attn_logits * mask)
            + &(&(&(&Tensor::zeros(mask.shape.clone(), false) + 1.0) - mask) * -3.4028235e+38);

        let o = attn_logits
            .softmax(3)
            .matmul(&v)
            .transpose(1, 2)
            .reshape(vec![b, t, c]);
        self.c_proj.forward(&o)
    }

    pub fn parameters(&self) -> Vec<&Value> {
        let mut params = self.c_attn.parameters();
        params.extend(self.c_proj.parameters());
        params
    }
    pub fn no_grad(&mut self) {
        self.c_attn.no_grad();
        self.c_proj.no_grad();
    }

    pub fn from_pretrained(&mut self, index: usize) {
        let cattn_weight_fpath =
            "pysrc/weights/h.{}.attn.c_attn.weight_2304_768.npy".replace("{}", &index.to_string());
        let cattn_bias_fpath =
            "pysrc/weights/h.{}.attn.c_attn.bias_2304.npy".replace("{}", &index.to_string());
        let cproj_weight_fpath =
            "pysrc/weights/h.{}.attn.c_proj.weight_768_768.npy".replace("{}", &index.to_string());
        let cproj_bias_fpath =
            "pysrc/weights/h.{}.attn.c_proj.bias_768.npy".replace("{}", &index.to_string());
        self.c_attn.weights = read_from_npy(&cattn_weight_fpath, vec![768, 2304]);
        self.c_attn.bias = read_from_npy(&cattn_bias_fpath, vec![2304]);
        self.c_proj.weights = read_from_npy(&cproj_weight_fpath, vec![768, 768]);
        self.c_proj.bias = read_from_npy(&cproj_bias_fpath, vec![768]);
        println!("Loaded weights for Attention layer {}", index);
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

    pub fn parameters(&self) -> Vec<&Value> {
        let mut params = self.ln_1.parameters();
        params.extend(self.attn.parameters());
        params.extend(self.ln_2.parameters());
        params.extend(self.mlp.parameters());
        params
    }

    pub fn no_grad(&mut self) {
        self.ln_1.no_grad();
        self.attn.no_grad();
        self.ln_2.no_grad();
        self.mlp.no_grad();
    }

    pub fn from_pretrained(&mut self, index: usize) {
        self.attn.from_pretrained(index);
        self.mlp.from_pretrained(index);

        // Load layer norms
        let ln1_weight_fpath =
            "pysrc/weights/h.{}.ln_1.weight_768.npy".replace("{}", &index.to_string());
        let ln1_bias_fpath =
            "pysrc/weights/h.{}.ln_1.bias_768.npy".replace("{}", &index.to_string());
        let ln2_weight_fpath =
            "pysrc/weights/h.{}.ln_2.weight_768.npy".replace("{}", &index.to_string());
        let ln2_bias_fpath =
            "pysrc/weights/h.{}.ln_2.bias_768.npy".replace("{}", &index.to_string());
        self.ln_1.weight = read_from_npy(&ln1_weight_fpath, vec![768]);
        self.ln_1.bias = read_from_npy(&ln1_bias_fpath, vec![768]);
        self.ln_2.weight = read_from_npy(&ln2_weight_fpath, vec![768]);
        self.ln_2.bias = read_from_npy(&ln2_bias_fpath, vec![768]);
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
        let lm_head = Linear::new(config.n_embd, config.vocab_size, false);
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
        let pos = Tensor::from_vec(
            (0..t).map(|x| x as f32).collect::<Vec<f32>>(),
            vec![t],
            false,
        );
        let pos_emb = self.wpe.forward(&pos);
        let tok_emb = self.wte.forward(&idx.squeeze(0));
        let mut x = (&tok_emb + &pos_emb).unsqueeze(0);
        for block in &self.h {
            println!("In {:?}", x.shape);
            x = block.forward(&x);
            println!("Out {:?}", x.shape);
        }
        x = self.ln_f.forward(&x);
        let logits = self.lm_head.forward(&x); //todo for infernce only need the last

        logits
    }

    pub fn forward_inference(&self, idx: &Tensor) -> Tensor {
        let (b, t) = (idx.shape[0], idx.shape[1]);
        assert_eq!(b, 1, "Batch size must be 1");
        let pos = Tensor::from_vec(
            (0..t).map(|x| x as f32).collect::<Vec<f32>>(),
            vec![t],
            false,
        );
        let pos_emb = self.wpe.forward(&pos);
        // println!("Pos Emb {}", pos_emb.sum(-1));
        let tok_emb = self.wte.forward(&idx.squeeze(0));
        // println!("Tok Emb {}", tok_emb.sum(-1));
        let mut x = (&tok_emb + &pos_emb).unsqueeze(0);
        for block in &self.h {
            // println!("In {}", x.sum(-1));
            x = block.forward(&x);
            // println!("Out {}", x.sum(-1));
        }
        x = self.ln_f.forward(&x);
        // println!("LN F {}", x.sum(-1));
        let logits = self.lm_head.forward(
            &x.slice(&vec![0..1, t - 1..t, 0..x.shape[2]])
                .squeeze(0)
                .squeeze(0),
        );

        logits
    }

    pub fn generate(&self, idx: &Tensor, max_new_tokens: usize, do_sample: bool) -> Tensor {
        let mut x = idx.clone();
        for _ in 0..max_new_tokens {
            // println!("Input {:?}", x.shape);
            let last_logit = self.forward_inference(&x);
            // let next_token = last_logit.argmax()
            if do_sample {
                let probs = last_logit.softmax(0);
                let mut rng = rand::thread_rng();
                let dist = WeightedIndex::new(
                    probs
                        .items
                        .iter()
                        .map(|x| x.item() as f64)
                        .collect::<Vec<_>>(),
                )
                .unwrap();
                let next_token = dist.sample(&mut rng);
                // println!("Output:{}", next_token);
                let next_token_tensor =
                    Tensor::from_vec(vec![next_token as f32], vec![1, 1], false);
                x = x.cat(&next_token_tensor, 1);
            } else {
                let next_token = last_logit.argmax();
                // println!("Output:{}", next_token);
                let next_token_tensor =
                    Tensor::from_vec(vec![next_token as f32], vec![1, 1], false);
                x = x.cat(&next_token_tensor, 1);
            }
        }
        x
    }

    pub fn parameters(&self) -> Vec<&Value> {
        let mut params = self.wte.parameters();
        params.extend(self.wpe.parameters());
        for block in &self.h {
            params.extend(block.parameters());
        }
        params.extend(self.ln_f.parameters());
        params.extend(self.lm_head.parameters());
        params
    }

    pub fn no_grad(&mut self) {
        self.wte.no_grad();
        self.wpe.no_grad();
        for block in self.h.iter_mut() {
            block.no_grad();
        }
        self.ln_f.no_grad();
        self.lm_head.no_grad();
    }

    pub fn from_pretrained(&mut self) {
        self.wte.weights =
            read_from_npy("pysrc/weights/wte.weight_50257_768.npy", vec![50257, 768]);
        println!("Loaded weights for Token Embedding");
        self.wpe.weights = read_from_npy("pysrc/weights/wpe.weight_1024_768.npy", vec![1024, 768]);
        println!("Loaded weights for Positional Embedding");
        for i in 0..self.h.len() {
            self.h[i].from_pretrained(i);
        }
        self.ln_f.weight = read_from_npy("pysrc/weights/ln_f.weight_768.npy", vec![768]);
        self.ln_f.bias = read_from_npy("pysrc/weights/ln_f.bias_768.npy", vec![768]);
        self.lm_head.weights = read_from_npy(
            "pysrc/weights/lm_head.weight_50257_768.npy",
            vec![768, 50257],
        );
    }

    pub fn display_num_params_per_layer(&self) {
        let mut total_params = 0;
        println!("======= Model Summary =======");
        println!(
            "Positional Embedding has {} parameters",
            self.wpe.parameters().len()
        );
        println!(
            "Token Embedding has {} parameters",
            self.wte.parameters().len()
        );
        for (i, layer) in self.h.iter().enumerate() {
            let layer_params = layer.parameters().len();
            total_params += layer_params;
            println!("Layer {} has {} parameters", i, layer_params);
        }
        println!("LayerNorm has {} parameters", self.ln_f.parameters().len());
        println!("LM Head has {} parameters", self.lm_head.parameters().len());
        total_params += self.ln_f.parameters().len();
        total_params += self.lm_head.parameters().len();
        total_params += self.wte.parameters().len();
        total_params += self.wpe.parameters().len();
        println!("GPT2 has {} parameters", total_params);
    }
}

#[cfg(test)]
mod tests {
    use r_nn::core::{tensor::Tensor, Value};
    use std::vec;

    use nn::Linear;

    use super::*;

    fn float_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < 0.001
    }

    #[test]
    fn test_lm_head() {
        let mut lm_head = Linear::new(768, 50257, false);

        lm_head.weights = read_from_npy(
            "pysrc/weights/lm_head.weight_50257_768.npy",
            vec![768, 50257],
        );
        let x = &Tensor {
            items: (0..768).map(|x| Value::new(x as f32, true)).collect(),
            shape: vec![1, 768],
        } / 768.0;
        let res = lm_head.forward(&x);
        assert_eq!(res.shape, vec![1, 50257]);
        assert_eq!(res.argmax(), 41403);
    }

    #[test]
    fn test_load() {
        let mut layer = Linear::new(10, 68, false);
        let tensor = read_from_npy("pysrc/test.npy", vec![10, 68]);
        let ref_out = read_from_npy("pysrc/out.npy", vec![2, 68]);
        let a = Tensor {
            items: (0..20).map(|x| Value::new(x as f32, true)).collect(),
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
        let config = GPT2Config {
            block_size: 10,
            vocab_size: 50,
            n_layer: 12,
            n_head: 10,
            n_embd: 100,
        };
        let attn = CausalSelfAttention::new(&config);
        let a = Tensor {
            items: (0..(1 * 10 * config.n_embd))
                .map(|x| Value::new(x as f32, false))
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
            items: (0..(1 * 10)).map(|x| Value::new(x as f32, true)).collect(),
            shape: vec![1, 10],
        };
        let res = gpt.forward(&a);
        assert_eq!(res.shape, vec![1, 10, config.vocab_size]);
    }

    #[test]
    fn test_gpt_generate() {
        let config = GPT2Config {
            block_size: 50,
            vocab_size: 50,
            n_layer: 1,
            n_head: 4,
            n_embd: 40,
        };
        let gpt = GPT::new(&config);
        let a = Tensor {
            items: (0..(1 * 10)).map(|x| Value::new(x as f32, true)).collect(),
            shape: vec![1, 10],
        };
        let res = gpt.generate(&a, 10, true);
        println!("{}", res);
        println!("{:?}", res.shape);
        assert_eq!(res.shape, vec![1, 20]);
    }

    #[test]
    fn test_block() {
        let config = GPT2Config::get_gpt_config();
        let mut block = Block::new(&config);
        block.from_pretrained(0);
        let x = &Tensor {
            items: (0..5376).map(|x| Value::new(x as f32, false)).collect(),
            shape: vec![1, 7, config.n_embd],
        } / 5376.0;
        let res = block.forward(&x);
        assert_eq!(res.shape, vec![1, 7, config.n_embd]);
        assert_eq!(res.argmax(), 5055);
    }
    #[test]
    fn test_mlp() {
        let config = GPT2Config::get_gpt_config();
        let mut mlp = MLP::new(&config);
        mlp.from_pretrained(0);
        let x = &Tensor {
            items: (0..768).map(|x| Value::new(x as f32, false)).collect(),
            shape: vec![1, config.n_embd],
        } / 768.0;
        let res = mlp.forward(&x);
        assert_eq!(res.shape, vec![1, config.n_embd]);
        println!("{}", res.sum(-1));
        assert!(float_eq(res.sum(-1).get_item(vec![0]).item(), 100.9246));
    }

    #[test]
    fn test_attn() {
        let config = GPT2Config::get_gpt_config();
        let mut attn = CausalSelfAttention::new(&config);
        attn.from_pretrained(0);
        let x = &Tensor {
            items: (0..5376).map(|x| Value::new(x as f32, false)).collect(),
            shape: vec![1, 7, config.n_embd],
        } / 5376.0;
        let res = attn.forward(&x);
        assert_eq!(res.shape, vec![1, 7, config.n_embd]);
        assert!(float_eq(res.get_item(vec![0, 2, 3]).item(), -1.3513));
        assert!(float_eq(res.get_item(vec![0, 4, 431]).item(), 3.5347));
    }

    #[test]
    fn test_embedding() {
        let cfg = GPT2Config::get_gpt_config();
        let mut pos_emb = nn::Embedding::new(cfg.block_size, cfg.n_embd);
        let mut tok_emb = nn::Embedding::new(cfg.vocab_size, cfg.n_embd);
        pos_emb.weights = read_from_npy("pysrc/weights/wpe.weight_1024_768.npy", vec![1024, 768]);
        tok_emb.weights = read_from_npy("pysrc/weights/wte.weight_50257_768.npy", vec![50257, 768]);
        let x = Tensor::from_vec(
            vec![15496.0 as f32, 11.0, 314.0, 1101.0, 257.0, 3303.0, 2746.0],
            vec![1, 7],
            false,
        );
        let pos = Tensor::from_vec(
            (0..7).map(|x| x as f32).collect::<Vec<f32>>(),
            vec![7],
            false,
        );
        let pos_emb_res = pos_emb.forward(&pos);
        let tok_emb_res = tok_emb.forward(&x.squeeze(0));
        let out = &pos_emb_res + &tok_emb_res;
        println!("{}", pos_emb_res.sum(-1));
        println!("{}", tok_emb_res.sum(-1));
        println!("{}", out.sum(-1));
    }
}
