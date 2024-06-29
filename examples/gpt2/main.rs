#[cfg(feature = "dhat-heap")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;
use r_nn::core::Tensor;

mod tokenizer;
mod transformer;

fn main() {
    #[cfg(feature = "dhat-heap")]
    let _profiler = dhat::Profiler::new_heap();

    let tokenizer = tokenizer::Tokenizer::new();
    let cfg = &transformer::GPT2Config::get_gpt_config();
    let gpt2 = transformer::GPT::new(cfg);
    let s = "Hello, I'm a language model";
    let tokens = tokenizer
        .encode(s)
        .iter()
        .map(|x| *x as f32)
        .collect::<Vec<f32>>();
    let seq_len = tokens.len();
    let tensor = Tensor::from_vec(tokens, vec![1, seq_len as usize]);
    println!("Input Tokens Shape {:?}", tensor.shape);
    let output = gpt2.forward(&tensor);

    println!("{:?}", output.shape);
}
