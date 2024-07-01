use r_nn::core::Tensor;

mod tokenizer;
mod transformer;

fn main() {
    let max_new_tokens = 15;
    let tokenizer = tokenizer::Tokenizer::new();
    let cfg = &transformer::GPT2Config::get_gpt_config();
    let mut gpt2 = transformer::GPT::new(cfg);
    gpt2.no_grad();
    gpt2.from_pretrained();
    let s = "Hello, I'm a language model";
    let tokens = tokenizer
        .encode(s)
        .iter()
        .map(|x| *x as f32)
        .collect::<Vec<f32>>();
    let seq_len = tokens.len();
    let tensor = Tensor::from_vec(tokens, vec![1, seq_len as usize], false);
    println!("Input Tokens Shape {:?}", tensor.shape);
    println!("Generating {} tokens", max_new_tokens);
    let output = gpt2.generate(&tensor, max_new_tokens, true).squeeze(0); // [20]
    let output_tokens = output
        .items
        .iter()
        .map(|x| x.item() as u32)
        .collect::<Vec<u32>>();
    let output_text = tokenizer.decode(output_tokens.as_slice());
    println!("Output Text: {:?}", output_text);
}
