#![allow(dead_code)]
use regex::Regex;
use std::{
    collections::{HashMap, HashSet},
    fs::{self, File},
    io::{BufRead, BufReader},
};

fn get_pairs(word: &[String]) -> HashSet<(String, String)> {
    let mut pairs = HashSet::new();
    for i in 0..word.len() - 1 {
        pairs.insert((word[i].clone(), word[i + 1].clone()));
    }
    pairs
}

fn bytes_to_unicode() -> HashMap<u8, char> {
    let mut bs: Vec<u8> = (33..=126).chain(161..=172).chain(174..=255).collect();
    let mut cs: Vec<u16> = bs.iter().map(|&b| b as u16).collect();
    let mut n = 0;
    for b in 0..=255u8 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    bs.into_iter()
        .zip(cs.into_iter().map(|n| char::from_u32(n as u32).unwrap()))
        .collect()
}
pub struct Tokenizer {
    byte_encoder: HashMap<u8, char>,
    byte_decoder: HashMap<char, u8>,
    encoder: serde_json::Value,
    decoder: HashMap<u64, String>,
    tokenizer: serde_json::Value,
    merges: HashMap<(String, String), usize>,
    pat: Regex,
}

fn read_json_file(path: &str) -> serde_json::Value {
    let data = fs::read_to_string(path).expect("Unable to read file");
    serde_json::from_str(&data).expect("Unable to parse JSON")
}

impl Tokenizer {
    pub fn new() -> Self {
        let vocab = read_json_file("pysrc/hf/models--openai-community--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e/vocab.json");
        let tokenizer = read_json_file("pysrc/hf/models--openai-community--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e/tokenizer.json");
        let bpe_file = File::open("pysrc/hf/models--openai-community--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e/merges.txt").unwrap();
        let bpe_reader = BufReader::new(bpe_file);
        let bpe_merges: Vec<String> = bpe_reader
            .lines()
            .skip(1)
            .filter_map(|line| line.ok())
            .filter(|line| !line.is_empty())
            .collect();
        let mut merges = HashMap::new();
        bpe_merges.iter().enumerate().for_each(|(i, line)| {
            let mut parts = line.split_whitespace();
            let pair = (
                parts.next().unwrap().to_string(),
                parts.next().unwrap().to_string(),
            );
            merges.insert(pair, i);
        });

        let pat = Regex::new(r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+")
            .unwrap();
        let byte_encoder = bytes_to_unicode();
        let byte_decoder: HashMap<char, u8> = byte_encoder.iter().map(|(a, &b)| (b, *a)).collect();
        let mut decoder = HashMap::new();
        for (key, value) in vocab.as_object().unwrap() {
            decoder.insert(value.as_u64().unwrap(), key.clone());
        }

        Self {
            byte_encoder: bytes_to_unicode(),
            byte_decoder,
            encoder: vocab,
            decoder,
            tokenizer,
            merges,
            pat,
        }
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut bpe_tokens = Vec::new();
        for token in self.pat.find_iter(text) {
            let token = token.as_str();
            let encoded_token: String = token
                .as_bytes()
                .iter()
                .map(|&b| self.byte_encoder[&b])
                .collect();
            for bpe_token in self.bpe(&encoded_token).split(' ') {
                if let Some(token_id) = self.encoder[bpe_token].as_u64() {
                    bpe_tokens.push(token_id as u32);
                }
            }
        }
        bpe_tokens
        // todo!();
    }
    pub fn decode(&self, tokens: &[u32]) -> String {
        let text: Vec<String> = tokens
            .iter()
            .filter_map(|&token| self.decoder.get(&(token as u64)).map(|s| s.clone()))
            .collect();
        let decoded: String = text
            .join("")
            .chars()
            .filter_map(|c| self.byte_decoder.get(&c))
            .map(|&b| b as char)
            .collect();
        decoded
    }

    fn bpe(&self, token: &str) -> String {
        let mut word: Vec<String> = token.chars().map(|c| c.to_string()).collect();
        let mut pairs = get_pairs(&word);

        if pairs.is_empty() {
            return token.to_string();
        }

        loop {
            let bigram = pairs
                .iter()
                .min_by_key(|&pair| self.merges.get(pair).unwrap_or(&usize::MAX))
                .cloned();

            if let Some((first, second)) = bigram {
                let mut new_word = Vec::new();
                if self.merges.get(&(first.clone(), second.clone())).is_none() {
                    break;
                }
                let mut i = 0;
                while i < word.len() {
                    if let Some(j) = word[i..].iter().position(|x| x == &first) {
                        new_word.extend(word[i..i + j].iter().cloned());
                        i += j;
                        if i < word.len() - 1 && word[i + 1] == second {
                            new_word.push(first.clone() + &second);
                            i += 2;
                        } else {
                            new_word.push(word[i].clone());
                            i += 1;
                        }
                    } else {
                        new_word.extend(word[i..].iter().cloned());
                        break;
                    }
                }
                word = new_word;
                if word.len() == 1 {
                    break;
                } else {
                    pairs = get_pairs(&word);
                }
            } else {
                break;
            }
        }

        word.join(" ")
    }
}

#[cfg(test)]

mod test {
    use super::*;

    #[test]
    fn encode_test() {
        let s = "Hello, I'm a language model";
        let tokenizer = Tokenizer::new();
        let encoded = tokenizer.encode(s);
        let decoded = tokenizer.decode(&encoded);
        let ans = [15496, 11, 314, 1101, 257, 3303, 2746];
        assert_eq!(encoded, ans);
        assert_eq!(decoded, s);
    }

    #[test]
    fn complex_test_tokenizer() {
        let s = "This is a test. A very long convoluted test for a GPT2 tokenizer";
        let tokenizer = Tokenizer::new();
        let encoded = tokenizer.encode(s);
        let decoded = tokenizer.decode(&encoded);
        let ans = [
            1212, 318, 257, 1332, 13, 317, 845, 890, 47370, 1332, 329, 257, 402, 11571, 17, 11241,
            7509,
        ];
        assert_eq!(decoded, s);
        assert_eq!(encoded, ans);
    }

    #[test]
    fn test_numbers() {
        let s = "1234567890fwieonfe#@$@#";
        let tokenizer = Tokenizer::new();
        let encoded = tokenizer.encode(s);
        let decoded = tokenizer.decode(&encoded);
        let ans = [
            10163, 2231, 30924, 3829, 44482, 494, 261, 5036, 2, 31, 3, 41573,
        ];
        assert_eq!(decoded, s);
        assert_eq!(encoded, ans);
    }
}
