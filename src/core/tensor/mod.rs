#![allow(dead_code)]
use std::ops::Range;

use rand_distr::Distribution;
use rand_distr::Normal;
pub mod matmul;
pub mod tensor_ops;

use super::Value;

#[derive(Debug)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub items: Vec<Value>,
}
struct IndexIterator {
    shape: Vec<usize>,
    current: Vec<usize>,
    done: bool,
}

impl IndexIterator {
    fn new(shape: &Vec<usize>) -> Self {
        IndexIterator {
            shape: shape.clone(),
            current: vec![0; shape.len()],
            done: false,
        }
    }
}

impl Iterator for IndexIterator {
    type Item = Vec<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        let result = self.current.clone();

        // Increment the current index
        for i in (0..self.shape.len()).rev() {
            self.current[i] += 1;
            if self.current[i] < self.shape[i] {
                break;
            }
            self.current[i] = 0;
            if i == 0 {
                self.done = true;
            }
        }

        Some(result)
    }
}

fn numel(shape: &Vec<usize>) -> usize {
    shape.iter().product()
}
pub fn broadcast_shape(curr_shape: &Vec<usize>, other_shape: &Vec<usize>) -> Vec<usize> {
    let mut result = Vec::new();
    let mut self_iter = curr_shape.iter().rev();
    let mut other_iter = other_shape.iter().rev();

    loop {
        match (self_iter.next(), other_iter.next()) {
            (Some(&a), Some(&b)) => {
                if a == b {
                    result.push(a);
                } else if a == 1 {
                    result.push(b);
                } else if b == 1 {
                    result.push(a);
                } else {
                    panic!("Incompatible shapes for broadcasting");
                }
            }
            (Some(&a), None) => result.push(a),
            (None, Some(&b)) => result.push(b),
            (None, None) => break,
        }
    }

    result.reverse();
    result
}

fn _get_index(mut curr_shape: Vec<usize>, idx: Vec<usize>, broadcasted_shape: Vec<usize>) -> usize {
    let prefix_dims = idx.len().saturating_sub(curr_shape.len());
    let is_broadcast = broadcasted_shape != curr_shape;
    if prefix_dims > 0 || is_broadcast {
        let mut prefix = vec![1; prefix_dims];
        prefix.extend(curr_shape.iter().cloned());
        curr_shape = prefix;
    }

    let num_elements = numel(&curr_shape);
    // Inclusive pointers
    let mut left = 0;
    let mut right = num_elements - 1;
    for i in 0..idx.len() {
        let curr_stride = right - left + 1;
        let mut item_idx = idx[i];
        // Handle broadcasting
        if curr_shape[i] != broadcasted_shape[i] {
            // This is a broadcasted dimension
            item_idx = 0;
        } else if item_idx >= curr_shape[i] {
            // This will panic anyways
            panic!(
                "Index out of bounds {}, for dimension {} with size {}",
                item_idx, i, curr_shape[i]
            );
        }
        let section_size = curr_stride / curr_shape[i];
        left += item_idx * section_size;
        right = left + section_size - 1;
    }

    left
}

impl Tensor {
    pub fn randn(shape: Vec<usize>) -> Self {
        let n = numel(&shape);
        let normal = Normal::new(0.0, 1.0).unwrap();
        let items = (0..n)
            .map(|_| Value::new(normal.sample(&mut rand::thread_rng())))
            .collect();
        Self { shape, items }
    }

    pub fn zeros(shape: Vec<usize>) -> Self {
        let n = numel(&shape);
        let items = vec![Value::new(0.0); n];
        Self { shape, items }
    }

    pub fn backward(&self) {
        if self.shape != vec![1] {
            panic!("Grad can be implicitly created only for scalar outputs");
        }
        self.items[0].backward();
    }

    pub fn reshape(&self, shape: Vec<usize>) -> Self {
        assert_eq!(numel(&shape), numel(&self.shape));
        Self {
            shape,
            items: self.items.clone(),
        }
    }

    pub fn t(&self) -> Self {
        // Underlying data is the same, pointers are just rearranged
        assert_eq!(self.shape.len(), 2);
        let mut items = Vec::with_capacity(self.items.len());
        for i in 0..self.shape[1] {
            for j in 0..self.shape[0] {
                items.push(self.items[j * self.shape[1] + i].clone());
            }
        }
        Self {
            shape: vec![self.shape[1], self.shape[0]],
            items,
        }
    }

    pub fn dot(&self, other: &Self) -> Self {
        assert_eq!(self.shape.len(), 1);
        assert_eq!(other.shape.len(), 1);
        assert_eq!(self.shape[0], other.shape[0]);
        let mut result = Value::new(0.0);
        for i in 0..self.shape[0] {
            result += &self.items[i] * &other.items[i];
        }
        Self {
            shape: vec![1],
            items: vec![result],
        }
    }

    pub fn squeeze(&self, dim: i32) -> Self {
        if dim == -1 {
            // Remove all dimensions of size 1
            Self {
                shape: self.shape.iter().filter(|&x| *x != 1).cloned().collect(),
                items: self.items.clone(),
            }
        } else {
            Self {
                shape: self
                    .shape
                    .iter()
                    .enumerate()
                    .filter(|&(i, _)| i != dim as usize)
                    .map(|(_, v)| *v)
                    .collect(),
                items: self.items.clone(),
            }
        }
    }

    pub fn unsqueeze(&self, dim: usize) -> Self {
        let mut new_shape = self.shape.clone();
        new_shape.insert(dim, 1);
        Self {
            shape: new_shape,
            items: self.items.clone(),
        }
    }

    fn _get_item(&self, idx: Vec<usize>, broadcasted_shape: Vec<usize>) -> &Value {
        &self.items[_get_index(self.shape.clone(), idx, broadcasted_shape)]
    }

    pub fn get_item(&self, idx: Vec<usize>) -> &Value {
        &self.items[_get_index(self.shape.clone(), idx, self.shape.clone())]
    }

    fn _get_slice(&self, idxs: &Vec<Range<usize>>, broadcasted_shape: Vec<usize>) -> Self {
        let mut result_shape = Vec::with_capacity(idxs.len());
        let mut result_items = Vec::new();

        // Calculate the new shape
        for range in idxs {
            result_shape.push(range.end - range.start);
        }

        // Use IndexIterator to iterate over all indices in the new shape
        let index_iter = IndexIterator::new(&result_shape);

        for idx in index_iter {
            // Map the index from the result shape to the original shape
            let original_idx: Vec<usize> = idx
                .iter()
                .zip(idxs.iter())
                .map(|(&i, range)| range.start + i)
                .collect();

            // Get the item from the original tensor
            let item = self
                ._get_item(original_idx, broadcasted_shape.clone())
                .clone();
            result_items.push(item);
        }

        Self {
            shape: result_shape,
            items: result_items,
        }
    }

    fn apply_fn(&self, other: &Tensor, f: fn(&Value, &Value) -> Value) -> Tensor {
        let result_shape = broadcast_shape(&self.shape, &other.shape);
        let num_elements = numel(&result_shape);

        let mut result_items = Vec::with_capacity(num_elements);

        for idx in IndexIterator::new(&result_shape) {
            let a = self._get_item(idx.clone(), result_shape.clone());
            let b = other._get_item(idx.clone(), result_shape.clone());
            result_items.push(f(a, b));
        }

        Tensor {
            shape: result_shape,
            items: result_items,
        }
    }

    fn sum(&self, dim: i32) -> Tensor {
        if dim == -1 {
            let sum = self.items.iter().sum();
            Self {
                shape: vec![1],
                items: vec![sum],
            }
        } else {
            let dim = dim as usize;
            let mut result_shape = self.shape.clone();
            result_shape.remove(dim);
            let mut result_items = Vec::with_capacity(numel(&result_shape));
            let mut index_iter = IndexIterator::new(&result_shape);
            while let Some(mut idx) = index_iter.next() {
                idx.insert(dim, 0);
                let mut sum = Value::new(0.0);
                for i in 0..self.shape[dim] {
                    idx[dim] = i;
                    sum += self._get_item(idx.clone(), result_shape.clone()).clone();
                }
                result_items.push(sum);
            }

            Self {
                shape: result_shape,
                items: result_items,
            }
        }
    }

    fn pow(&self, other: &Value) -> Tensor {
        let items = self.items.iter().map(|x| x.pow(other)).collect();
        Self {
            shape: self.shape.clone(),
            items,
        }
    }

    fn exp(&self) -> Tensor {
        let items = self.items.iter().map(|x| x.exp()).collect();
        Self {
            shape: self.shape.clone(),
            items,
        }
    }

    fn softmax(&self, dim: usize) -> Tensor {
        if dim >= self.shape.len() {
            panic!("Dimension out of range");
        }

        let mut result = Tensor::zeros(self.shape.clone());
        let axis_size = self.shape[dim];
        let outer_dims: usize = self.shape[..dim].iter().product();
        let inner_dims: usize = self.shape[dim + 1..].iter().product();

        for outer in 0..outer_dims {
            for inner in 0..inner_dims {
                // Find max value along the specified dimension
                let mut max_val = f32::NEG_INFINITY;
                for i in 0..axis_size {
                    let idx = outer * axis_size * inner_dims + i * inner_dims + inner;
                    max_val = max_val.max(self.items[idx].item());
                }
                let max_val = Value::new(max_val);
                let mut exp_sum = Value::new(0.0);
                for i in 0..axis_size {
                    let idx = outer * axis_size * inner_dims + i * inner_dims + inner;
                    let exp_val = (&self.items[idx] - &max_val).exp();
                    exp_sum += exp_val.clone();
                    result.items[idx] = exp_val;
                }

                for i in 0..axis_size {
                    let idx = outer * axis_size * inner_dims + i * inner_dims + inner;
                    result.items[idx] = &result.items[idx] / &exp_sum;
                }
            }
        }

        result
    }

    fn matmul(&self, other: &Tensor) -> Tensor {
        if self.shape.len() == 1 && other.shape.len() == 1 {
            self.dot(other)
        } else if self.shape.len() == 1 {
            let out = matmul::matmulnd(&self.unsqueeze(0), other);
            out.squeeze((out.shape.len() - 2) as i32)
        } else if other.shape.len() == 1 {
            let out = matmul::matmulnd(self, &other.unsqueeze(1));
            out.squeeze((out.shape.len() - 1) as i32)
        } else {
            matmul::matmulnd(self, other)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn float_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-4
    }

    #[test]
    fn slicing_test() {
        {
            let a = Tensor {
                items: (0..3822).map(|x| Value::new(x as f32)).collect(),
                shape: vec![7, 26, 21],
            };
            let b = a._get_slice(&vec![0..1, 7..19, 0..21], a.shape.clone());
            assert_eq!(b.shape, vec![1, 12, 21]);
        }

        {
            let a = Tensor {
                items: (0..56).map(|x| Value::new(x as f32)).collect(),
                shape: vec![1, 2, 4, 7],
            };
            let b = a
                ._get_slice(&vec![0..1, 1..2, 1..4, 2..5], a.shape.clone())
                .squeeze(0)
                .squeeze(0);
            assert_eq!(b.shape, vec![3, 3]);
            let vals = [37, 38, 39, 44, 45, 46, 51, 52, 53];
            for i in 0..9 {
                assert_eq!(b.items[i].item(), vals[i] as f32);
            }
        }

        {
            let a = Tensor {
                items: (0..28).map(|x| Value::new(x as f32)).collect(),
                shape: vec![4, 7],
            };
            let b = a._get_slice(&vec![1..4, 2..5], a.shape.clone());
            assert_eq!(b.shape, vec![3, 3]);
            let vals = [9, 10, 11, 16, 17, 18, 23, 24, 25];
            for i in 0..9 {
                assert_eq!(b.items[i].item(), vals[i] as f32);
            }
        }
    }

    #[test]
    fn arithmetic_tensor_test() {
        // No broadcasting
        {
            let a = Tensor {
                items: (0..6).map(|x| Value::new(x as f32)).collect(),
                shape: vec![2, 3],
            };
            let b = Tensor {
                items: (0..6).map(|x| Value::new(x as f32)).collect(),
                shape: vec![2, 3],
            };
            let c = &a + &b;
            let d = &b + &a;
            assert_eq!(c, d);
            let ans = [0., 2., 4., 6., 8., 10.];
            assert_eq!(c.shape, vec![2, 3]);
            for i in 0..6 {
                assert_eq!(c.items[i].item(), ans[i]);
            }
        }

        // (2, 3) + (3,) = (2, 3)
        {
            let a = Tensor {
                items: (0..6).map(|x| Value::new(x as f32)).collect(),
                shape: vec![2, 3],
            };
            let b = Tensor {
                items: (0..3).map(|x| Value::new(x as f32)).collect(),
                shape: vec![3],
            };
            let c = &a + &b;
            let d = &b + &a;
            assert_eq!(c, d);
            let ans = [0., 2., 4., 3., 5., 7.];
            assert_eq!(c.shape, vec![2, 3]);

            for i in 0..6 {
                assert_eq!(c.items[i].item(), ans[i]);
            }
        }

        // (2, 5, 2) + (5, 1) = (2, 5, 2)
        {
            let a = Tensor {
                items: (0..20).map(|x| Value::new(x as f32)).collect(),
                shape: vec![2, 5, 2],
            };
            let b = Tensor {
                items: (0..5).map(|x| Value::new(x as f32)).collect(),
                shape: vec![5, 1],
            };
            let c = &a + &b;
            let d = &b + &a;
            assert_eq!(c, d);
            let ans = [
                0., 1., 3., 4., 6., 7., 9., 10., 12., 13., 10., 11., 13., 14., 16., 17., 19., 20.,
                22., 23.,
            ];
            assert_eq!(c.shape, vec![2, 5, 2]);
            for i in 0..20 {
                assert_eq!(c.items[i].item(), ans[i]);
            }
        }

        // (1, 2, 3, 1) * (2, 1, 4) = (1, 2, 3, 4)
        {
            let a = Tensor {
                items: (0..6).map(|x| Value::new(x as f32)).collect(),
                shape: vec![1, 2, 3, 1],
            };
            let b = Tensor {
                items: (0..8).map(|x| Value::new(x as f32)).collect(),
                shape: vec![2, 1, 4],
            };
            let c = &a * &b;
            let d = &b * &a;
            assert_eq!(c, d);
            let ans = [
                0., 0., 0., 0., 0., 1., 2., 3., 0., 2., 4., 6., 12., 15., 18., 21., 16., 20., 24.,
                28., 20., 25., 30., 35.,
            ];
            assert_eq!(c.shape, vec![1, 2, 3, 4]);
            for i in 0..24 {
                assert_eq!(c.items[i].item(), ans[i]);
            }
        }

        // (3, 2, 3, 2) / (1, 1, 2) = (3, 2, 3, 2)
        // (3, 2, 3, 2) + (3, 2, 3, 2) = (3, 2, 3, 2)
        {
            let a = Tensor {
                items: (0..36).map(|x| Value::new(x as f32)).collect(),
                shape: vec![3, 2, 3, 2],
            };
            let b = Tensor {
                items: (1..3).map(|x| Value::new(x as f32)).collect(),
                shape: vec![1, 1, 2],
            };
            let c = &a / &b;
            let d = &a + &c;
            let ans = [
                0.0, 1.5, 4.0, 4.5, 8.0, 7.5, 12.0, 10.5, 16.0, 13.5, 20.0, 16.5, 24.0, 19.5, 28.0,
                22.5, 32.0, 25.5, 36.0, 28.5, 40.0, 31.5, 44.0, 34.5, 48.0, 37.5, 52.0, 40.5, 56.0,
                43.5, 60.0, 46.5, 64.0, 49.5, 68.0, 52.5,
            ];
            assert_eq!(d.shape, vec![3, 2, 3, 2]);
            for i in 0..36 {
                assert_eq!(d.items[i].item(), ans[i]);
            }
        }

        // Scalar test
        {
            let a = Tensor {
                items: (0..6).map(|x| Value::new(x as f32)).collect(),
                shape: vec![2, 3],
            };

            let mut c = &(&(&a + 10.0) / 2.0) * 4.0;
            c += 1.0;
            let ans = [21., 23., 25., 27., 29., 31.];
            assert_eq!(c.shape, vec![2, 3]);
            for i in 0..6 {
                assert_eq!(c.items[i].item(), ans[i]);
            }
        }
    }

    #[test]
    fn softmax_test() {
        let a = Tensor {
            items: (1..4).map(|x| Value::new(x as f32)).collect(),
            shape: vec![3],
        };
        let b = a.softmax(0);
        let ans = [0.0900, 0.2447, 0.6652];
        assert_eq!(b.shape, vec![3]);
        for i in 0..3 {
            assert!(float_eq(b.items[i].item(), ans[i]));
        }
    }

    #[test]
    fn matmul_test() {
        // Type 1: 1d @ 1d - dot product
        {
            let a = Tensor {
                items: (0..6).map(|x| Value::new(x as f32)).collect(),
                shape: vec![6],
            };
            let b = Tensor {
                items: (6..12).map(|x| Value::new(x as f32)).collect(),
                shape: vec![6],
            };
            let c = a.matmul(&b);
            assert_eq!(c.shape, vec![1]);
            assert_eq!(c.items[0].item(), 145.0);
        }

        // Type 2a: 1d @ 2d
        {
            let a = Tensor {
                items: (3..5).map(|x| Value::new(x as f32)).collect(),
                shape: vec![2],
            };
            let b = Tensor {
                items: (0..6).map(|x| Value::new(x as f32)).collect(),
                shape: vec![2, 3],
            };
            let o = a.matmul(&b);
            let l = o.softmax(0);
            let y = Tensor {
                items: vec![Value::new(0.5), Value::new(0.5), Value::new(0.5)],
                shape: vec![3],
            };
            let loss = &(&l - &y).pow(&Value::new(2.0)).sum(-1);
            loss.backward();
            assert_eq!(o.shape, vec![3]);
            assert!(float_eq(loss.get_item(vec![0]).item(), 0.7482));
            let a_grad = [0.0018, 0.0018];
            let b_grad_sum = -3.7696e-07;
            for i in 0..2 {
                assert!(float_eq(a.items[i].data.borrow().grad, a_grad[i]));
            }
            let b_grad_computed_sum = b.items.iter().map(|x| x.data.borrow().grad).sum();
            assert!(float_eq(b_grad_sum, b_grad_computed_sum));
        }

        // Type 2b 1d @ 4d
        {
            let a = Tensor {
                items: (3..5).map(|x| Value::new(x as f32)).collect(),
                shape: vec![2],
            };
            let b = Tensor {
                items: (0..24).map(|x| Value::new(x as f32)).collect(),
                shape: vec![2, 3, 2, 2],
            };
            let o = a.matmul(&b);
            let l = o.softmax(0);
            let y = Tensor {
                items: vec![Value::new(0.5), Value::new(0.5)],
                shape: vec![2],
            };
            let loss = &(&l - &y).pow(&Value::new(2.0)).sum(-1);
            loss.backward();
            assert_eq!(o.shape, vec![2, 3, 2]);
            assert_eq!(loss.shape, vec![1]);
            assert!(float_eq(loss.get_item(vec![0]).item(), 3.0));
        }

        // Type 3a 2d @ 1d
        {
            let a = Tensor {
                items: (0..6).map(|x| Value::new(x as f32)).collect(),
                shape: vec![3, 2],
            };
            let b = Tensor {
                items: (0..2).map(|x| Value::new(x as f32)).collect(),
                shape: vec![2],
            };
            let o = a.matmul(&b);
            let l = o.softmax(0);
            let y = Tensor {
                items: vec![Value::new(0.5), Value::new(0.5), Value::new(0.5)],
                shape: vec![3],
            };
            let loss = &(&l - &y).pow(&Value::new(2.0)).sum(-1);
            loss.backward();
            assert_eq!(o.shape, vec![3]);
            assert!(float_eq(loss.get_item(vec![0]).item(), 0.5154));
        }

        // Type 3b 4d @ 1d

        {
            let a = Tensor {
                items: (0..24).map(|x| Value::new(x as f32)).collect(),
                shape: vec![2, 3, 2, 2],
            };
            let b = Tensor {
                items: (14..16).map(|x| Value::new(x as f32)).collect(),
                shape: vec![2],
            };
            let o = a.matmul(&b);
            let l = o.softmax(0);
            let y = Tensor {
                items: vec![Value::new(0.5), Value::new(0.5)],
                shape: vec![2],
            };
            let loss = &(&l - &y).pow(&Value::new(2.0)).sum(-1);
            loss.backward();
            assert_eq!(o.shape, vec![2, 3, 2]);
            assert_eq!(loss.shape, vec![1]);
            assert!(float_eq(loss.get_item(vec![0]).item(), 3.0));
        }

        // Type 4: (3, 3, 2) @ (2, 2) = (3, 3, 2) - more comprehensive nd tests are in matmul.rs
        {
            let a = Tensor {
                items: (0..18).map(|x| Value::new(x as f32)).collect(),
                shape: vec![3, 3, 2],
            };
            let b = Tensor {
                items: (0..4).map(|x| Value::new(x as f32)).collect(),
                shape: vec![2, 2],
            };
            let c = a.matmul(&b);
            assert_eq!(c.shape, vec![3, 3, 2]);
            let ans = [
                2., 3., 6., 11., 10., 19., 14., 27., 18., 35., 22., 43., 26., 51., 30., 59., 34.,
                67.,
            ];
            for i in 0..18 {
                assert_eq!(c.items[i].item(), ans[i]);
            }
        }
    }
}
