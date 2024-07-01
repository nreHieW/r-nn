use std::{cmp::max, ops::Range, vec};

use crate::core::{
    tensor::{broadcast_shape, numel, IndexIterator, Tensor},
    Value,
};

pub fn matmul_contiguous(a: &Tensor, b: &Tensor) -> Tensor {
    // a @ b
    // The core idea is that the stride is the number of cols (the width)

    let a_height = a.shape[0];
    let a_width = a.shape[1];
    let b_width = b.shape[1];
    let b_height = b.shape[0];
    if a_width != b_height {
        panic!("Matrix dimensions do not match");
    }
    let mut items = Vec::with_capacity(a_height * b_width); // With capacity is one allocation only
    for i in 0..a_height {
        for j in 0..b_width {
            let mut curr = Value::new(0.0, true);
            for k in 0..b_height {
                curr += a.get_item(vec![i, k]) * b.get_item(vec![k, j]);
            }
            items.push(curr);
        }
    }
    Tensor {
        shape: vec![a_height, b_width],
        items,
    }
}

pub fn matmul_contiguous_transpose(a: &Tensor, b: &Tensor) -> Tensor {
    // Breakpoint for speed improvement is ~ 256

    let b = b.t();
    let a_height = a.shape[0];
    let b_width = b.shape[1];
    let b_height = b.shape[0];

    let mut items = Vec::with_capacity(a_height * b_width); // With capacity is one allocation only
    for i in 0..a_height {
        for j in 0..b_width {
            let mut curr = Value::new(0.0, true);
            for k in 0..b_height {
                curr += a.get_item(vec![i, k]) * b.get_item(vec![j, k]);
            }
            items.push(curr);
        }
    }
    Tensor {
        shape: vec![a_height, b_width],
        items,
    }
}

fn _matmul2d(a: &Tensor, b: &Tensor) -> Tensor {
    if (b.shape[0] == b.shape[1])
        && (max(a.shape[0], a.shape[1]) > 256 || max(b.shape[0], b.shape[1]) > 256)
    {
        matmul_contiguous_transpose(a, b)
    } else {
        matmul_contiguous(a, b)
    }
}

pub fn matmulnd(a: &Tensor, b: &Tensor) -> Tensor {
    let a_shape = a.shape.clone();
    let b_shape = b.shape.clone();

    if a_shape[a.shape.len() - 1] != b_shape[b_shape.len() - 2] {
        panic!(
            "Matrix dimensions do not match, Got {:?} and {:?}",
            a_shape, b_shape
        );
    }

    // 2 d case
    if a_shape.len() == 2 && b_shape.len() == 2 {
        return _matmul2d(a, b);
    }

    // Broadcast the batch dimensions by only considering dims until the last 2
    let batch_shape = broadcast_shape(
        &a_shape[..a_shape.len() - 2].to_vec(),
        &b_shape[..b_shape.len() - 2].to_vec(),
    );
    let mut res_shape = batch_shape.clone();
    res_shape.push(a_shape[a_shape.len() - 2]);
    res_shape.push(b_shape[b_shape.len() - 1]);
    let mut index_iter = IndexIterator::new(&batch_shape);

    let mut result_items: Vec<Value> = Vec::with_capacity(numel(&res_shape));
    let a_mat_slice = vec![0..a_shape[a_shape.len() - 2], 0..a_shape[a_shape.len() - 1]];
    let b_mat_slice = vec![0..b_shape[b_shape.len() - 2], 0..b_shape[b_shape.len() - 1]];
    let mut a_mat_shape = vec![a_shape[a_shape.len() - 2], a_shape[a_shape.len() - 1]];
    a_mat_shape = [batch_shape.clone(), a_mat_shape].concat();
    let mut b_mat_shape = vec![b_shape[b_shape.len() - 2], b_shape[b_shape.len() - 1]];
    b_mat_shape = [batch_shape.clone(), b_mat_shape].concat();

    while let Some(broadcast_index) = index_iter.next() {
        let batch_slice_ranges: Vec<Range<usize>> =
            broadcast_index.iter().map(|&i| i..i + 1).collect(); // batches
        let mut a_slice = a._get_slice(
            &[&batch_slice_ranges[..], &a_mat_slice[..]].concat(),
            a_mat_shape.clone(),
        );
        for _ in 0..a_slice.shape.len() - 2 {
            a_slice = a_slice.squeeze(0);
        }

        let mut b_slice = b._get_slice(
            &[&batch_slice_ranges[..], &b_mat_slice[..]].concat(),
            b_mat_shape.clone(),
        );
        for _ in 0..b_slice.shape.len() - 2 {
            b_slice = b_slice.squeeze(0);
        }
        let res_slice = _matmul2d(&a_slice, &b_slice);
        result_items.extend(res_slice.items);
    }

    Tensor {
        shape: res_shape,
        items: result_items,
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::core::{Tensor, Value};

    #[test]
    fn grad_matmul() {
        // (3, 3, 2) @ (2, 2) = (3, 3, 2)
        {
            let a = Tensor {
                items: (0..18).map(|x| Value::new(x as f32, true)).collect(),
                shape: vec![3, 3, 2],
            };
            let b = Tensor {
                items: (0..4).map(|x| Value::new(x as f32, true)).collect(),
                shape: vec![2, 2],
            };
            let o = a.matmul(&b); // (3, 3, 2) @ (2, 2) = (3, 3,2)
            let l = o.sum(-1);
            l.backward();
            assert_eq!(o.shape, vec![3, 3, 2]);
            let ans_vals = [
                2., 3., 6., 11., 10., 19., 14., 27., 18., 35., 22., 43., 26., 51., 30., 59., 34.,
                67.,
            ];
            for i in 0..18 {
                assert_eq!(o.items[i].item(), ans_vals[i]);
            }
            let grad_a = [
                1., 5., 1., 5., 1., 5., 1., 5., 1., 5., 1., 5., 1., 5., 1., 5., 1., 5.,
            ];
            let grad_b = [72., 72., 81., 81.];
            for i in 0..18 {
                assert_eq!(a.items[i].data.borrow().grad.unwrap(), grad_a[i]);
            }
            for i in 0..4 {
                assert_eq!(b.items[i].data.borrow().grad.unwrap(), grad_b[i]);
            }
        }

        // (2, 1, 3, 2) @ (2, 2, 3) = (2, 2, 3, 3)
        {
            let a = Tensor {
                items: (0..12).map(|x| Value::new(x as f32, true)).collect(),
                shape: vec![2, 1, 3, 2],
            };
            let b = Tensor {
                items: (0..12).map(|x| Value::new(x as f32, true)).collect(),
                shape: vec![2, 2, 3],
            };
            let o = a.matmul(&b); // (2, 1, 3, 2) @ (2, 2, 3) = (2, 2, 3, 3)
            let l = o.sum(-1);
            l.backward();
            assert_eq!(o.shape, vec![2, 2, 3, 3]);
            let ans_vals = [
                3., 4., 5., 9., 14., 19., 15., 24., 33., 9., 10., 11., 39., 44., 49., 69., 78.,
                87., 21., 34., 47., 27., 44., 61., 33., 54., 75., 99., 112., 125., 129., 146.,
                163., 159., 180., 201.,
            ];
            for i in 0..36 {
                assert_eq!(o.items[i].item(), ans_vals[i]);
            }
            assert_eq!(l.get_item(vec![0]).item(), 2232.0);

            let a_grad = [24., 42., 24., 42., 24., 42., 24., 42., 24., 42., 24., 42.];
            let b_grad = [30., 30., 30., 36., 36., 36., 30., 30., 30., 36., 36., 36.];
            for i in 0..12 {
                assert_eq!(a.items[i].data.borrow().grad.unwrap(), a_grad[i]);
            }
            for i in 0..12 {
                assert_eq!(b.items[i].data.borrow().grad.unwrap(), b_grad[i]);
            }
        }

        // (4, 1, 2, 3) @ (3, 1) = (4, 1, 2, 1)
        // (4, 1, 2, 1) @ (1, 1, 2) = (4, 1, 2, 2)
        {
            let a = Tensor {
                items: (10..34).map(|x| Value::new(x as f32, true)).collect(),
                shape: vec![4, 1, 2, 3],
            };
            let b = Tensor {
                items: (1..4).map(|x| Value::new(x as f32, true)).collect(),
                shape: vec![3, 1],
            };
            let c = Tensor {
                items: (3..5).map(|x| Value::new(x as f32, true)).collect(),
                shape: vec![1, 1, 2],
            };
            let o = a.matmul(&b).matmul(&c);
            let l = o.sum(-1);
            l.backward();
            assert_eq!(o.shape, vec![4, 1, 2, 2]);
            let ans_vals = [
                204., 272., 258., 344., 312., 416., 366., 488., 420., 560., 474., 632., 528., 704.,
                582., 776.,
            ];
            for i in 0..16 {
                assert_eq!(o.items[i].item(), ans_vals[i]);
            }
            assert_eq!(l.get_item(vec![0]).item(), 7336.0);
            let a_grad = [
                7., 14., 21., 7., 14., 21., 7., 14., 21., 7., 14., 21., 7., 14., 21., 7., 14., 21.,
                7., 14., 21., 7., 14., 21.,
            ];
            let b_grad = [1148., 1204., 1260.];
            let c_grad = [1048., 1048.];
            for i in 0..24 {
                assert_eq!(a.items[i].data.borrow().grad.unwrap(), a_grad[i]);
            }
            for i in 0..3 {
                assert_eq!(b.items[i].data.borrow().grad.unwrap(), b_grad[i]);
            }
            for i in 0..2 {
                assert_eq!(c.items[i].data.borrow().grad.unwrap(), c_grad[i]);
            }
        }
    }
    fn matmul_contiguous_test() {
        let ans = [42., 45., 48., 150., 162., 174., 258., 279., 300.];
        let a = Tensor {
            items: (0..9).map(|x| Value::new(x as f32, true)).collect(),
            shape: vec![9],
        };
        let b = Tensor {
            items: (9..18).map(|x| Value::new(x as f32, true)).collect(),
            shape: vec![3, 3],
        };

        let c = matmul_contiguous(&a.reshape(vec![3, 3]), &b);
        for i in 0..9 {
            assert_eq!(c.items[i].item(), ans[i]);
        }
    }
    #[test]
    fn matmul_contiguous_transpose_test() {
        let ans = [42., 45., 48., 150., 162., 174., 258., 279., 300.];
        let a = Tensor {
            items: (0..9).map(|x| Value::new(x as f32, true)).collect(),
            shape: vec![9],
        };
        let b = Tensor {
            items: (9..18).map(|x| Value::new(x as f32, true)).collect(),
            shape: vec![3, 3],
        };

        let c = matmul_contiguous_transpose(&a.reshape(vec![3, 3]), &b);
        for i in 0..9 {
            assert_eq!(c.items[i].item(), ans[i]);
        }
    }
}
