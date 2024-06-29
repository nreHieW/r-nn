use std::fmt::{Display, Formatter, Result};
// use std::hash::{Hash, Hasher};
// use std::iter::Sum;
use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub};

use crate::core::{tensor::Tensor, Value};

impl PartialEq for Tensor {
    fn eq(&self, other: &Self) -> bool {
        self.shape == other.shape
            && self
                .items
                .iter()
                .zip(other.items.iter())
                .all(|(a, b)| a.item() == b.item())
    }
}

impl Eq for Tensor {}

// Tensor - Tensor Operations
impl Add for &Tensor {
    type Output = Tensor;

    fn add(self, other: Self) -> Self::Output {
        self.apply_fn(other, |a, b| a + b)
    }
}

impl Sub for &Tensor {
    type Output = Tensor;

    fn sub(self, other: Self) -> Self::Output {
        self.apply_fn(other, |a, b| a - b)
    }
}

impl Mul for &Tensor {
    // Element-wise multiplication
    type Output = Tensor;

    fn mul(self, other: Self) -> Self::Output {
        self.apply_fn(other, |a, b| a * b)
    }
}

impl Div for &Tensor {
    type Output = Tensor;

    fn div(self, other: Self) -> Self::Output {
        self.apply_fn(other, |a, b| a / b)
    }
}

// Scalar - Tensor Operations
impl Add<f32> for &Tensor {
    type Output = Tensor;

    fn add(self, other: f32) -> Self::Output {
        let items = self.items.iter().map(|x| x + other).collect::<Vec<Value>>();
        Tensor {
            shape: self.shape.clone(),
            items,
        }
    }
}

impl Add<i32> for &Tensor {
    type Output = Tensor;

    fn add(self, other: i32) -> Self::Output {
        let items = self.items.iter().map(|x| x + other).collect::<Vec<Value>>();
        Tensor {
            shape: self.shape.clone(),
            items,
        }
    }
}

impl Sub<f32> for &Tensor {
    type Output = Tensor;

    fn sub(self, other: f32) -> Self::Output {
        let items = self.items.iter().map(|x| x - other).collect::<Vec<Value>>();
        Tensor {
            shape: self.shape.clone(),
            items,
        }
    }
}

impl Sub<i32> for &Tensor {
    type Output = Tensor;

    fn sub(self, other: i32) -> Self::Output {
        let items = self.items.iter().map(|x| x - other).collect::<Vec<Value>>();
        Tensor {
            shape: self.shape.clone(),
            items,
        }
    }
}

impl Mul<f32> for &Tensor {
    type Output = Tensor;

    fn mul(self, other: f32) -> Self::Output {
        let items = self.items.iter().map(|x| x * other).collect::<Vec<Value>>();
        Tensor {
            shape: self.shape.clone(),
            items,
        }
    }
}

impl Mul<i32> for &Tensor {
    type Output = Tensor;

    fn mul(self, other: i32) -> Self::Output {
        let items = self.items.iter().map(|x| x * other).collect::<Vec<Value>>();
        Tensor {
            shape: self.shape.clone(),
            items,
        }
    }
}

impl Div<f32> for &Tensor {
    type Output = Tensor;

    fn div(self, other: f32) -> Self::Output {
        let items = self.items.iter().map(|x| x / other).collect::<Vec<Value>>();
        Tensor {
            shape: self.shape.clone(),
            items,
        }
    }
}

impl Div<i32> for &Tensor {
    type Output = Tensor;

    fn div(self, other: i32) -> Self::Output {
        let items = self.items.iter().map(|x| x / other).collect::<Vec<Value>>();
        Tensor {
            shape: self.shape.clone(),
            items,
        }
    }
}

impl Neg for &Tensor {
    type Output = Tensor;

    fn neg(self) -> Self::Output {
        let items = self.items.iter().map(|x| -x).collect::<Vec<Value>>();
        Tensor {
            shape: self.shape.clone(),
            items,
        }
    }
}

impl AddAssign<Tensor> for Tensor {
    fn add_assign(&mut self, other: Tensor) {
        *self = &*self + &other;
    }
}

// Tensor - Scalar Operations

impl Add<&Tensor> for f32 {
    type Output = Tensor;

    fn add(self, other: &Tensor) -> Self::Output {
        let items = other.items.iter().map(|x| self + x).collect::<Vec<Value>>();
        Tensor {
            shape: other.shape.clone(),
            items,
        }
    }
}

impl Add<&Tensor> for i32 {
    type Output = Tensor;

    fn add(self, other: &Tensor) -> Self::Output {
        let items = other.items.iter().map(|x| self + x).collect::<Vec<Value>>();
        Tensor {
            shape: other.shape.clone(),
            items,
        }
    }
}

impl Mul<&Tensor> for f32 {
    type Output = Tensor;

    fn mul(self, other: &Tensor) -> Self::Output {
        let items = other.items.iter().map(|x| self * x).collect::<Vec<Value>>();
        Tensor {
            shape: other.shape.clone(),
            items,
        }
    }
}

impl Mul<&Tensor> for i32 {
    type Output = Tensor;

    fn mul(self, other: &Tensor) -> Self::Output {
        let items = other.items.iter().map(|x| self * x).collect::<Vec<Value>>();
        Tensor {
            shape: other.shape.clone(),
            items,
        }
    }
}

impl Div<&Tensor> for f32 {
    type Output = Tensor;

    fn div(self, other: &Tensor) -> Self::Output {
        let items = other.items.iter().map(|x| self / x).collect::<Vec<Value>>();
        Tensor {
            shape: other.shape.clone(),
            items,
        }
    }
}

impl Div<&Tensor> for i32 {
    type Output = Tensor;

    fn div(self, other: &Tensor) -> Self::Output {
        let items = other.items.iter().map(|x| self / x).collect::<Vec<Value>>();
        Tensor {
            shape: other.shape.clone(),
            items,
        }
    }
}

impl Sub<&Tensor> for f32 {
    type Output = Tensor;

    fn sub(self, other: &Tensor) -> Self::Output {
        let items = other.items.iter().map(|x| self - x).collect::<Vec<Value>>();
        Tensor {
            shape: other.shape.clone(),
            items,
        }
    }
}

impl Sub<&Tensor> for i32 {
    type Output = Tensor;

    fn sub(self, other: &Tensor) -> Self::Output {
        let items = other.items.iter().map(|x| self - x).collect::<Vec<Value>>();
        Tensor {
            shape: other.shape.clone(),
            items,
        }
    }
}

impl AddAssign<f32> for Tensor {
    fn add_assign(&mut self, other: f32) {
        *self = &*self + other;
    }
}

impl AddAssign<i32> for Tensor {
    fn add_assign(&mut self, other: i32) {
        *self = &*self + other;
    }
}

impl Display for Tensor {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(f, "Tensor(")?;
        self.fmt_recursive(f, 0, &self.shape, 0)?;
        write!(f, ")")
    }
}

impl Tensor {
    fn fmt_recursive(
        &self,
        f: &mut Formatter<'_>,
        start_idx: usize,
        shape: &[usize],
        depth: usize,
    ) -> Result {
        if shape.is_empty() {
            write!(f, "{:.5}", self.items[start_idx].item())
        } else {
            write!(f, "[")?;
            let dim = shape[0];
            let sub_size: usize = shape[1..].iter().product();
            for i in 0..dim {
                if i > 0 {
                    if depth == 0 {
                        write!(f, ",\n ")?;
                        write!(f, "\t")?;
                    } else {
                        write!(f, ", ")?;
                    }
                }
                self.fmt_recursive(f, start_idx + i * sub_size, &shape[1..], depth + 1)?;
            }
            write!(f, "]")
        }
    }
}
