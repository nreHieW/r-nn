use crate::core::value::Value;
use std::fmt::{self, Display, Formatter};
use std::hash::{Hash, Hasher};
use std::iter::Sum;
use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub};
use std::rc::Rc;

impl Add for &Value {
    type Output = Value;

    fn add(self, other: &Value) -> Value {
        self.add_other(other)
    }
}

impl Add<i32> for &Value {
    type Output = Value;

    fn add(self, other: i32) -> Value {
        self.add_other(&Value::new(other as f32, true))
    }
}

impl Add<&Value> for i32 {
    type Output = Value;

    fn add(self, other: &Value) -> Value {
        Value::new(self as f32, true).add_other(other)
    }
}

impl Add<f32> for &Value {
    type Output = Value;

    fn add(self, other: f32) -> Value {
        self.add_other(&Value::new(other, true))
    }
}

impl Add<&Value> for f32 {
    type Output = Value;

    fn add(self, other: &Value) -> Value {
        Value::new(self, true).add_other(other)
    }
}

impl Mul for &Value {
    type Output = Value;

    fn mul(self, other: &Value) -> Value {
        self.mul_other(other)
    }
}

impl Mul<i32> for &Value {
    type Output = Value;

    fn mul(self, other: i32) -> Value {
        self.mul_other(&Value::new(other as f32, true))
    }
}

impl Mul<&Value> for i32 {
    type Output = Value;

    fn mul(self, other: &Value) -> Value {
        Value::new(self as f32, true).mul_other(other)
    }
}

impl Mul<f32> for &Value {
    type Output = Value;

    fn mul(self, other: f32) -> Value {
        self.mul_other(&Value::new(other, true))
    }
}

impl Mul<&Value> for f32 {
    type Output = Value;

    fn mul(self, other: &Value) -> Value {
        other.mul_other(&Value::new(self, true))
    }
}

impl Div for &Value {
    type Output = Value;

    fn div(self, other: &Value) -> Value {
        self.mul_other(&other.pow(-1.0))
    }
}

impl Div<i32> for &Value {
    type Output = Value;
    fn div(self, other: i32) -> Value {
        self.mul_other(&Value::new(1.0 / other as f32, true))
    }
}

impl Div<&Value> for i32 {
    type Output = Value;

    fn div(self, other: &Value) -> Value {
        Value::new(self as f32, true).mul_other(&other.pow(-1.0))
    }
}

impl Div<f32> for &Value {
    type Output = Value;

    fn div(self, other: f32) -> Value {
        self.mul_other(&Value::new(1.0 / other, true))
    }
}

impl Div<&Value> for f32 {
    type Output = Value;

    fn div(self, other: &Value) -> Value {
        other.pow(-1.0).mul_other(&Value::new(self, true))
    }
}

impl Neg for &Value {
    type Output = Value;

    fn neg(self) -> Value {
        self * -1
    }
}

impl Sub for &Value {
    type Output = Value;

    fn sub(self, other: &Value) -> Value {
        self.add_other(&other.neg())
    }
}
impl Sub<i32> for &Value {
    type Output = Value;
    fn sub(self, other: i32) -> Value {
        self.add_other(&Value::new(-other as f32, true))
    }
}

impl Sub<&Value> for i32 {
    type Output = Value;

    fn sub(self, other: &Value) -> Value {
        Value::new(self as f32, true).add_other(&(-other))
    }
}

impl Sub<f32> for &Value {
    type Output = Value;

    fn sub(self, other: f32) -> Value {
        self.add_other(&Value::new(-other, true))
    }
}

impl Sub<&Value> for f32 {
    type Output = Value;

    fn sub(self, other: &Value) -> Value {
        Value::new(self, true).add_other(&other.neg())
    }
}

impl Display for Value {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        let mut s = "Value(data=".to_string();
        s.push_str(&self.data.borrow().data.to_string());
        s.push_str(", Children=[");
        for child in self.data.borrow().children.iter() {
            s.push_str(&child.data.borrow().data.to_string());
            s.push_str(", ");
        }
        s.push_str("])");
        write!(f, "{}", s)
    }
}

impl Hash for Value {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Hash the pointer address
        Rc::as_ptr(&self.data).hash(state);
    }
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        // Check if both Rc instances point to the same memory
        Rc::ptr_eq(&self.data, &other.data)
    }
}

impl Eq for Value {}

impl Sum for Value {
    fn sum<I: Iterator<Item = Value>>(iter: I) -> Self {
        let mut sum = Value::new(0.0, true);
        for i in iter {
            sum = &sum + &i;
        }
        sum
    }
}

impl<'a> Sum<&'a Value> for Value {
    fn sum<I: Iterator<Item = &'a Value>>(iter: I) -> Self {
        let mut sum = Value::new(0.0, true);
        for i in iter {
            sum = &sum + i;
        }
        sum
    }
}

impl AddAssign for Value {
    fn add_assign(&mut self, other: Self) {
        *self = &*self + &other;
    }
}

impl PartialOrd for Value {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.data
            .borrow()
            .data
            .partial_cmp(&other.data.borrow().data)
    }
}
