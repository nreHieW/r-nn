use std::collections::HashSet;
use std::fmt::{self, Display, Formatter};
use std::hash::{Hash, Hasher};
use std::ops::{Div, Neg, Sub};
use std::{
    cell::RefCell,
    ops::{Add, Mul},
    rc::Rc,
};
type BackwardFn = fn(Value) -> ();

#[derive(Clone, Debug)]
struct ValueData {
    data: f32,
    children: Vec<Value>,
    grad: f32,
    backward_fn: Option<BackwardFn>,
}
#[derive(Clone, Debug)]
pub struct Value {
    data: Rc<RefCell<ValueData>>,
}

fn topological_sort(value: &Value, visited: &mut HashSet<Value>, output: &mut Vec<Value>) {
    if !visited.contains(value) {
        visited.insert(value.clone());
        for child in value.data.borrow().children.iter() {
            topological_sort(child, visited, output);
        }
        output.push(value.clone());
    }
}
impl Value {
    pub fn new(data: f32) -> Value {
        Value {
            data: Rc::new(RefCell::new(ValueData {
                data,
                children: vec![],
                grad: 0.0,
                backward_fn: None,
            })),
        }
    }

    pub fn backward(&self) {
        let mut visited = HashSet::new();
        let mut sorted = vec![];
        topological_sort(self, &mut visited, &mut sorted);
        self.data.borrow_mut().grad = 1.0;
        for value in sorted.iter().rev() {
            if let Some(f) = value.data.borrow().backward_fn {
                f(value.clone());
            }
        }
    }

    pub fn add_other(&self, other: &Value) -> Value {
        let out = Value {
            data: Rc::new(RefCell::new(ValueData {
                data: self.data.borrow().data + other.data.borrow().data,
                children: vec![self.clone(), other.clone()],
                grad: 0.0,
                backward_fn: None,
            })),
        };

        let f = |out: Value| -> () {
            let child1 = &out.data.borrow().children[0]; // self
            let child2 = &out.data.borrow().children[1]; // other

            child1.data.borrow_mut().grad += 1.0 * out.data.borrow().grad;
            child2.data.borrow_mut().grad += 1.0 * out.data.borrow().grad;
        };
        out.data.borrow_mut().backward_fn = Some(f);
        out
    }

    pub fn mul_other(&self, other: &Value) -> Value {
        let out = Value {
            data: Rc::new(RefCell::new(ValueData {
                data: self.data.borrow().data * other.data.borrow().data,
                children: vec![self.clone(), other.clone()],
                grad: 0.0,
                backward_fn: None,
            })),
        };

        let f = |out: Value| -> () {
            let child1 = &out.data.borrow().children[0]; // self
            let child2 = &out.data.borrow().children[1]; // other

            child1.data.borrow_mut().grad += child2.data.borrow().data * out.data.borrow().grad;
            child2.data.borrow_mut().grad += child1.data.borrow().data * out.data.borrow().grad;
        };
        out.data.borrow_mut().backward_fn = Some(f);
        out
    }

    pub fn pow(&self, other: &Value) -> Value {
        let out = Value {
            data: Rc::new(RefCell::new(ValueData {
                data: self.data.borrow().data.powf(other.data.borrow().data),
                children: vec![self.clone(), other.clone()],
                grad: 0.0,
                backward_fn: None,
            })),
        };

        let f = |out: Value| -> () {
            let child1 = &out.data.borrow().children[0]; // self
            let child2 = &out.data.borrow().children[1]; // other
            let other_data = child2.data.borrow().data;
            let self_data = child1.data.borrow().data;

            child1.data.borrow_mut().grad +=
                other_data * self_data.powf(other_data - 1.0) * out.data.borrow().grad;
        };
        out.data.borrow_mut().backward_fn = Some(f);
        out
    }

    pub fn exp(&self) -> Value {
        let out = Value {
            data: Rc::new(RefCell::new(ValueData {
                data: self.data.borrow().data.exp(),
                children: vec![self.clone()],
                grad: 0.0,
                backward_fn: None,
            })),
        };

        let f = |out: Value| -> () {
            let child = &out.data.borrow().children[0]; // self
            child.data.borrow_mut().grad += out.data.borrow().grad * out.data.borrow().data;
        };
        out.data.borrow_mut().backward_fn = Some(f);
        out
    }
}

impl Add for &Value {
    type Output = Value;

    fn add(self, other: &Value) -> Value {
        self.add_other(other)
    }
}

impl Add<i32> for &Value {
    type Output = Value;

    fn add(self, other: i32) -> Value {
        self.add_other(&Value::new(other as f32))
    }
}

impl Add<&Value> for i32 {
    type Output = Value;

    fn add(self, other: &Value) -> Value {
        Value::new(self as f32).add_other(other)
    }
}

impl Add<f32> for &Value {
    type Output = Value;

    fn add(self, other: f32) -> Value {
        self.add_other(&Value::new(other))
    }
}

impl Add<&Value> for f32 {
    type Output = Value;

    fn add(self, other: &Value) -> Value {
        Value::new(self).add_other(other)
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
        self.mul_other(&Value::new(other as f32))
    }
}

impl Mul<&Value> for i32 {
    type Output = Value;

    fn mul(self, other: &Value) -> Value {
        Value::new(self as f32).mul_other(other)
    }
}

impl Mul<f32> for &Value {
    type Output = Value;

    fn mul(self, other: f32) -> Value {
        self.mul_other(&Value::new(other))
    }
}

impl Mul<&Value> for f32 {
    type Output = Value;

    fn mul(self, other: &Value) -> Value {
        other.mul_other(&Value::new(self))
    }
}

impl Div for &Value {
    type Output = Value;

    fn div(self, other: &Value) -> Value {
        self.mul_other(&other.pow(&Value::new(-1.0)))
    }
}

impl Div<i32> for &Value {
    type Output = Value;
    fn div(self, other: i32) -> Value {
        self.mul_other(&Value::new(1.0 / other as f32))
    }
}

impl Div<&Value> for i32 {
    type Output = Value;

    fn div(self, other: &Value) -> Value {
        Value::new(self as f32).mul_other(&other.pow(&Value::new(-1.0)))
    }
}

impl Div<f32> for &Value {
    type Output = Value;

    fn div(self, other: f32) -> Value {
        self.mul_other(&Value::new(1.0 / other))
    }
}

impl Div<&Value> for f32 {
    type Output = Value;

    fn div(self, other: &Value) -> Value {
        other.pow(&Value::new(-1.0)).mul_other(&Value::new(self))
    }
}

impl Neg for &Value {
    type Output = Value;

    fn neg(self) -> Value {
        Value::new(-self.data.borrow().data)
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
        self.add_other(&Value::new(-other as f32))
    }
}

impl Sub<&Value> for i32 {
    type Output = Value;

    fn sub(self, other: &Value) -> Value {
        Value::new(self as f32).add_other(&(-other))
    }
}

impl Sub<f32> for &Value {
    type Output = Value;

    fn sub(self, other: f32) -> Value {
        self.add_other(&Value::new(-other))
    }
}

impl Sub<&Value> for f32 {
    type Output = Value;

    fn sub(self, other: &Value) -> Value {
        Value::new(self).add_other(&other.neg())
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

// impl Hash for ValueData {
//     fn hash<H: Hasher>(&self, state: &mut H) {
//         self.data.to_bits().hash(state);
//         self.grad.to_bits().hash(state);
//         self.children.hash(state);
//         self.backward_fn.hash(state);
//     }
// }

// impl Hash for Value {
//     fn hash<H: Hasher>(&self, state: &mut H) {
//         self.data.borrow().hash(state);
//     }
// }

// impl PartialEq for ValueData {
//     fn eq(&self, other: &Self) -> bool {
//         self.data == other.data
//             && self.grad == other.grad
//             && self.children == other.children
//             && self.backward_fn == other.backward_fn
//     }
// }

// impl PartialEq for Value {
//     fn eq(&self, other: &Self) -> bool {
//         self.data.borrow().eq(&other.data.borrow())
//     }
// }

// impl Eq for ValueData {}
// impl Eq for Value {}
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

#[cfg(test)]
mod tests {
    use super::*;

    fn float_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < 0.0001
    }

    #[test]
    fn arithmetic_test() {
        let a = Value::new(3.0);
        let b = Value::new(2.0);
        let c = Value::new(0.0);
        let d = &a + &b;
        let e = &(&d * &a) * &b;
        let f = &(&e + &c) * &a;
        let g = &f.pow(&b) + &e;
        let h = &g / &a;
        assert_eq!(d.data.borrow().data, 5.0);
        assert_eq!(e.data.borrow().data, 30.0);
        assert_eq!(f.data.borrow().data, 90.0);
        assert_eq!(g.data.borrow().data, 8130.0);
        assert_eq!(h.data.borrow().data, 2710.0);
        assert_eq!(f.data.borrow().children.len(), 2);
        assert_eq!(g.data.borrow().children.len(), 2);
    }

    #[test]
    fn indiv_backward() {
        let a = Value::new(2.0);
        let b = a.pow(&Value::new(2.0));
        b.backward();
        assert_eq!(a.data.borrow().grad, 4.0);
        assert_eq!(b.data.borrow().grad, 1.0);
        let x = Value::new(3.0);
        let y = x.exp();
        y.backward();
        assert!(float_eq(x.data.borrow().grad, 20.0855));
    }

    #[test]
    fn basic_backward() {
        /*
        a = 2, b = -3, c = 10, f = -2
        e = a * b = -6
        d = e + c = 4
        L = d * f = -8
        > L = (a * b + c ) * f

        > dL/da = f * b = -2 * -3 = 6
        > Gradient of L = 1 because (L + 0.0001 - L) / 0.0001

        > dL/dd = f = -2
        > dL/df = d = 4

        we have dd/dc (and dd/de) which are 1 (Addition case)
        > dL/dc = dL/dd * dd/dc = f * 1 = -2
        > dL/de = dL/dd * dd/de = f * 1 = -2

        We have de/da = b and de/db = a
        > dL/da = dL/de * de/da = -2 * b = -2 * -3 = 6
        > dL/db = dL/de * de/db = -2 * a = -2 * 2 = -4
        */
        let a = Value::new(2.0);
        let b = Value::new(-3.0);
        let c = Value::new(10.0);
        let f = Value::new(-2.0);
        let e = &a * &b; // -6
        let d = &e + &c; // 4
        let l = &d * &f; // -8
        l.backward();
        assert_eq!(a.data.borrow().grad, 6.0);
        assert_eq!(b.data.borrow().grad, -4.0);
        assert_eq!(c.data.borrow().grad, -2.0);
        assert_eq!(d.data.borrow().grad, -2.0);
        assert_eq!(e.data.borrow().grad, -2.0);
        assert_eq!(f.data.borrow().grad, 4.0);
        assert_eq!(l.data.borrow().grad, 1.0);
    }

    #[test]
    fn test_eq_for_topo() {
        let a = Value::new(2.0);
        let b = Value::new(2.0);
        let mut s = HashSet::new();
        s.insert(a.clone());
        s.insert(b.clone());
        assert_eq!(s.len(), 2);
    }

    #[test]
    fn neuron_backward() {
        /*
        inputs: x1 = 2, x2 = 0
        weights: w1 = -3 , w2 = 1, b = 6.881373587

        > x1w1 = x1*w1
        > x2w2 = x2*w2
        > x1w1x2w2 = x1w1 + x2w2
        > n = x1w1x2w2 + b
        > O = tanh(n)

        Gradient of O = 1

        dO/dn = 1 - tanh(n) ** 2 = 1 - O ** 2 = 0.5

        We have dn/db = 1 and dn/d(x1w1x2w2) = 1
        dO/d(x1w1x2w2) = dO/dn * dn/d(x1w1x2w2) = 0.5 * 1 = 0.5
        dO/db = dO/dn * dn/db = 0.5 * 1 = 0.5
        dO/x2w2 = 0.5
        dO/x1w1 = 0.5

        We have dx2w2/dx2 = w2 and dx2w2/dw2 = x2
        dO/dx2 = dO/dx2w2 * dx2w2/dx2 = 0.5 * w2 = 0.5 * 1 = 0.5
        dO/dw2 = dO/dx2w2 * dx2w2/dw2 = 0.5 * x2 = 0.5 * 0 = 0

        We have dx1w1/dx1 = w1 and dx1w1/dw1 = x1
        dO/dx1 = dO/dx1w1 * dx1w1/dx1 = 0.5 * w1 = 0.5 * -3 = -1.5
        dO/dw1 = dO/dx1w1 * dx1w1/dw1 = 0.5 * x1 = 0.5 * 2 = 1
        */
        let x1 = Value::new(2.0);
        let x2 = Value::new(0.0);
        let w1 = Value::new(-3.0);
        let w2 = Value::new(1.0);
        let b = Value::new(6.881373587);
        let x1w1 = &x1 * &w1;
        let x2w2 = &x2 * &w2;
        let x1w1x2w2 = &x1w1 + &x2w2;

        let n = &x1w1x2w2 + &b;
        // let o = n.tanh()
        let o = &(&(2 * &n).exp() - 1) / &(&(2 * &n).exp() + 1);
        o.backward();

        assert!(float_eq(x1.data.borrow().grad, -1.5));
        assert!(float_eq(x2.data.borrow().grad, 0.5));
        assert!(float_eq(w1.data.borrow().grad, 1.0));
        assert!(float_eq(w2.data.borrow().grad, 0.0));
        assert!(float_eq(x1w1.data.borrow().grad, 0.5));
        assert!(float_eq(x2w2.data.borrow().grad, 0.5));
        assert!(float_eq(x1w1x2w2.data.borrow().grad, 0.5));
        assert!(float_eq(b.data.borrow().grad, 0.5));
        assert!(float_eq(n.data.borrow().grad, 0.5));
        assert!(float_eq(o.data.borrow().grad, 1.0));
    }

    #[test]
    fn use_twice_simple() {
        let a = Value::new(2.0);
        let b = &a + &a;
        b.backward();
        assert_eq!(a.data.borrow().grad, 2.0);
        assert_eq!(b.data.borrow().grad, 1.0);
    }

    #[test]
    fn use_twice_complex() {
        /*
        a = -2, b = 3
        d = a * b = -6
        e = a + b = 1
        f = d * e = -6

        df/de = d = -6
        df/dd = e = 1

        df/da = df/dd * dd/da + df/de * de/da = 1 * b + -6 * 1 = -3
        df/db = df/dd * dd/db + df/de * de/db = 1 * a + -6 * 1 = -8
        */
        let a = Value::new(-2.0);
        let b = Value::new(3.0);
        let d = &a * &b;
        let e = &a + &b;
        let f = &d * &e;
        f.backward();
        assert_eq!(a.data.borrow().grad, -3.0);
        assert_eq!(b.data.borrow().grad, -8.0);
        assert_eq!(d.data.borrow().grad, 1.0);
        assert_eq!(e.data.borrow().grad, -6.0);
        assert_eq!(f.data.borrow().grad, 1.0);
    }
}
