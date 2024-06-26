#![allow(dead_code)]
use std::collections::HashSet;
use std::{cell::RefCell, rc::Rc};
type BackwardFn = fn(Value) -> ();
pub mod value_ops;

#[derive(Debug)] // Note no clone
pub struct ValueData {
    pub data: f32,
    children: Vec<Value>,
    pub grad: Option<f32>,
    backward_fn: Option<BackwardFn>,
}

#[derive(Clone, Debug)]
pub struct Value {
    pub data: Rc<RefCell<ValueData>>,
    pub requires_grad: bool,
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
    pub fn new(data: f32, requires_grad: bool) -> Value {
        // println!("Creating new value with data: {}", data);
        Value {
            data: Rc::new(RefCell::new(ValueData {
                data,
                children: vec![],
                grad: None,
                backward_fn: None,
            })),
            requires_grad,
        }
    }

    pub fn backward(&self) {
        let mut visited = HashSet::new();
        let mut sorted = vec![];
        topological_sort(self, &mut visited, &mut sorted);
        self.data.borrow_mut().grad = Some(1.0);
        for value in sorted.iter().rev() {
            if let Some(f) = value.data.borrow().backward_fn {
                f(value.clone());
            }
            value.data.borrow_mut().backward_fn = None;
            value.data.borrow_mut().children = vec![];
        }
    }

    pub fn add_other(&self, other: &Value) -> Value {
        let req_grad = self.requires_grad && other.requires_grad;
        let out = Value {
            data: Rc::new(RefCell::new(ValueData {
                data: self.data.borrow().data + other.data.borrow().data,
                children: if req_grad {
                    vec![self.clone(), other.clone()]
                } else {
                    vec![]
                },
                grad: None,
                backward_fn: None,
            })),
            requires_grad: req_grad,
        };
        if !req_grad {
            return out;
        }

        let f = |out: Value| -> () {
            let child1 = &out.data.borrow().children[0]; // self
            let child2 = &out.data.borrow().children[1]; // other
            let child1_grad = child1.data.borrow().grad.unwrap_or(0.0);
            child1.data.borrow_mut().grad =
                Some(child1_grad + 1.0 * out.data.borrow().grad.unwrap_or(0.0));
            let child2_grad = child2.data.borrow().grad.unwrap_or(0.0);
            child2.data.borrow_mut().grad =
                Some(child2_grad + 1.0 * out.data.borrow().grad.unwrap_or(0.0));
        };
        out.data.borrow_mut().backward_fn = Some(f);
        out
    }

    pub fn mul_other(&self, other: &Value) -> Value {
        let req_grad = self.requires_grad && other.requires_grad;
        let out = Value {
            data: Rc::new(RefCell::new(ValueData {
                data: self.data.borrow().data * other.data.borrow().data,
                children: if req_grad {
                    vec![self.clone(), other.clone()]
                } else {
                    vec![]
                },
                grad: None,
                backward_fn: None,
            })),
            requires_grad: req_grad,
        };

        if !req_grad {
            return out;
        }

        let f = |out: Value| -> () {
            let out_grad = out.data.borrow().grad.unwrap_or(0.0);
            let child1 = &out.data.borrow().children[0]; // self
            let child2 = &out.data.borrow().children[1]; // other

            let child1_data = child1.data.borrow().data;
            let child2_data = child2.data.borrow().data;

            {
                let mut child1_borrow = child1.data.borrow_mut();
                child1_borrow.grad =
                    Some(child1_borrow.grad.unwrap_or(0.0) + child2_data * out_grad);
            }

            {
                let mut child2_borrow = child2.data.borrow_mut();
                child2_borrow.grad =
                    Some(child2_borrow.grad.unwrap_or(0.0) + child1_data * out_grad);
            }
        };
        out.data.borrow_mut().backward_fn = Some(f);
        out
    }

    pub fn pow(&self, other: f32) -> Value {
        let other = Value::new(other as f32, self.requires_grad);
        let out = Value {
            data: Rc::new(RefCell::new(ValueData {
                data: self.data.borrow().data.powf(other.data.borrow().data),
                children: if self.requires_grad {
                    vec![self.clone(), other.clone()]
                } else {
                    vec![]
                },
                grad: None,
                backward_fn: None,
            })),
            requires_grad: self.requires_grad,
        };
        if !self.requires_grad {
            return out;
        }

        let f = |out: Value| -> () {
            let child1 = &out.data.borrow().children[0]; // self
            let child2 = &out.data.borrow().children[1]; // other
            let other_data = child2.data.borrow().data;
            let self_data = child1.data.borrow().data;
            let child1_grad = child1.data.borrow().grad.unwrap_or(0.0);
            child1.data.borrow_mut().grad = Some(
                child1_grad
                    + other_data
                        * self_data.powf(other_data - 1.0)
                        * out.data.borrow().grad.unwrap_or(0.0),
            );
        };
        out.data.borrow_mut().backward_fn = Some(f);
        out
    }

    pub fn exp(&self) -> Value {
        let out = Value {
            data: Rc::new(RefCell::new(ValueData {
                data: (self.data.borrow().data as f64).exp() as f32,
                children: if self.requires_grad {
                    vec![self.clone()]
                } else {
                    vec![]
                },
                grad: None,
                backward_fn: None,
            })),
            requires_grad: self.requires_grad,
        };
        if !self.requires_grad {
            return out;
        }

        let f = |out: Value| -> () {
            let child = &out.data.borrow().children[0]; // self
            let g = out.data.borrow().grad.unwrap_or(0.0);
            child.data.borrow_mut().grad = Some(g * out.data.borrow().data);
        };
        out.data.borrow_mut().backward_fn = Some(f);
        out
    }
    pub fn ln(&self) -> Value {
        let out = Value {
            data: Rc::new(RefCell::new(ValueData {
                data: self.data.borrow().data.ln(),
                children: if self.requires_grad {
                    vec![self.clone()]
                } else {
                    vec![]
                },
                grad: None,
                backward_fn: None,
            })),
            requires_grad: self.requires_grad,
        };
        if !self.requires_grad {
            return out;
        }

        let f = |out: Value| -> () {
            let child = &out.data.borrow().children[0]; // self
            let g = out.data.borrow().grad.unwrap_or(0.0);
            let child_data = child.data.borrow().data;
            // local grad = 1 / x
            child.data.borrow_mut().grad = Some(g * (1.0 / child_data));
        };
        out.data.borrow_mut().backward_fn = Some(f);
        out
    }

    pub fn relu(&self) -> Value {
        let out = Value {
            data: Rc::new(RefCell::new(ValueData {
                data: self.data.borrow().data.max(0.0),
                children: if self.requires_grad {
                    vec![self.clone()]
                } else {
                    vec![]
                },
                grad: None,
                backward_fn: None,
            })),
            requires_grad: self.requires_grad,
        };
        if !self.requires_grad {
            return out;
        }

        let f = |out: Value| -> () {
            let child = &out.data.borrow().children[0]; // self
            let g = out.data.borrow().grad.unwrap_or(0.0);
            let child_data = child.data.borrow().data;
            child.data.borrow_mut().grad = Some(g * (child_data > 0.0) as i32 as f32);
        };
        out.data.borrow_mut().backward_fn = Some(f);
        out
    }

    pub fn tanh(&self) -> Value {
        let out = Value {
            data: Rc::new(RefCell::new(ValueData {
                data: self.data.borrow().data.tanh(),
                children: if self.requires_grad {
                    vec![self.clone()]
                } else {
                    vec![]
                },
                grad: None,
                backward_fn: None,
            })),
            requires_grad: self.requires_grad,
        };
        if !self.requires_grad {
            return out;
        }

        let f = |out: Value| -> () {
            let child = &out.data.borrow().children[0]; // self
            let g = out.data.borrow().grad.unwrap_or(0.0);
            let child_data = child.data.borrow().data;
            // local grad = 1 - tanh(x)^2
            child.data.borrow_mut().grad = Some(g * (1.0 - child_data.powi(2)));
        };
        out.data.borrow_mut().backward_fn = Some(f);
        out
    }

    pub fn item(&self) -> f32 {
        self.data.borrow().data
    }

    pub fn zero_grad(&self) {
        self.data.borrow_mut().grad = None;
    }

    pub fn no_grad(&mut self) {
        self.requires_grad = false;
        self.data.borrow_mut().grad = None;
        self.data.borrow_mut().backward_fn = None;
        self.data.borrow_mut().children = vec![];
    }

    pub fn update(&self, update_amt: f32) {
        let g = self.data.borrow().grad.unwrap_or(0.0); // If grad is None, then 0.0, so no update
        self.data.borrow_mut().data += update_amt * g;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn float_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < 0.0001
    }

    #[test]
    fn arithmetic_test() {
        let a = Value::new(3.0, true);
        let b = Value::new(2.0, true);
        let c = Value::new(0.0, true);
        let d = &a + &b;
        let e = &(&d * &a) * &b;
        let f = &(&e + &c) * &a;
        let g = &f.pow(2.0) + &e;
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
    fn sum_test() {
        let a = Value::new(3.0, true);
        let b = Value::new(2.0, true);
        let c = Value::new(0.0, true);
        let items = vec![a.clone(), b.clone(), c.clone()];
        let sum = items.iter().sum::<Value>();
        sum.backward();
        assert_eq!(a.data.borrow().grad, Some(1.0));
        assert_eq!(b.data.borrow().grad, Some(1.0));
        assert_eq!(c.data.borrow().grad, Some(1.0));
        assert_eq!(sum.data.borrow().grad, Some(1.0));
        assert_eq!(sum.data.borrow().data, 5.0);
    }

    #[test]
    fn indiv_backward() {
        let a = Value::new(2.0, true);
        let b = a.pow(2.0);
        b.backward();
        assert_eq!(a.data.borrow().grad, Some(4.0));
        assert_eq!(b.data.borrow().grad, Some(1.0));
        let x = Value::new(3., true);
        let y = x.exp();
        y.backward();
        assert!(float_eq(x.data.borrow().grad.unwrap(), 20.0855));
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
        let a = Value::new(2.0, true);
        let b = Value::new(-3.0, true);
        let c = Value::new(10.0, true);
        let f = Value::new(-2.0, true);
        let e = &a * &b; // -6
        let d = &e + &c; // 4
        let l = &d * &f; // -8
        l.backward();
        assert_eq!(a.data.borrow().grad, Some(6.0));
        assert_eq!(b.data.borrow().grad, Some(-4.0));
        assert_eq!(c.data.borrow().grad, Some(-2.0));
        assert_eq!(d.data.borrow().grad, Some(-2.0));
        assert_eq!(e.data.borrow().grad, Some(-2.0));
        assert_eq!(f.data.borrow().grad, Some(4.0));
        assert_eq!(l.data.borrow().grad, Some(1.0));
    }

    #[test]
    fn test_eq_for_topo() {
        let a = Value::new(2.0, true);
        let b = Value::new(2.0, true);
        let mut s = HashSet::new();
        s.insert(a.clone());
        s.insert(b.clone());
        assert_eq!(s.len(), 2);
    }

    #[test]
    fn topo() {
        let a = Value::new(2.0, true);
        let b = Value::new(-3.0, true);
        let c = Value::new(10.0, true);
        let f = Value::new(-2.0, true);
        let e = &a * &b; // -6
        let d = &e + &c; // 4
        let g = &d * &f; // -8
        let mut visited = HashSet::new();
        let mut sorted = vec![];
        topological_sort(&g, &mut visited, &mut sorted);
        assert_eq!(sorted.len(), 7);

        let a = Value::new(2.0, true);
        let b = &a + &a;
        let mut visited = HashSet::new();
        let mut sorted = vec![];
        topological_sort(&b, &mut visited, &mut sorted);
        assert_eq!(sorted.len(), 2);
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
        let x1 = Value::new(2.0, true);
        let x2 = Value::new(0.0, true);
        let w1 = Value::new(-3.0, true);
        let w2 = Value::new(1.0, true);
        let b = Value::new(6.881373587, true);
        let x1w1 = &x1 * &w1;
        let x2w2 = &x2 * &w2;
        let x1w1x2w2 = &x1w1 + &x2w2;

        let n = &x1w1x2w2 + &b;
        // let o = n.tanh()
        let o = &(&(2 * &n).exp() - 1) / &(&(2 * &n).exp() + 1);
        o.backward();

        assert!(float_eq(x1.data.borrow().grad.unwrap(), -1.5));
        assert!(float_eq(x2.data.borrow().grad.unwrap(), 0.5));
        assert!(float_eq(w1.data.borrow().grad.unwrap(), 1.0));
        assert!(float_eq(w2.data.borrow().grad.unwrap(), 0.0));
        assert!(float_eq(x1w1.data.borrow().grad.unwrap(), 0.5));
        assert!(float_eq(x2w2.data.borrow().grad.unwrap(), 0.5));
        assert!(float_eq(x1w1x2w2.data.borrow().grad.unwrap(), 0.5));
        assert!(float_eq(b.data.borrow().grad.unwrap(), 0.5));
        assert!(float_eq(n.data.borrow().grad.unwrap(), 0.5));
        assert!(float_eq(o.data.borrow().grad.unwrap(), 1.0));
    }

    #[test]
    fn use_twice_simple() {
        let a = Value::new(2.0, true);
        let b = &a + &a;
        b.backward();
        assert_eq!(a.data.borrow().grad, Some(2.0));
        assert_eq!(b.data.borrow().grad, Some(1.0));

        let a = Value::new(2.0, true);
        let c = &a * &a;
        c.backward();
        assert_eq!(a.data.borrow().grad, Some(4.0));
        assert_eq!(c.data.borrow().grad, Some(1.0));
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
        let a = Value::new(-2.0, true);
        let b = Value::new(3.0, true);
        let d = &a * &b;
        let e = &a + &b;
        let f = &d * &e;
        f.backward();
        assert_eq!(a.data.borrow().grad, Some(-3.0));
        assert_eq!(b.data.borrow().grad, Some(-8.0));
        assert_eq!(d.data.borrow().grad, Some(1.0));
        assert_eq!(e.data.borrow().grad, Some(-6.0));
        assert_eq!(f.data.borrow().grad, Some(1.0));
    }
}
