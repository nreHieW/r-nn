use crate::core::Value;

pub struct BaseSGD<'a> {
    pub lr: f32,
    parameters: Vec<&'a Value>,
}

impl<'a> BaseSGD<'a> {
    pub fn new(parameters: Vec<&'a Value>, lr: f32) -> Self {
        if lr <= 0.0 {
            panic!("Learning rate should be positive");
        }
        Self { lr, parameters }
    }

    pub fn step(&self) {
        for param in &self.parameters {
            param.update(-self.lr);
        }
    }

    pub fn zero_grad(&self) {
        for param in &self.parameters {
            param.zero_grad();
        }
    }

    pub fn _update_lr(&mut self, lr: f32) {
        self.lr = lr;
    }
}
