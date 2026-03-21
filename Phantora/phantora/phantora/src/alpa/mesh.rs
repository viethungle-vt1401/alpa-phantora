#[derive(Clone)]
pub struct DeviceMesh {
    pub devices: Vec<usize>,
}

impl DeviceMesh {
    pub fn new(n: usize) -> Self {
        Self {
            devices: (0..n).collect(),
        }
    }
}