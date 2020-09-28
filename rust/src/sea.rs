use ndarray::{s, Array, Ix1, Ix2};
use ndarray_rand::rand::prelude::SeedableRng;
use ndarray_rand::rand::rngs::StdRng;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
//use crate::base_generator::BaseGenerator;

#[derive(Debug)]
pub struct SeaGenerator {
    pub random_state: u64,
    pub function: u8,
    pub threshold: f64,
    rng: StdRng,
}


impl SeaGenerator{
    pub fn next_sample(&mut self, a: usize) -> (Array<f64, Ix2>, Array<usize, Ix1>) {
            let X = Array::random_using((a, 3), Uniform::new(0., 10.), &mut self.rng);
            let y = self.compare_threshold(&X);
            (X, y)
    }

//    fn get_n_atts(&self) -> u32 {
//        3
//    }
//
//    fn get_n_classes(&self) -> u32 {
//        2
//    }
}


impl SeaGenerator {
    pub fn new(function: u8, random_state: u64) -> SeaGenerator {
        let mut sea = SeaGenerator {
            random_state,
            function,
            threshold: 0.0,
            rng: SeedableRng::seed_from_u64(random_state),
        };
        sea.set_threshold();
        sea
    }

    pub fn set_threshold(&mut self) {
        if self.function == 0 {
            self.threshold = 8.0;
        } else if self.function == 1 {
            self.threshold = 9.0;
        } else if self.function == 2 {
            self.threshold = 7.0;
        } else if self.function == 3 {
            self.threshold = 9.5;
        }
    }

    fn compare_threshold(&self, X: &Array<f64, Ix2>) -> Array<usize, Ix1> {
        let q = X.slice(s![.., 0]);
        let r = X.slice(s![.., 1]);
        let sum = &q + &r;
        let h = sum.mapv(|sum| sum > self.threshold);
        set_where_true(h, 2)
    }

    pub fn get_n_atts() -> u32{
        3
    }

    pub fn get_n_classes() -> u32{
        2
    }
}

fn set_where_true(arr: Array<bool, Ix1>, val: usize) -> Array<usize, Ix1> {
    let mut y: Array<usize, Ix1> = Array::zeros(arr.len());
    for i in 0..arr.len() {
        if arr[i] {
            y[i] = 1;
        }
    }
    y
}
