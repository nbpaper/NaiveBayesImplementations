use ndarray::{s, Array, Ix1, Ix2};
use ndarray_rand::rand::prelude::SeedableRng;
use ndarray_rand::rand::rngs::StdRng;
use ndarray_rand::rand_distr::{Distribution, Uniform, Normal};
use rand::distributions::WeightedIndex;
use ndarray_rand::RandomExt;
//use crate::base_generator::BaseGenerator;

#[derive(Debug)]
pub struct RandomRBFGenerator {
    pub random_state: u64,
    pub n_features: u32,
    pub n_classes: u32,
    pub n_centroids: usize,
    centroids: Vec<Centroid>,
    centroid_weights: Vec<f64>,
    rng: StdRng,
}

#[derive(Debug)]
struct Centroid {
    std_dev: f64,
    class_label: usize,
    center: Array<f64, Ix1>
}

impl RandomRBFGenerator{
    pub fn next_sample(&mut self, a: usize) -> (Array<f64, Ix2>, Array<usize, Ix1>) {
        let uniform_dist = Uniform::new(0, self.n_classes);
        let normal_dist = Normal::new(0.0, 1.0).unwrap();
        let weight_dist = WeightedIndex::new(&self.centroid_weights).unwrap();
//        println!("{:?}", self.centroid_weights);
        let mut X: Array<f64, Ix2> = Array::zeros((a, self.n_features as usize));
        let mut y: Array<usize, Ix1> = Array::zeros(a);


        for i in 0..a {
            let curr_centroid = &self.centroids[weight_dist.sample(&mut self.rng) as usize];
            let mut att_vals = Array::random_using(self.n_features as usize, Uniform::new(0.0, 1.), &mut self.rng);
            att_vals = &att_vals * 2.0 - 1.0;
            let magnitude = ((&att_vals * &att_vals).sum() as f64).sqrt();
            let desired_magnitude = normal_dist.sample(&mut self.rng);
            let scale = desired_magnitude / magnitude;
            att_vals = att_vals * scale;
            att_vals += &curr_centroid.center;
            X.slice_mut(s![i, ..]).assign(&att_vals);
            y[i] = curr_centroid.class_label;
        }
        (X, y)
    }

    fn get_n_atts(&self) -> u32 {
        self.n_features
    }

    fn get_n_classes(&self) -> u32 {
        self.n_classes
    }
}


impl RandomRBFGenerator {
    pub fn new(random_state: u64, n_features: u32, n_classes: u32, n_centroids: usize) -> RandomRBFGenerator {
        let mut rbf = RandomRBFGenerator {
            random_state,
            n_features,
            n_classes,
            n_centroids,
            centroids: Vec::new(),
            centroid_weights: Vec::new(),
            rng: SeedableRng::seed_from_u64(random_state),
        };
        rbf.generate_centroids();
        rbf
    }

    fn generate_centroids(&mut self) {
        let coords = Array::random_using((self.n_centroids as usize, self.n_features as usize), Uniform::new(0., 1.), &mut self.rng);
        let labels = Array::random_using(self.n_centroids as usize, Uniform::new(0, self.n_classes), &mut self.rng);
        let std_devs = Array::random_using(self.n_centroids as usize, Uniform::new(0.0, 1.), &mut self.rng);
        let centroid_weights = Array::random_using(self.n_centroids as usize, Uniform::new(0.0, 1.), &mut self.rng);

        for i in 0usize..self.n_centroids {
            self.centroids.push(Centroid{
                std_dev:std_devs[i],
                class_label: labels[i] as usize,
                center: &coords.slice(s![i, ..]) * 1.0
            });
            self.centroid_weights.push(centroid_weights[i]);

        }
    }

//    pub fn get_n_atts() -> u32{
//        3
//    }
//
//    pub fn get_n_classes() -> u32{
//        2
//    }
}