use ndarray::{Array, Ix1};

pub struct AccuracyMetric{
    pub hits: f64,
    pub total: f64
}

impl AccuracyMetric {
    pub fn add_value(&mut self, correct: bool){
        if correct{
            self.hits += 1.0;
        }
        self.total += 1.0;
    }

    pub fn add_values(&mut self, correct:  Array<bool, Ix1>){
        for i in 0..correct.len() {
            if correct[i] {
                self.hits += 1.0;
            }
            self.total += 1.0;
        }
    }

    pub fn get_acc(&self) -> f64{
        self.hits/self.total
    }
}