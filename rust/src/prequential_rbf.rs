//#[path = "./"] mo]d sea;
use ndarray::{Array, Ix1};
use crate::random_rbf::RandomRBFGenerator;
use crate::naive_bayes::NaiveBayes;
use crate::acc_metric::AccuracyMetric;
//use crate::base_generator::BaseGenerator;
use std::vec::Vec;


pub struct PrequentialRBF {
    n_wait: usize,
    max_samples: usize,
    batch_size: usize,
    generator: RandomRBFGenerator,
    clf: NaiveBayes,
    metric_analyzer: AccuracyMetric
}

impl PrequentialRBF {
    pub fn new(clf: NaiveBayes, generator: RandomRBFGenerator, n_wait: usize, max_samples: usize, batch_size:usize) -> PrequentialRBF {
        PrequentialRBF {
            n_wait,
            max_samples,
            generator,
            batch_size,
            clf,
            metric_analyzer:AccuracyMetric{ hits: 0.0, total: 0.0 }
        }
    }

    pub fn run(&mut self){
        let mut count: usize = 0;
        let mut count_n_wait : usize = 0;
        let mut vec_accs: Vec<f64> = Vec::new();
        while count < self.max_samples{
            let (x, y) = self.generator.next_sample(self.batch_size);
            let preds = self.clf.predict(&x);
            self.update_metric(&y, &preds);
            self.clf.partial_fit(&x, &y);
            count_n_wait += self.batch_size;
            count += self.batch_size;
            if count_n_wait == self.n_wait{
                vec_accs.push(self.metric_analyzer.get_acc());
                count_n_wait = 0;
            }
        }
        println!("Accurracy is {}", vec_accs.iter().sum::<f64>() / vec_accs.len() as f64)

    }

    fn update_metric(&mut self, y_true: &Array<usize, Ix1>, y_pred: &Array<usize, Ix1>){
        for i in 0..y_pred.len(){
            self.metric_analyzer.add_value(y_pred[i] == y_true[i]);
        }
    }
}
