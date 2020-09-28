use ndarray::{ArrayD, ArrayViewD, ArrayViewMutD, Ix2};
use numpy::{IntoPyArray, PyArrayDyn, PyArray2, PyArray1};
use pyo3::prelude::{pymodule, Py, PyModule, PyResult, Python};
use pyo3::wrap_pyfunction;
//mod sea;
//use skmultiflow_fast::sea;
//use skmultiflow_fast::naive_bayes;
//use skmultiflow_fast::acc_metric;
//use skmultiflow_fast::prequential;
//use skmultiflow_fast::prequential;
mod prequential_rbf;
mod prequential_sea;
mod sea;
mod random_rbf;
mod acc_metric;
mod naive_bayes;

//#[pyfunction]
#[pymodule]
fn prequential_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    #[pyfn(m, "run_prequential")]
    fn run_prequential(py: Python, n_wait: usize, max_samples: usize, batch_size: usize) {
        let mut preq = prequential_sea::PrequentialSEA::new(
        naive_bayes::NaiveBayes::new(2, 3),
        sea::SeaGenerator::new(1, 89),
        n_wait,
        max_samples,
        batch_size
    );
        preq.run();
    }

    #[pyfn(m, "run_prequential_RBF")]
    fn run_prequential_RBF(py: Python, n_wait: usize, max_samples: usize, batch_size: usize, n_features: u32, n_classes: u32) {
        let mut preq = prequential_rbf::PrequentialRBF::new(
        naive_bayes::NaiveBayes::new(n_classes as usize, n_features as usize),
        random_rbf::RandomRBFGenerator::new(1, n_features, n_classes, 50),
        n_wait,
        max_samples,
        batch_size
    );
        preq.run();
    }

    #[pyfn(m, "run_partial_fit")]
    fn run_partial_fit(n: usize){
        let mut nb = naive_bayes::NaiveBayes::new(2, 3);
        let mut s = sea::SeaGenerator::new(1, 89);
        let (x, y) = s.next_sample(n);
        nb.partial_fit(&x, &y);
    }

    #[pyfn(m, "run_predict")]
    fn run_predict(n: usize){
        let mut nb = naive_bayes::NaiveBayes::new(2, 3);
        let mut s = sea::SeaGenerator::new(1, 89);
        let (x, y) = s.next_sample(n);
        nb.predict(&x);
    }

    Ok(())
}


//#[pymodule]
//fn myrustlib(_py: Python, m: &PyModule) -> PyResult<()> {
//    m.add_wrapped(wrap_pyfunction!(sea_next_sample))?;
//
//    Ok(())
//}
