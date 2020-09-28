use ndarray::{Array, Ix1, Ix2, s, ArrayView};
//use ndarray_stats::SummaryStatisticsExt::sum;

#[derive(Debug)]
pub struct NaiveBayes {
    pub classes_count: Array<f64, Ix1>,
    pub att_max_value: Array<f64, Ix2>,
    pub att_mean_value: Array<f64, Ix2>,
    pub att_var_value: Array<f64, Ix2>
}

impl NaiveBayes {
    pub fn new(class_number: usize, n_att: usize) -> NaiveBayes{
        NaiveBayes {
            classes_count: Array::zeros(class_number),
            att_max_value: Array::zeros((class_number, n_att)),
            att_mean_value: Array::zeros((class_number, n_att)),
            att_var_value: Array::zeros((class_number, n_att))
        }
    }

    pub fn partial_fit(&mut self, x: &Array<f64, Ix2>, y: &Array<usize, Ix1>){
        let mut curr_class: usize;
        let mut curr_arr: ArrayView<f64, Ix1>;
        let mut last_mean: f64;
        for i in 0..x.nrows(){
            curr_arr = x.slice(s![i, ..]);
            curr_class = y[i] as usize;
            self.classes_count[curr_class] += 1.0;
            for j in 0..self.att_mean_value.ncols() {
                last_mean = self.att_mean_value[[curr_class, j]];
                self.att_mean_value[[curr_class, j]] +=
                    (curr_arr[j] - last_mean) /  self.classes_count[curr_class] as f64;
                self.att_var_value[[curr_class, j]] +=
                    (curr_arr[j] - last_mean) * (curr_arr[j] - self.att_mean_value[[curr_class, j]]);
            }
        }
    }

//    pub fn partial_fit2(&mut self, X: Array<f64, Ix2>, y: Array<u32, Ix1>){
//        let mut curr_class: usize;
//        let mut curr_arr: ArrayView<f64, Ix1>;
//        let mut last_mean: ArrayView<f64, Ix1>;
//        for i in 0..X.nrows(){
//            curr_arr = X.slice(s![i, ..]);
//            curr_class = y[i] as usize;
//            self.classes_count[curr_class] += 1.0;
//            last_mean = self.att_mean_value.slice(s![curr_class, ..]);
//            last_mean = last_mean.clone();
//            let mean_att: Array<f64, Ix1> = &self.att_mean_value.slice(s![curr_class, ..]) +
//                &((&curr_arr - &last_mean) / self.classes_count[curr_class]);
////            println!("{:?}", val);
//            self.att_mean_value.slice_mut(s![curr_class, ..]).assign(&mean_att);
//            let mean_att = mean_att.clone();
//            let var_att: Array<f64, Ix1> = &self.att_var_value.slice(s![curr_class, ..]) +
//                &((&curr_arr - &last_mean) *
//                    (&curr_arr - &mean_att.clone()));
//

//            self.att_var_value.slice_mut(s![curr_class, ..]).assign(
//                &self.att_mean_value.slice(s![curr_class, ..]) +
//
//            );
//            for j in 0..self.att_mean_value.ncols() {
//                last_mean = self.att_mean_value[[curr_class, j]];
//                self.att_mean_value[[curr_class, j]] +=
//                    (curr_arr[j] - last_mean) /  self.classes_count[curr_class] as f64;
//                self.att_var_value[[curr_class, j]] +=
//                    (curr_arr[j] - last_mean) * (curr_arr[j] - self.att_mean_value[[curr_class, j]]);
//            }
//        }
//    }

    pub fn predict(&mut self, x: &Array<f64, Ix2>) -> Array<usize, Ix1>{
        let probs = self.predict_proba(x);
        argmax2d(probs)
    }

    pub fn predict_proba(&mut self, x: &Array<f64, Ix2>) -> Array<f64, Ix2>{
        let sum_inst = self.classes_count.sum();
        let mut probs : Array<f64, Ix2> = Array::zeros((x.nrows(), self.classes_count.len()));
        let n_features = x.ncols();
        for i in 0..x.nrows(){
            let curr_inst = x.slice(s![i, ..]);
            probs.slice_mut(s![i, ..]).assign( &(&self.classes_count/sum_inst));
            for class in 0..self.classes_count.len(){
                let std_dev = self.get_stds(class);
                let mean = self.get_means(class);
                let diff : Array<f64, Ix1> = &curr_inst - &mean;
                let mut pdf : Array<f64, Ix1> = (&std_dev * 2.5066282746310002).mapv(|a| a.powi(-1));
                pdf = pdf * ((&diff * &diff / (&std_dev * &std_dev * 2.0)) * -1.0).mapv(f64::exp);
                probs[[i, class]] *= productory(pdf);
            }
        }
        probs
    }

    pub fn score(&mut self, x: &Array<f64, Ix2>, y: &Array<usize, Ix1>) -> f64{
        let preds = self.predict(&x);
        let mut hits = 0.0;
        for i in 0..preds.len(){
            if preds[i] == y[i]{
                hits += 1.0;
            }
        }
        hits/y.len() as f64
    }

    fn get_variances(&self, class_index: usize) -> Array<f64, Ix1>{
        &self.att_var_value.slice(s![class_index, ..]) / self.classes_count[class_index]
    }

    fn get_stds(&self, class_index: usize) -> Array<f64, Ix1>{
        self.get_variances(class_index).mapv(f64::sqrt)
    }

    fn get_means(&self, class_index: usize) -> Array<f64, Ix1>{
        &self.att_mean_value.slice(s![class_index, ..]) * 1.0
    }
}

fn productory(x: Array<f64, Ix1>) -> f64{
    let mut val = 1.0;
    for i in 0..x.len(){
        val *= x[i];
    }
    val
}

fn argmax2d(x: Array<f64, Ix2>) -> Array<usize, Ix1>{
    let mut res : Array<usize, Ix1> = Array::zeros(x.nrows());
    for i in 0..x.nrows(){
        res[i] = argmax1d(&x.slice(s![i, ..])*1.0);
    }
    res
}

fn argmax1d(x: Array<f64, Ix1>) -> usize{
    let mut max_index = 0;
    for (i, v) in x.iter().enumerate().skip(1) {
        if x[max_index] < *v {
            max_index = i;
        }
    }
    max_index
}

//def get_variance(self):
//    return self._variance_sum / (self._weight_sum - 1.0) if self._weight_sum > 1.0 else 0.0


