//
// Created by Lucca Portes on 19/08/20.
//

#ifndef NBC___BASE_GENERATOR_H
#define NBC___BASE_GENERATOR_H

#include <random>
#include <utility>
#include <vector>

class BaseGenerator {
protected:
    std::mt19937 random_state;
public:
    virtual std::tuple<std::vector<std::vector<double>>, std::vector<int>> next_sample(int sample_size)=0;
    virtual int get_n_features()=0;
    virtual int get_n_classes()=0;
};

#endif //NBC___BASE_GENERATOR_H

//
// Created by Lucca Portes on 13/09/20.
//

#ifndef NBC___RANDOM_RBF_H
#define NBC___RANDOM_RBF_H

#include <utility>
#include <vector>

class Centroid{
public:
    Centroid(std::vector<double> center, double class_label, double std_dev): center(center), class_label(class_label), std_dev(std_dev){}
    std::vector<double> center;
    double class_label, std_dev;
};

class RandomRBFGenerator : public BaseGenerator{
public:
    std::tuple<std::vector<std::vector<double> >, std::vector<int> > next_sample(int sample_size) override;
    int get_n_features() override;
    int get_n_classes() override;
    RandomRBFGenerator (int random_state=42, int n_features=2, int n_classes=2, int n_centroids=50);
private:
    std::mt19937 random_state;
    int n_features, n_classes, n_centroids;
    std::vector<Centroid> centroids;
    std::vector<double> centroid_weights;
    std::discrete_distribution<int> dist_centroids;
    std::uniform_real_distribution<double> dist_att;
    void generate_centroids();
};

#endif //NBC___RANDOM_RBF_H

//
// Created by Lucca Portes on 19/08/20.
//

#ifndef NBC___SEA_GENERATOR_H
#define NBC___SEA_GENERATOR_H

#include <vector>
#include <tuple>

class SEAGenerator : public BaseGenerator{
public:
    SEAGenerator();
    explicit SEAGenerator(int random_state);
    SEAGenerator(int random_state, int function_number);
    std::tuple<std::vector<std::vector<double>>, std::vector<int>> next_sample(int sample_size) override;
    int get_n_features();
    int get_n_classes();

private:
    double threshold;
    std::vector<int> def_class(std::vector<std::vector<double> > X);
    double def_treshold(int function_number);

};

#endif //NBC___SEA_GENERATOR_H

//
// Created by Lucca Portes on 23/08/20.
//

#ifndef NBC___UTILS_H
#define NBC___UTILS_H

#include <vector>

std::vector<double> vec_scalar_sum(std::vector<double> v, double s);
std::vector<double> vec_scalar_mul(std::vector<double> v, double s);
void vec_scalar_mul_void(std::vector<double> &v, double s);
std::vector<double> vec_scalar_div(std::vector<double> v, double s);
void vec_scalar_div_void(std::vector<double> &v, double s);
std::vector<double> vec_scalar_pow(std::vector<double> v, double s);
void vec_scalar_pow_void(std::vector<double> &v, double s);
std::vector<double> vec_sqrt_all(std::vector<double> v);
void vec_sqrt_all_void(std::vector<double> &v);
std::vector<double> vec_exp_all(std::vector<double> v);
void vec_exp_all_void(std::vector<double> &v);
std::vector<double> vec_vec_sub(std::vector<double> &v1, std::vector<double> v2);
void vec_vec_sub_void(std::vector<double> &v1, std::vector<double> v2);
std::vector<double> vec_vec_mul(std::vector<double> v1, std::vector<double> v2);
void vec_vec_mul_void(std::vector<double> &v1, std::vector<double> v2);
// std::vector<double> ve
std::vector<double> vec_vec_div(std::vector<double> v1, std::vector<double> v2);
void vec_vec_div_void(std::vector<double> &v1, std::vector<double> v2);
std::vector<double> vec_vec_add(std::vector<double> v1, std::vector<double> v2);
void vec_vec_assign(std::vector<double> &v1, std::vector<double> v2);
// void ve
// _void(std::vector<double> &v1, std::vector<double> v2);
double productory(std::vector<double> v);
std::vector<int> argmax2d(std::vector<std::vector<double> > v);
int argmax1d(std::vector<double> v);
void print_vector(std::vector<double> v);

#endif //NBC___UTILS_H

//
// Created by Lucca Portes on 23/08/20.
//

#ifndef NBC___CLASSIFIER_H
#define NBC___CLASSIFIER_H

#include <vector>
#include <iostream>

class Classifier {
public:
    virtual std::vector<int> predict(std::vector<std::vector<double>> X){
        std::cout << "não devia estar aq" << std::endl;
        return std::vector<int>();
    }
    virtual void partial_fit(std::vector<std::vector<double>> X, std::vector<int> y){}
};

#endif //NBC___CLASSIFIER_H

//
// Created by Lucca Portes on 23/08/20.
//

#ifndef NBC___NAIVE_BAYES_H
#define NBC___NAIVE_BAYES_H

#include <vector>

class NaiveBayes : public Classifier{
private:
    std::vector<double> classes_count;
    std::vector<std::vector<double> > att_max_value;
    std::vector<std::vector<double> > att_mean_value;
    std::vector<std::vector<double> > att_var_value;
    std::vector<std::vector<double> > probs_buffer;

    std::vector<double> get_variances(int class_index);
    std::vector<double> get_stds(int class_index);
    std::vector<double> get_means(int class_index);

public:
    NaiveBayes(int class_number, int att_number);

    NaiveBayes();
    ~NaiveBayes(){}

    void partial_fit(std::vector<std::vector<double> > X, std::vector<int> y) override;
    std::vector<int> predict(std::vector<std::vector<double> > X) override;
    void predict_proba(std::vector<std::vector<double>> X);
    double score(std::vector<std::vector<double> > X, std::vector<int> y);

};

#endif //NBC___NAIVE_BAYES_H

//
// Created by Lucca Portes on 23/08/20.
//

#ifndef NBC___ACC_METRIC_H
#define NBC___ACC_METRIC_H

class AccMetric {
private:
    double hits = 0;
    double total = 0;
public:
    void add_value(bool correct);
    double get_acc();
};

#endif //NBC___ACC_METRIC_H

//
// Created by Lucca Portes on 23/08/20.
//

#ifndef NBC___PREQUENTIAL_H
#define NBC___PREQUENTIAL_H

//#include "naive_bayes.h"

#include <vector>
#include <tuple>

class Prequential {
private:
    Classifier *clf;
    BaseGenerator *gen;
    int n_wait, max_samples, batch_size;
    AccMetric metric_analyzer;
public:
    Prequential(Classifier *clf, BaseGenerator *gen, int n_wait, int max_samples, int batch_size);
    void update_metric(std::vector<int> y_true, std::vector<int> y_pred);
    void run();

};

#endif //NBC___PREQUENTIAL_H
//
// Created by Lucca Portes on 13/09/20.
//

RandomRBFGenerator::RandomRBFGenerator(int random_state, int n_features, int n_classes, int n_centroids) {
        this->random_state = std::mt19937(random_state);
        this->n_features = n_features;
        this->n_classes = n_classes;
        this->n_centroids = n_centroids;
        this->dist_att = std::uniform_real_distribution<double>(0, 1);
        this->generate_centroids();
        this->dist_centroids = std::discrete_distribution<int>(std::begin(this->centroid_weights), std::end(this->centroid_weights));
        
}

void RandomRBFGenerator::generate_centroids() {
    // std::uniform_real_distribution<double> dist_att(0, 1);
    std::uniform_int_distribution<int> dist_class(0,this->n_classes-1);

    for (int i = 0; i < this->n_centroids; i++){
        std::vector<double> centroid_coords;
        for (int j = 0; j < this->n_features; j++){
            centroid_coords.push_back(this->dist_att(this->random_state));
        }
        auto label = dist_class(this->random_state);
        auto std_dev = this->dist_att(this->random_state);
        this->centroids.emplace_back(centroid_coords, label, std_dev);
        this->centroid_weights.push_back(this->dist_att(this->random_state));
    }
}

std::tuple<std::vector<std::vector<double> >, std::vector<int> > RandomRBFGenerator::next_sample(int sample_size) {
    std::vector<std::vector<double> > X;
    X.reserve(sample_size);
    std::vector<int> y;

    for (int i=0; i < sample_size; i++){
        auto curr_centroid = this->centroids[this->dist_centroids(this->random_state)];
        std::vector<double> att_vals(this->n_features);
        double magnitude = 0.0;
        for (int j=0; j < this->n_features; j++){
            auto att = (this->dist_att(this->random_state) * 2.0) - 1.0;
            att_vals[j] = att;
            magnitude += att * att;
        }
        magnitude = std::sqrt(magnitude);
        double desired_mag = std::normal_distribution<>(0, curr_centroid.std_dev)(this->random_state);
        double scale = desired_mag / magnitude;
        for (int i = 0; i < att_vals.size(); i++){
            att_vals[i] = curr_centroid.center[i] + att_vals[i] * scale;
        }
        X.push_back(att_vals);
        y.push_back(curr_centroid.class_label);
    }
    return std::make_tuple(X, y);
}

int RandomRBFGenerator::get_n_features() {
    return this->n_features;
}

int RandomRBFGenerator::get_n_classes() {
    return this->n_classes;
}

//
// Created by Lucca Portes on 19/08/20.
//

#include <random>
#include <iostream>

SEAGenerator::SEAGenerator(int random_state) : BaseGenerator() {
    this->random_state = std::mt19937(random_state);
    this->threshold = this->def_treshold(0);
}

SEAGenerator::SEAGenerator(int random_state, int function_number) : BaseGenerator() {
    this->random_state = std::mt19937(random_state);
    this->threshold = this->def_treshold(function_number);
}

SEAGenerator::SEAGenerator() : BaseGenerator() {
    this->random_state = std::mt19937(42);
    this->threshold = this->def_treshold(0);;
}

std::tuple<std::vector<std::vector<double>>, std::vector<int>> SEAGenerator::next_sample(int sample_size) {
    std::vector<std::vector<double>> X;
    X.reserve(sample_size);

    std::uniform_real_distribution<double> dist(1.0, 10.0);

    for (int i=0; i < sample_size; i++){
        std::vector<double> inside_vec {dist(this->random_state), dist(this->random_state), dist(this->random_state)};
        X.push_back(inside_vec);
    }

    auto y = def_class(X);

    return std::make_tuple(X, y);
}

std::vector<int> SEAGenerator::def_class(std::vector<std::vector<double> > X) {
    auto y = std::vector<int>();
    y.reserve(X.size());
    for (auto i: X) {
        if (i[0] + i[1] > this->threshold){
            y.push_back(0);
        } else{
            y.push_back(1);
        }
    }
    return y;
}

double SEAGenerator::def_treshold(int function_number) {
    if (function_number == 0){
        return 8.0;
    } else if (function_number == 1){
        return 9.0;
    } else if (function_number == 2){
        return 7.0;
    } else if (function_number == 3){
        return 9.5;
    }
    return -1;
}

int SEAGenerator::get_n_features() {
    return 3;
}

int SEAGenerator::get_n_classes() {
    return 2;
}//
// Created by Lucca Portes on 23/08/20.
//

#include <vector>
#include <cmath>
#include <iostream>

std::vector<double> vec_scalar_sum(std::vector<double> v, double s) {
    std::vector<double> ret(v.size());
    for (auto i = 0; i < v.size(); i++)
        ret[i] = v[i] + s;
    return ret;
}

std::vector<double> vec_scalar_div(std::vector<double> v, double s) {
    std::vector<double> ret(v.size());
    for (auto i = 0; i < v.size(); i++)
        ret[i] = v[i] / s;
    return ret;
}

void vec_scalar_div_void(std::vector<double> &v, double s) {
    for (auto i = 0; i < v.size(); i++)
        v[i] = v[i] / s;
}


std::vector<double> vec_sqrt_all(std::vector<double> v) {
    std::vector<double> ret(v.size());
    for (auto i = 0; i < v.size(); i++)
        ret[i] = std::sqrt(v[i]);
    return ret;
}

void vec_sqrt_all_void(std::vector<double> &v) {
    for (auto i = 0; i < v.size(); i++)
        v[i] = std::sqrt(v[i]);
}

std::vector<double> vec_exp_all(std::vector<double> v) {
    std::vector<double> ret(v.size());
    for (auto i = 0; i < v.size(); i++)
        ret[i] = std::exp(v[i]);
    return ret;
}

void vec_exp_all_void(std::vector<double> &v) {
    for (auto i = 0; i < v.size(); i++)
        v[i] = std::exp(v[i]);
}

std::vector<double> vec_vec_sub(std::vector<double> &v1, std::vector<double> v2) {
    std::vector<double> ret(v1.size());
    for (auto i = 0; i < v1.size(); i++)
        ret[i] = v1[i] - v2[i];
    return ret;
}

void vec_vec_sub_void(std::vector<double> &v1, std::vector<double> v2) {
    // std::vector<double> ret(v1.size());
    for (auto i = 0; i < v1.size(); i++)
        v1[i] = v1[i] - v2[i];
    // return ret;
}

std::vector<double> vec_vec_mul(std::vector<double> v1, std::vector<double> v2) {
    std::vector<double> ret(v1.size());
    for (auto i = 0; i < v1.size(); i++)
        ret[i] = v1[i] * v2[i];
    return ret;
}

void vec_vec_mul_void(std::vector<double> &v1, std::vector<double> v2) {
    // std::vector<double> ret(v1.size());
    for (auto i = 0; i < v1.size(); i++)
        v1[i] = v1[i] * v2[i];
    // return ret;
}

std::vector<double> vec_scalar_mul(std::vector<double> v, double s) {
    std::vector<double> ret(v.size());
    for (auto i = 0; i < v.size(); i++)
        ret[i] = v[i] * s;
    return ret;
}

void vec_scalar_mul_void(std::vector<double> &v, double s) {
    for (auto i = 0; i < v.size(); i++)
        v[i] = v[i] * s;
}

std::vector<double> vec_scalar_pow(std::vector<double> v, double s) {
    std::vector<double> ret(v.size());
    for (auto i = 0; i < v.size(); i++)
        ret[i] = std::pow(v[i], s);
    return ret;
}

void vec_scalar_pow_void(std::vector<double> &v, double s) {
    for (auto i = 0; i < v.size(); i++)
        v[i] = std::pow(v[i], s);
}

// std::vector<double> ve
std::vector<double> vec_vec_div(std::vector<double> v1, std::vector<double> v2) {
    std::vector<double> ret(v1.size());
    for (auto i = 0; i < v1.size(); i++)
//        if (v2[i] != 0) {
            ret[i] = v1[i] / v2[i];
//        } else{
//            ret[i] = 0;
//        }
    return ret;
}

std::vector<double> vec_vec_add(std::vector<double> v1, std::vector<double> v2) {
    std::vector<double> ret(v1.size());
    for (auto i = 0; i < v1.size(); i++)
            ret[i] = v1[i] + v2[i];
    return ret;
}

void vec_vec_assign(std::vector<double> &v1, std::vector<double> v2) {
    for (auto i = 0; i < v1.size(); i++)
            v1[i] = v2[i];
}

void vec_vec_div_void(std::vector<double> &v1, std::vector<double> v2) {
    for (auto i = 0; i < v1.size(); i++)
            v1[i] = v1[i] / v2[i];
}

double productory(std::vector<double> v) {
    auto val = 1.0;
    for (auto i = 0; i < v.size(); i++) {
        val *= v[i];
    }
    return val;
}

int argmax1d(std::vector<double> v){
    auto max_index = 0;
    for (auto i = 0; i < v.size(); i++) {
        if (v[max_index] < v[i]){
                    max_index = i;
            }
    }
    return max_index;
}

std::vector<int> argmax2d(std::vector<std::vector<double> > v){
    std::vector<int> ret(v.size());
    for (auto i = 0; i < v.size(); i++)
        ret[i] = argmax1d(v[i]);
    return ret;
}

void print_vector(std::vector<double> v){
    for (auto i : v) {
        std::cout << i << " ";
    }
    std::cout << std::endl;
}//
// Created by Lucca Portes on 23/08/20.
//

#include <numeric>
#include <cmath>

NaiveBayes::NaiveBayes(int class_number, int att_number) {
    this->probs_buffer = std::vector<std::vector<double> >(1, std::vector<double>(att_number, 0.0));
    for (int i = 0; i < class_number; i++){
        this->classes_count.push_back(0.0);
        this->att_max_value.emplace_back(att_number, 0);
        this->att_mean_value.emplace_back(att_number, 0);
        this->att_var_value.emplace_back(att_number, 0);
    }
}

void NaiveBayes::partial_fit(std::vector<std::vector<double> > X, std::vector<int> y) {
    for (auto row_index = 0 ; row_index < X.size(); row_index++){
        auto curr_array = X[row_index];
        auto curr_class = y[row_index];
        this->classes_count[curr_class]++;
        for (auto col_index = 0; col_index < curr_array.size(); col_index++) {
            auto last_mean = this->att_mean_value[curr_class][col_index];
            this->att_mean_value[curr_class][col_index] +=
                    (curr_array[col_index] - last_mean) / this->classes_count[curr_class];
            this->att_var_value[curr_class][col_index] +=
                    (curr_array[col_index] - last_mean) * (curr_array[col_index] - this->att_mean_value[curr_class][col_index]);
        }
        
    }
}

std::vector<int> NaiveBayes::predict(std::vector<std::vector<double> > X) {
    this->predict_proba(X);
    return argmax2d(this->probs_buffer);
}

void NaiveBayes::predict_proba(std::vector<std::vector<double> > X) {
    if (this->probs_buffer.size() < X.size()){
        this->probs_buffer.reserve(X.size()-1);
        for (int i=0; i < X.size()-1; i++){
            this->probs_buffer.push_back(
                std::vector<double>(this->att_mean_value.size(), 0.0)
                );
        }
    }
    int sum_inst = std::accumulate(this->classes_count.begin(), this->classes_count.end(), 0);
    auto n_features = X[0].size();
    for (auto index = 0; index < X.size(); index++){
        auto curr_inst = X[index];
        this->probs_buffer[index] = vec_scalar_div(this->classes_count, sum_inst);
        // vec_vec_assign(this->probs_buffer[index], this->classes_count);
        // vec_scalar_div_void(this->probs_buffer[index], sum_inst);

        for (auto class_ind = 0; class_ind < this->classes_count.size(); class_ind++){
            auto std_dev = this->get_stds(class_ind);
            auto diff = vec_vec_sub(curr_inst, this->att_mean_value[class_ind]);

            vec_scalar_pow_void(diff, 2);

            auto pdf = vec_scalar_mul(std_dev, 2.5066282746310002);
            vec_scalar_pow_void(pdf, -1);

            vec_scalar_pow_void(std_dev, 2);

            vec_scalar_mul_void(std_dev, 2);

            vec_vec_div_void(diff, std_dev);

            vec_scalar_mul_void(diff, -1);

            vec_exp_all_void(diff);

            vec_vec_mul_void(pdf, diff);

            this->probs_buffer[index][class_ind] *= productory(pdf);
        }
    }
    // return probs;
}

std::vector<double> NaiveBayes::get_variances(int class_index) {
    return vec_scalar_div(this->att_var_value[class_index], this->classes_count[class_index]);
}

std::vector<double> NaiveBayes::get_stds(int class_index) {
    auto vars = this->get_variances(class_index);
    // vec_sqrt_all_void(vars);
    vars = vec_sqrt_all(vars);
    return vars;
}

std::vector<double> NaiveBayes::get_means(int class_index) {
    return this->att_mean_value[class_index];
}

double NaiveBayes::score(std::vector<std::vector<double> > X, std::vector<int> y) {
    auto preds = this->predict(X);
    double hits = 0;
    for (auto i=0; i < y.size(); i++){
        if (preds[i] == y[i]){
            hits += 1;
        }
    }
    return hits/y.size();
}

NaiveBayes::NaiveBayes() {}

//
// Created by Lucca Portes on 23/08/20.
//

void AccMetric::add_value(bool correct) {
    if (correct){
        this->hits++;
    }
    this->total++;
}

double AccMetric::get_acc() {
    if (this->total == 0)
        return 0.0;
    return this->hits / this->total;
}
//
// Created by Lucca Portes on 23/08/20.
//

#include <vector>
#include <numeric>
#include <iostream>

Prequential::Prequential(Classifier *clf, BaseGenerator *gen, int n_wait, int max_samples, int batch_size) {
    this->clf = clf;
    this->gen = gen;
    this->n_wait = n_wait;
    this->max_samples = max_samples;
    this->batch_size = batch_size;
}

void Prequential::update_metric(std::vector<int> y_true, std::vector<int> y_pred) {
    for (auto i = 0; i < y_pred.size(); i++){
        this->metric_analyzer.add_value(y_pred[i] == y_true[i]);
    }
}

void Prequential::run() {
    int count = 0;
    int count_n_wait = 0;
    std::vector<double> vec_accs;
    while (count < this->max_samples) {
        std::vector<std::vector<double>> X;
        std::vector<int> y;
        std::tie(X, y) = this->gen->next_sample(this->batch_size);
        auto preds = this->clf->predict(X);
        this->update_metric(y, preds);
        this->clf->partial_fit(X, y);
        count_n_wait += this->batch_size;
        count += this->batch_size;
        if (count_n_wait == this->n_wait){
            vec_accs.push_back(this->metric_analyzer.get_acc());
            count_n_wait = 0;
        }
    }
    double sum = std::accumulate(vec_accs.begin(), vec_accs.end(), 0.0);
    double mean = sum / vec_accs.size();
    std::cout << "Accurracy is " << mean << std::endl;
}

void run_prequential(std::string clf, std::string gen, int n_wait, int max_samples, int batch_size){
    NaiveBayes clf_obj;
    SEAGenerator gen_obj;
    if (clf == "NB"){
        clf_obj = NaiveBayes(2, 3);
    } else{
        throw std::invalid_argument("Classifier not available");
    }
    if (gen == "SEA"){
        gen_obj = SEAGenerator(42, 0);
    } else{
        throw std::invalid_argument("Generator not available");
    }
    auto preq = Prequential(&clf_obj, &gen_obj, n_wait, max_samples, batch_size);
    preq.run();
}

void run_prequentialRBF(std::string clf, std::string gen, int n_wait, int max_samples, int batch_size, int n_features, int n_classes){
    NaiveBayes clf_obj;
    RandomRBFGenerator gen_obj;
    if (clf == "NB"){
        clf_obj = NaiveBayes(n_classes, n_features);
    } else{
        throw std::invalid_argument("Classifier not available");
    }
    if (gen == "RBF"){
        gen_obj = RandomRBFGenerator(42, n_features, n_classes, 50);
    } else{
        throw std::invalid_argument("Generator not available");
    }
    auto preq = Prequential(&clf_obj, &gen_obj, n_wait, max_samples, batch_size);
    preq.run();
}
