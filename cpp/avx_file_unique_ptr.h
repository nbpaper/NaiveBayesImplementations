#include <immintrin.h>
#include <iostream>
#include <memory>
#include <vector>
#include <random>
#include <cmath>
#include <stdexcept>

class AVXArray{
public:
    inline static AVXArray zeros(int shape);
    inline void assign(const AVXArray &b);
    inline double sum();
    inline double prod();
    inline double max();
    inline int argmax();
    inline void sqrt_void();
    inline AVXArray sqrt();
    inline AVXArray exp();
    inline void exp_void();
    inline AVXArray operator+(const AVXArray& b);
    inline void add_void(const AVXArray& b);
    inline AVXArray operator-(const AVXArray& b);
    inline void sub_void(const AVXArray& b);
    inline AVXArray operator/(const AVXArray& b);
    inline AVXArray operator/(const double& b);
    inline void div_void (const AVXArray& b);
    inline void div_void (const double & b);
    inline friend AVXArray operator/(const double & b, const AVXArray& arr_orig);
    inline void be_div_by_void(const double & b);
    inline AVXArray operator*(const AVXArray& b);
    inline AVXArray operator*(const double & b);
    inline void mul_void (const AVXArray& b);
    inline void mul_void (const double & b);
    inline double operator[] (int index);
    inline void set(double v, int index);
    inline friend std::ostream & operator<< (std::ostream &out, const AVXArray &c);
    inline int Size() const;

   private:
   int size;
    int real_size;
    // __m256d* array;
    std::unique_ptr<__m256d[]> array;
};

AVXArray AVXArray::zeros(int shape) {
    AVXArray arr = AVXArray();
    arr.size = shape;
    int real_size = std::ceil(shape / 4.0);
    arr.real_size = real_size;
    arr.array = std::make_unique<__m256d[]>(shape);
    for (int i = 0; i < real_size; i++){
        arr.array[i] = _mm256_set1_pd(0);
    }
    return arr;
}

void AVXArray::assign(const AVXArray &b) {
    if (this->size != b.size){
        throw std::out_of_range ("Arrays must be of the same length to be summed");
    }
    for (int i=0; i < this->real_size; i++){
        this->array[i] = b.array[i];
    }
}


AVXArray AVXArray::operator+(const AVXArray &b) {
    if (this->size != b.size){
        throw std::out_of_range ("Arrays must be of the same length to be summed");
    }
    AVXArray arr = AVXArray();
    arr.size = this->size;
    arr.real_size = this->real_size;
    arr.array = std::make_unique<__m256d[]>(this->real_size);
    for (int i=0; i < this->real_size; i++){
        arr.array[i] = _mm256_add_pd(this->array[i], b.array[i]);
    }
    return arr;
}

void AVXArray::add_void(const AVXArray &b) {
    if (this->size != b.size){
        throw std::out_of_range ("Arrays must be of the same length to be summed");
    }
    for (int i=0; i < this->real_size; i++){
        this->array[i] = _mm256_add_pd(this->array[i], b.array[i]);
    }
}

AVXArray AVXArray::operator-(const AVXArray &b) {
    if (this->size != b.size){
        throw std::out_of_range ("Arrays must be of the same length to be summed");
    }
    AVXArray arr = AVXArray();
    arr.size = this->size;
    arr.real_size = this->real_size;
    arr.array = std::make_unique<__m256d[]>(this->real_size);
    for (int i=0; i < this->real_size; i++){
        arr.array[i] = _mm256_sub_pd(this->array[i], b.array[i]);
    }
    return arr;
}

void AVXArray::sub_void(const AVXArray &b) {
    if (this->size != b.size){
        throw std::out_of_range ("Arrays must be of the same length to be summed");
    }
    for (int i=0; i < this->real_size; i++){
        this->array[i] = _mm256_sub_pd(this->array[i], b.array[i]);
    }
}

AVXArray AVXArray::operator/(const AVXArray &b) {
    if (this->size != b.size){
        throw std::out_of_range ("Arrays must be of the same length to be summed");
    }
    AVXArray arr = AVXArray();
    arr.size = this->size;
    arr.real_size = this->real_size;
    arr.array = std::make_unique<__m256d[]>(this->real_size);
    for (int i=0; i < this->real_size; i++){
        arr.array[i] = _mm256_div_pd(this->array[i], b.array[i]);
    }
    return arr;
}

AVXArray AVXArray::operator/(const double &b) {
    __m256d scalar_array = _mm256_set1_pd(b);
    AVXArray arr = AVXArray();
    arr.size = this->size;
    arr.real_size = this->real_size;
    arr.array = std::make_unique<__m256d[]>(this->real_size);
    for (int i=0; i < this->real_size; i++){
        arr.array[i] = _mm256_div_pd(this->array[i], scalar_array);
    }
    return arr;
}

void AVXArray::div_void(const AVXArray &b) {
    if (this->size != b.size){
        throw std::out_of_range ("Arrays must be of the same length to be summed");
    }
    for (int i=0; i < this->real_size; i++){
        this->array[i] = _mm256_div_pd(this->array[i], b.array[i]);
    }
}

void AVXArray::div_void(const double &b) {
    __m256d scalar_array = _mm256_set1_pd(b);
    for (int i=0; i < this->real_size; i++){
        this->array[i] = _mm256_div_pd(this->array[i], scalar_array);
    }
}

AVXArray operator/(const double &b, const AVXArray& arr_orig) {
    __m256d scalar_array = _mm256_set1_pd(b);
    AVXArray arr = AVXArray();
    arr.size = arr_orig.size;
    arr.real_size = arr_orig.real_size;
    arr.array = std::make_unique<__m256d[]>(arr_orig.real_size);
    for (int i=0; i < arr_orig.real_size; i++){
        arr.array[i] = _mm256_div_pd(scalar_array, arr_orig.array[i]);
    }
    return arr;
}

void AVXArray::be_div_by_void(const double &b) {
    __m256d scalar_array = _mm256_set1_pd(b);
    for (int i=0; i < this->real_size; i++){
        this->array[i] = _mm256_div_pd(scalar_array, this->array[i]);
    }
}

AVXArray AVXArray::operator*(const AVXArray &b) {
    if (this->size != b.size){
        throw std::out_of_range ("Arrays must be of the same length to be summed");
    }
    AVXArray arr = AVXArray();
    arr.size = this->size;
    arr.real_size = this->real_size;
    arr.array = std::make_unique<__m256d[]>(this->real_size);
    for (int i=0; i < this->real_size; i++){
        arr.array[i] = _mm256_mul_pd(this->array[i], b.array[i]);
    }
    return arr;
}

void AVXArray::mul_void(const AVXArray &b) {
    if (this->size != b.size){
        throw std::out_of_range ("Arrays must be of the same length to be summed");
    }
    for (int i=0; i < this->real_size; i++){
        this->array[i] = _mm256_mul_pd(this->array[i], b.array[i]);
    }
}


//
int AVXArray::Size() const {
    return this->size;
}
//
std::ostream &operator<<(std::ostream &out, const AVXArray &c) {
   if (c.size == 0){
       out << "[]";
   } else {
       out << "[ ";
       for (int i = 0; i < c.real_size - 1; i++) {
           out << c.array[i][3] << " " << c.array[i][2] << " " << c.array[i][1] << " " << c.array[i][0] << " ";
       }
       auto n_valid = c.size % 4;
       if (n_valid == 0){
           n_valid = 4;
       }
       for (int i = 0; i < n_valid; i++) {
           out << c.array[c.real_size-1][3-i] << " ";
       }
       out << "]";
   }
   return out;
}

double AVXArray::operator[](int index){
    if (index < 0 || index >= this->size){
        throw std::out_of_range ("invalid index");
    }
    int father_index = index / 4;
    int child_index = 3 - (index % 4);
    return this->array[father_index][child_index];
}

void AVXArray::set(double v, int index) {
    if (index < 0 || index >= this->size){
        throw std::out_of_range ("invalid index");
    }
    int father_index = index / 4;
    int child_index = 3 - (index % 4);
    this->array[father_index][child_index] = v;
}
// //
double AVXArray::sum() {
    double res = 0.0;
    for (int i = 0; i < this->real_size; i++){
        __m256d s = _mm256_hadd_pd(this->array[i],this->array[i]);
        res += ((double*)&s)[0] + ((double*)&s)[2];
    }

//    auto n_valid = this->size % 4;
//    if (n_valid == 0){
//        n_valid = 4;
//    }
//    for (int i = 0; i < n_valid; i++) {
//        res += this->array[this->array.size()-1][3-i];
//    }
    return res;
}

AVXArray AVXArray::exp() {
    AVXArray arr = AVXArray();
    arr.size = this->size;
    arr.real_size = this->real_size;
    arr.array = std::make_unique<__m256d[]>(this->real_size);
    for (int i = 0; i < this->real_size-1; i++){
        arr.array[i] = _mm256_set_pd(
                std::exp(this->array[i][3]),
                std::exp(this->array[i][2]),
                std::exp(this->array[i][1]),
                std::exp(this->array[i][0]));
    }
    auto n_valid = this->size % 4;
    arr.array[this->real_size-1] = _mm256_set1_pd(0);
    if (n_valid == 0){
        n_valid = 4;
    }
    for (int i = 0; i < n_valid; i++) {
        arr.array[this->real_size-1][3-i] = std::exp(this->array[this->real_size-1][3-i]);
    }

    return arr;
}

void AVXArray::exp_void() {
    for (int i = 0; i < this->real_size-1; i++){
        this->array[i][0]=std::exp(this->array[i][0]);
        this->array[i][1]=std::exp(this->array[i][1]);
        this->array[i][2]=std::exp(this->array[i][2]);
        this->array[i][3]=std::exp(this->array[i][3]);
    }
    auto n_valid = this->size % 4;
    if (n_valid == 0){
        n_valid = 4;
    }
    for (int i = 0; i < n_valid; i++) {
        this->array[this->real_size-1][3-i] = std::exp(this->array[this->real_size-1][3-i]);
    }
}

AVXArray AVXArray::sqrt() {
    AVXArray arr = AVXArray();
    arr.size = this->size;
    arr.real_size = this->real_size;
    arr.array = std::make_unique<__m256d[]>(this->real_size);
    for (int i=0; i < this->real_size; i++){
        arr.array[i] = _mm256_sqrt_pd(this->array[i]);
    }
    return arr;
}

void AVXArray::sqrt_void() {
    for (int i=0; i < this->real_size; i++){
        this->array[i] = _mm256_sqrt_pd(this->array[i]);
    }
}



AVXArray AVXArray::operator*(const double &b) {
    __m256d scalar_array = _mm256_set1_pd(b);
    AVXArray arr = AVXArray();
    arr.size = this->size;
    arr.real_size = this->real_size;
    arr.array = std::make_unique<__m256d[]>(this->real_size);
    for (int i=0; i < this->real_size; i++){
        arr.array[i] = _mm256_mul_pd(this->array[i], scalar_array);
    }
    return arr;
}

void AVXArray::mul_void(const double &b) {
    __m256d scalar_array = _mm256_set1_pd(b);
    for (int i=0; i < this->real_size; i++){
        this->array[i] = _mm256_mul_pd(this->array[i], scalar_array);
    }
}

double AVXArray::prod() {
    double res = 1.0;
    for (int i = 0; i < this->real_size-1; i++){
        res *= this->array[i][0] * this->array[i][1] * this->array[i][2] * this->array[i][3];
    }
    auto n_valid = this->size % 4;
    if (n_valid == 0){
        n_valid = 4;
    }
    for (int i = 0; i < n_valid; i++) {
        res *= this->array[this->real_size-1][3-i];
    }
    return res;
}


double AVXArray::max() {
    double* max_values = new double [this->real_size];
    for (int i = 0; i < this->real_size-1; i++) {
        __m256d y = _mm256_permute2f128_pd(this->array[i], this->array[i], 1); // permute 128-bit values
        __m256d m1 = _mm256_max_pd(this->array[i], y); // m1[0] = max(x[0], x[2]), m1[1] = max(x[1], x[3]), etc.
        __m256d m2 = _mm256_permute_pd(m1, 5); // set m2[0] = m1[1], m2[1] = m1[0], etc.
        __m256d m = _mm256_max_pd(m1, m2); // all m[0] ... m[3] contain the horizontal max(x[0], x[1], x[2], x[3])
        max_values[i] = m[0];
    }
    auto n_valid = this->size % 4;
    double max_last_array = this->array[this->real_size-1][3];
    if (n_valid == 0){
        n_valid = 4;
    }
    for (int i = 0; i < n_valid; i++) {
        if (this->array[this->real_size-1][3-i] > max_last_array){
            max_last_array = this->array[this->real_size-1][3-i];
        }
    }
    max_values[this->real_size-1] = max_last_array;
    double max_value = max_values[0];
    for (int i = 1; i < this->real_size; i++){
        if (max_values[i] > max_value){
            max_value = max_values[i];
        }
    }
    delete [] max_values;
    return max_value;
}

// //
int AVXArray::argmax() {
    double max_value = this->max();
    for (int i = 0 ; i < this->size ; i++){
        if (max_value == (*this)[i]){
            return i;
        }
    }
    return -1;
}


class BaseGeneratorAVX{
public:
    virtual std::tuple<std::vector<AVXArray> , AVXArray > next_sample(int sample_size)=0;
    virtual int get_n_features()=0;
    virtual int get_n_classes()=0;
    virtual ~BaseGeneratorAVX(){};
protected:
    std::mt19937 random_state;
};

// #endif //NBC___BASE_GENERATOR_AVX_H

// //
// // Created by Lucca Portes on 10/09/20.
// //

// #ifndef NBC___RANDOM_RBF_AVX_H
// #define NBC___RANDOM_RBF_AVX_H

// #include <random>
// // #include "../aligned_vector.h"
// //#include "SIMDArray.h"
// //#include <tuple>
// // #include <vector>

class CentroidAVX{
public:
    CentroidAVX(AVXArray center, double class_label, double std_dev){
        this->center = std::move(center);
        this->class_label = class_label;
        this->std_dev = std_dev;

    }
    AVXArray center;
    double class_label, std_dev;
};

class RandomRBFGeneratorAVX : public BaseGeneratorAVX{
public:
    std::tuple<std::vector<AVXArray> , AVXArray > next_sample(int sample_size) override;
    int get_n_features() override;
    int get_n_classes() override;
    RandomRBFGeneratorAVX (int random_state=42, int n_features=2, int n_classes=2, int n_centroids=50);
private:
    std::mt19937 random_state;
    int n_features, n_classes, n_centroids;
    std::vector<CentroidAVX> centroids;
    std::vector<double> centroid_weights;
    void generate_centroids();
};

class SeaGeneratorAVX : public BaseGeneratorAVX{
public:
    std::tuple<std::vector<AVXArray>, AVXArray > next_sample(int sample_size) override;
    int get_n_features() override;
    int get_n_classes() override;
    SeaGeneratorAVX (int random_state, int function_number);
    // AVXArray def_class(std::vector<AVXArray> X);
private:
    double threshold;
    double def_treshold(int function_number);
};

class AccMetricAVX {

public:
    double hits = 0;
    double total = 0;
    void add_value(bool correct);
    double get_acc();
};

class ClassifierAVX {

public:
    virtual AVXArray predict(std::vector<AVXArray> &X){
        return AVXArray::zeros(1);
    };
    virtual void partial_fit(std::vector<AVXArray> &X, AVXArray &y){};
    virtual ~ClassifierAVX() = default;
};

class NaiveBayesAVX : public ClassifierAVX{
private:
    std::vector<AVXArray> att_max_value;
    std::vector<AVXArray> att_mean_value;
    std::vector<AVXArray> att_var_value;

    std::vector<AVXArray> probs_buffer;

    AVXArray get_variances(int class_index);
    AVXArray get_stds(int class_index);
    AVXArray* get_means(int class_index);

public:
    AVXArray classes_count;
    NaiveBayesAVX(int class_number, int att_number);
    ~NaiveBayesAVX() override= default;

    // NaiveBayesAVX(){}

    void partial_fit(std::vector<AVXArray> &X, AVXArray &y) override;
    AVXArray predict(std::vector<AVXArray> &X) override;
    void predict_proba(std::vector<AVXArray> &X);
    double score(std::vector<AVXArray> X, AVXArray y);
};

class PrequentialAVX {
private:
    ClassifierAVX *clf;
    BaseGeneratorAVX *gen;
    int n_wait, max_samples, batch_size, hits, total;
public:
    AccMetricAVX metric_analyzer;
    PrequentialAVX(ClassifierAVX *clf, BaseGeneratorAVX *gen, int n_wait, int max_samples, int batch_size);
    void update_metric(AVXArray &y_true, AVXArray &y_pred);
    void run();
};

RandomRBFGeneratorAVX::RandomRBFGeneratorAVX(int random_state, int n_features, int n_classes, int n_centroids){
    this->random_state = std::mt19937(random_state);
    this->n_features = n_features;
    this->n_classes = n_classes;
    this->n_centroids = n_centroids;
    this->generate_centroids();
}

void RandomRBFGeneratorAVX::generate_centroids(){
    std::uniform_real_distribution<double> dist_att(0, 1);
    std::uniform_int_distribution<int> dist_class(0,this->n_classes-1);

    for (int i = 0; i < this->n_centroids; i++){
        auto centroid_coords = AVXArray::zeros(n_features);
        for (int j = 0; j < this->n_features; j++){
            centroid_coords.set(dist_att(this->random_state), j);
        }
        auto label = dist_class(this->random_state);
        auto std_dev = dist_att(this->random_state);
        this->centroids.emplace_back(std::move(centroid_coords), label, std_dev);
        this->centroid_weights.push_back(dist_att(this->random_state));
    }
}

std::tuple<std::vector<AVXArray> , AVXArray > RandomRBFGeneratorAVX::next_sample(int sample_size){
    std::vector<AVXArray> X;
    X.reserve(sample_size);
    AVXArray y = AVXArray::zeros(sample_size);
    std::discrete_distribution<int> dist_centroids(std::begin(this->centroid_weights), std::end(this->centroid_weights));
    std::uniform_real_distribution<double> dist_att(0, 1);

    for (int i=0; i < sample_size; i++){
        auto centr_ind = dist_centroids(this->random_state);
        // auto curr_centroid = this->centroids[centr_ind];
        AVXArray att_vals = AVXArray::zeros(this->n_features);
        for (int j=0; j < this->n_features; j++){
            att_vals.set((dist_att(this->random_state) * 2.0) - 1.0, j);
        }
        double magnitude = std::sqrt((att_vals * att_vals).sum());
        double desired_mag = std::normal_distribution<>(0, this->centroids[centr_ind].std_dev)(this->random_state);
        double scale = desired_mag / magnitude;
        att_vals = this->centroids[centr_ind].center + att_vals * scale;
        X.push_back(std::move(att_vals));
        y.set(this->centroids[centr_ind].class_label, i);
    }
    return std::make_tuple(std::move(X), std::move(y));
}

int RandomRBFGeneratorAVX::get_n_features() {
    return this->n_features;
}

int RandomRBFGeneratorAVX::get_n_classes() {
    return this->n_classes;
}

SeaGeneratorAVX::SeaGeneratorAVX(int random_state, int function_number) {
    this->random_state = std::mt19937(random_state);
    this->threshold = this->def_treshold(function_number);
}

double SeaGeneratorAVX::def_treshold(int function_number) {
    if (function_number == 0){
        return 8.0;
    } else if (function_number == 1){
        return 9.0;
    } else if (function_number == 2){
        return 7.0;
    } else if (function_number == 3){
        return 9.5;
    }
    throw std::invalid_argument("Invalid function number");
}

std::tuple<std::vector<AVXArray> , AVXArray > SeaGeneratorAVX::next_sample(int sample_size) {
    std::vector<AVXArray> X;
    X.reserve(sample_size);

    std::uniform_real_distribution<double> dist(1.0, 10.0);

    for (int i=0; i < sample_size; i++){
        auto row = AVXArray::zeros(3);
        row.set(dist(this->random_state), 0);
        row.set(dist(this->random_state), 1);
        row.set(dist(this->random_state), 2);
        X.push_back(std::move(row));
    }

    // auto y = def_class(std::move(X));

     auto y = AVXArray::zeros(X.size());
        for (int i = 0 ; i < X.size(); i++) {
            auto feat_1 =  X[i][0];
            auto feat_2 =  X[i][1];
            auto sum = feat_1 + feat_2;
            if (sum > this->threshold){
                y.set(0, i);
            } else{
                y.set(1, i);
            }
        }


    return std::make_tuple(std::move(X), std::move(y));

}

int SeaGeneratorAVX::get_n_features() {
    return 3;
}

int SeaGeneratorAVX::get_n_classes() {
    return 2;
}

void AccMetricAVX::add_value(bool correct) {
    if (correct){
        this->hits++;
    }
    this->total++;
}

double AccMetricAVX::get_acc() {
    if (this->total == 0)
        return 0.0;
    return this->hits / this->total;
}

NaiveBayesAVX::NaiveBayesAVX(int class_number, int att_number) {
    this->classes_count = AVXArray::zeros(class_number);
    this->att_max_value.reserve(class_number);
    this->att_mean_value.reserve(class_number);
    this->att_var_value.reserve(class_number);
    this->probs_buffer.push_back(std::move(AVXArray::zeros(class_number)));
    for (int i = 0; i < class_number; i++){
        auto a = AVXArray::zeros(att_number);
        auto b = AVXArray::zeros(att_number);
        auto c = AVXArray::zeros(att_number);

        this->att_max_value.push_back(std::move(a));
        this->att_mean_value.push_back(std::move(b));
        this->att_var_value.push_back(std::move(c));
    }
}
int counter = 0;
void NaiveBayesAVX::partial_fit(std::vector<AVXArray> &X, AVXArray &y) {
    for (auto row_index = 0; row_index < X.size(); row_index++){
        // std::cout << X[row_index] << "\n";

        auto curr_class = y[row_index];
        // std::cout << curr_class << "\n";
        this->classes_count.set(this->classes_count[curr_class] + 1, curr_class);
        // std::cout << this->classes_count << "\n";

        // std::cout << this->att_mean_value[curr_class] << "\n";

        // AVXArray last_vars = this->get_variances(curr_class);
        // std::cout << last_vars << "\n";

        // std::cout << this->att_var_value[curr_class] << "\n";

        auto a = X[row_index] - this->att_mean_value[curr_class];

        // std::cout << a << "\n";

        auto b = a / this->classes_count[curr_class];

        // std::cout << b << "\n";

        this->att_mean_value[curr_class].add_void(b);

        // std::cout << this->att_mean_value[curr_class] << "\n";

        X[row_index].sub_void(this->att_mean_value[curr_class]);

        // std::cout << X[row_index] << "\n";

        a.mul_void(X[row_index]);

        // std::cout << X[row_index] << "\n";

        this->att_var_value[curr_class].add_void(a);

        // std::cout << this->att_var_value[curr_class]<< "\n";
        // std::cout << this->att_mean_value[curr_class]<< "\n";
        // if (++counter > 5){
        //     throw std::out_of_range("kkkk");
        // }
    }
}


AVXArray NaiveBayesAVX::predict(std::vector<AVXArray> &X) {
    auto predictions = AVXArray::zeros(X.size());
    this->predict_proba(X);
    for (auto i = 0; i < predictions.Size(); i++){
        predictions.set(this->probs_buffer[i].argmax(), i);
    }
    return predictions;
}

void NaiveBayesAVX::predict_proba(std::vector<AVXArray> &X) {
    double sum_inst = this->classes_count.sum();
    if (this->probs_buffer.size() < X.size()){
        this->probs_buffer.reserve(X.size()-1);
        for (int i=0; i < X.size()-1; i++){
            this->probs_buffer.push_back(
                std::move(
                    AVXArray::zeros(this->classes_count.Size())
                    )
                );
        }
    }
    for (auto index = 0; index < X.size(); index++){
        // this->probs_buffer[index] = this->classes_count / sum_inst;
        this->probs_buffer[index].assign(this->classes_count);
        this->probs_buffer[index].div_void(sum_inst);
        
        for (auto class_ind = 0; class_ind < this->classes_count.Size(); class_ind++){
            AVXArray std_dev = this->get_stds(class_ind);
            AVXArray* mean = this->get_means(class_ind);
            AVXArray diff = X[index] - *mean;


            auto pdf = (std_dev * 2.5066282746310002);
            pdf.be_div_by_void(1.0);
            std_dev.mul_void(std_dev);
            std_dev.mul_void(2.0);
            diff.mul_void(diff);
            diff.div_void(std_dev);
            diff.mul_void(-1.0);
            diff.exp_void();
            pdf.mul_void(diff);
            
            this->probs_buffer[index].set(this->probs_buffer[index][class_ind] * pdf.prod(), class_ind);
        }
    }
}

double NaiveBayesAVX::score(std::vector<AVXArray> X, AVXArray y) {
    auto preds = this->predict(X);
    double hits = 0;
    for (auto i=0; i < y.Size(); i++){
        if (preds[i] == y[i]){
            hits += 1;
        }
    }
    return hits/y.Size();
}

AVXArray NaiveBayesAVX::get_variances(int class_index) {
    return this->att_var_value[class_index] / this->classes_count[class_index];
}

AVXArray NaiveBayesAVX::get_stds(int class_index) {
    AVXArray vars = this->get_variances(class_index);
    vars.sqrt_void();
    return vars;
}

AVXArray* NaiveBayesAVX::get_means(int class_index) {
    return &this->att_mean_value[class_index];
}

PrequentialAVX::PrequentialAVX(ClassifierAVX *clf, BaseGeneratorAVX *gen, int n_wait, int max_samples, int batch_size) {
    this->clf = clf;
    this->gen = gen;
    this->n_wait = n_wait;
    this->max_samples = max_samples;
    this->batch_size = batch_size;
    this->metric_analyzer = AccMetricAVX();
    this->hits = 0;
    this->total = 0;
}

void PrequentialAVX::update_metric(AVXArray &y_true, AVXArray &y_pred) {
    for (auto i = 0; i < y_pred.Size(); i++){
        this->metric_analyzer.add_value(y_pred[i] == y_true[i]);
    }
}

void PrequentialAVX::run() {
    int count = 0;
    int count_n_wait = 0;
    std::vector<double> vec_accs;
    while (count < this->max_samples) {
        std::vector<AVXArray> X;
        AVXArray y;
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


void run_prequentialAVX(int n_wait, int max_samples, int batch_size){
        auto clf_obj = NaiveBayesAVX(2, 3);
        auto gen_obj = SeaGeneratorAVX(42, 0);
        auto preq = PrequentialAVX(&clf_obj, &gen_obj, n_wait, max_samples, batch_size);
        preq.run();
}

void run_prequentialRBFAVX(int n_wait, int max_samples, int batch_size, int n_features, int n_classes){
    auto clf_obj = NaiveBayesAVX(n_classes, n_features);
    auto gen_obj = RandomRBFGeneratorAVX(42, n_features, n_classes, 50);
    auto preq = PrequentialAVX(&clf_obj, &gen_obj, n_wait, max_samples, batch_size);
    preq.run();
}


