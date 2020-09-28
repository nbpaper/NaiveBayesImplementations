from libcpp.string cimport string

cdef extern from "normal_file.h":
    void run_prequential(string, string, int, int, int)
    void run_prequentialRBF(string, string, int, int, int, int, int)

#cdef extern from "avx_file.h":
#    void run_prequentialAVX(string, string, int, int, int)
#    void run_prequentialRBFAVX(string, string, int, int, int, int, int)

cdef extern from "avx_file_unique_ptr.h":
    void run_prequentialAVX(int, int, int)
    void run_prequentialRBFAVX(int, int, int, int, int)



def prequential(str clf, str gen, int n_wait, int max_samples, int batch_size):
    clf_ = clf.encode('utf-8')
    gen_ = gen.encode('utf-8')
    run_prequential(clf_, gen_, n_wait, max_samples, batch_size)

def prequentialRBF(str clf, str gen, int n_wait, int max_samples, int batch_size, int n_features, int n_classes):
    clf_ = clf.encode('utf-8')
    gen_ = gen.encode('utf-8')
    run_prequentialRBF(clf_, gen_, n_wait, max_samples, batch_size, n_features, n_classes)

def prequentialRBFAVX(int n_wait, int max_samples, int batch_size, int n_features, int n_classes):
    run_prequentialRBFAVX(n_wait, max_samples, batch_size, n_features, n_classes)

def prequentialAVX(int n_wait, int max_samples, int batch_size):
    run_prequentialAVX(n_wait, max_samples, batch_size)
