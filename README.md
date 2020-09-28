# NaiveBayesImplementations
This repository contains the code of the methods used in the paper 'Analysis on Python Performance for Data Stream Mining'.

## Compiling C++ with an without AVX
Please make sure your CPU supports AVX.

First you need to install the Python library Cython:  
```
pip install cython
```

On the ```cpp``` folder, simply run:  
```
python setup.py build_ext --inplace
```

Then simply import the ```.so``` file generated in the same folder.

Run as:

```
prequential_c.prequential("NB", "SEA", n_wait, n_instances, batch_size)
prequential_c.prequentialAVX(n_wait, n_instances, batch_size)
prequential_c.prequentialRBF("NB", "RBF", n_wait, n_instances, batch_size, n_features, n_classes)
prequential_c.prequentialRBFAVX(n_wait, n_instances, batch_size, n_features, n_classes)

```

## Compiling Rust
Please make sure you have Rust installed.

First change your rust envinroment to nightly
```
rustup default nightly
```

Then simply run
```
cargo build --release
```

It will generate a ```.so``` file in ```/rust/target/release```  
Rename it  
```mv libmyrustlib.so prequential_rust.so```

Then simply import it and use it as:
```
import prequential_rust
prequential_rust.run_prequential(n_wait, n_instances, batch_size)
prequential_rust.run_prequential_RBF(n_wait, n_instances, batch_size, n_features, n_classes)
```