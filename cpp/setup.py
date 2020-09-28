# from distutils.core import setup
# from distutils.extension import Extension
# from Cython.Distutils import build_ext
#
#
# setup(
#   name='prequentialC',
#   ext_modules=[
#     Extension('prequentialC',
#               sources=['prequential_c.pyx'],
#               extra_compile_args=['-mavx', '-std=c++14', '-o3'],
#               language='c++')
#     ],
#   cmdclass={'build_ext': build_ext}
# )

from distutils.core import setup
from Cython.Build import cythonize
import os

os.environ['CFLAGS'] = '-Wall -std=c++17 -mavx -faligned-new'
setup(ext_modules = cythonize(
       "prequential_c.pyx",            # our Cython source
       # sources=["Rectangle.cpp"],  # additional source file(s)
       language="c++",

    # generate C++ code
      ))
