from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize

import numpy as np

sourcefiles = ['entropy_tools.pyx']#,'entropy_c_lib.c']
compile_options = ['-O3']

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("entropy_tools", sourcefiles, extra_compile_args = compile_options,\
    include_dirs=[np.get_include()])],
)
# SIMD compile options:
# http://koturn.hatenablog.com/entry/2016/07/18/090000
