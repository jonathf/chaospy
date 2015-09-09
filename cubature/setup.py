from distutils.core import setup
from distutils.extension import Extension
try:
    from Cython.Distutils import build_ext
    ext_modules = [Extension('_cubature', ['_cubature.pyx'])]
    cmdclass = {'build_ext': build_ext}
except ImportError:
    ext_modules = [Extension('_cubature', ['_cubature.c'])]
    cmdclass = {}
    pass
setup(
cmdclass = cmdclass,
ext_modules = ext_modules,
)
