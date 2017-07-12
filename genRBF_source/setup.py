# -*- coding: utf-8 -*-

from distutils.command.build_ext import build_ext
from distutils.core import setup, Extension
from distutils.sysconfig import customize_compiler

from Cython.Build import cythonize

__author__ = "Łukasz Struski"


class my_build_ext(build_ext):
    def build_extensions(self):
        customize_compiler(self.compiler)
        try:
            self.compiler.compiler_so.remove("-Wstrict-prototypes")
        except (AttributeError, ValueError):
            pass
        build_ext.build_extensions(self)


setup(
    cmdclass={'build_ext': my_build_ext},
    ext_modules=cythonize(Extension(
        "cRBFkernel",
        sources=["cRBFkernel.pyx"],
        language="c++",
    )))
