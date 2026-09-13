#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
os.path.dirname(os.path.abspath(__file__))

setup(
    name="tile_local_aggregate_prob_sq",
    packages=['tile_local_aggregate_prob_sq'],
    ext_modules=[
        CUDAExtension(
            name="tile_local_aggregate_prob_sq._C",
            sources=[
            "src/aggregator_impl.cu",
            "src/forward.cu",
            "src/backward.cu",
            "local_aggregate.cu",
            "ext.cpp"],


            extra_compile_args={"nvcc": ["-Xcompiler", "-fno-gnu-unique"]})
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
