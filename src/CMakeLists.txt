set(SOURCE_FILES
    "cudaMat4.hpp"
    "glslUtility.hpp"
    "glslUtility.cpp"
    "utilityCore.hpp"
    "utilityCore.cpp"
    "kernel.h"
    "kernel.cu"
    "pointcloud.h"
    "pointcloud.cpp"
    "svd3.h"
    "kdtree.cpp"
    "kdtree.hpp"
    )

cuda_add_library(src
    ${SOURCE_FILES}
    OPTIONS -arch=sm_50
    )
