add_mlir_library(MatMulOptimization
  BatchMatMulOptimize.cpp
	MatMulOptimize.cpp
  MatMulVectorization.cpp
  MatMulParallelVectorization.cpp
  MatMulTransposeBVec.cpp
  LINK_LIBS PUBLIC
  BuddyUtils
)

add_mlir_library(BatchMatMulOptimization
  BatchMatMulOptimize.cpp
)

add_mlir_library(MatMulParallelVectorization
  MatMulParallelVectorization.cpp
)

add_mlir_library(MatMulTransposeBVec
  MatMulTransposeBVec.cpp 
)
