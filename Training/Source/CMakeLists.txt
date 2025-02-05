cmake_minimum_required(VERSION 3.18)


project(Training LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(WIN32)

    set(PNG_ROOT_DIR "D:/Projects/libpng.ins")
    set(PNG_PNG_INCLUDE_DIR "${PNG_ROOT_DIR}/include")
    set(PNG_LIBRARY "${PNG_ROOT_DIR}/lib/libpng16.lib")
    set(PNG_LIBRARY_PATH ${PNG_LIBRARY})
    set(PNG_DLL_DIR "${PNG_ROOT_DIR}/bin")


    set(CUDNN_ROOT_DIR "C:/Program Files/NVIDIA/CUDNN/v9.7")
    set(CUDNN_INCLUDE_PATH ${CUDNN_ROOT_DIR}/include/12.8)
    set(CUDNN_LIBRARY "${CUDNN_ROOT_DIR}/lib/12.8/x64/cudnn.lib")
    set(CUDNN_LIBRARY_PATH ${CUDNN_LIBRARY})
    set(CUDNN_DLL_DIR "${CUDNN_ROOT_DIR}/bin/12.8")


    set(LIBTORCHPATH D:/Projects/libtorch.cu.12.6)
	set(CMAKE_PREFIX_PATH D:/Projects/libtorch.cu.12.6)

 
	set(Torch_DIR ${LIBTORCHPATH}/share/cmake/Torch/)
	set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6")
     set(nvtx3_dir ${CUDA_TOOLKIT_ROOT_DIR}/include/nvtx3)
    set(USE_NVTX ON CACHE BOOL "Use NVTX" FORCE)
    set(CUDA_nvToolsExt_LIBRARY "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64/nvToolsExt64_1.dll")
    set(CMAKE_CUDA_HOST_IMPLICIT_LINK_LIBRARIES "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64/cuda.lib")
    
	
   
endif(WIN32)
  
set(USE_CUDNN ON CACHE BOOL "Use cuDNN" FORCE)
set(USE_CUSPARSELT ON CACHE BOOL "Use cuSPARSELt")
set(USE_CUDSS ON CACHE BOOL "Use cuDSS")
set(USE_CUFILE ON CACHE BOOL "Use cuFile")
set(CAFFE2_USE_CUDNN ON CACHE BOOL "CAFFE2_USE_CUDNN")


set(CMAKE_CUDA_ARCHITECTURES 86)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler /permissive-")
find_package(Torch REQUIRED)
find_package(CUDA REQUIRED)
find_package(PNG REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
if(USE_CUDNN)
    find_library(CUDNN_LIBRARY
        NAMES cudnn
        PATHS ${CUDNN_LIBRARY_DIR}
        NO_DEFAULT_PATH
        REQUIRED)
endif()

set(SOURCE 
    IDataSet.h 
    CIFAR100.h 
    TrainModel.h 
    TrainModel.cpp 
    CIFAR100DataSet.h 
    ProgressBar.h 
    FileSaver.h 
    FileSaver.cpp 
    CIFAR100Module.h
    CIFAR100ClassNames.h 
    HeirarchicalLoss.h 
    HeirarchicalLoss.cpp
    ResidualBlock.h 
    CIFAR100CoarseModule.h 
    CIFAR100FineModule.h 
    TrainSplitModels.h 
    TrainSplitModels.cpp
    TrainModelsMultiGPU.h 
    TrainModelsMultiGPU.cpp 
    ReduceLROnPlateauScheduler.h 
    ReduceLROnPlateauScheduler.cpp
    main.cpp
)
add_executable(${PROJECT_NAME} 
  	${SOURCE}
)


if(USE_CUDNN)
    target_include_directories(${PROJECT_NAME} PRIVATE ${CUDNN_INCLUDE_DIR} ${PNG_PNG_INCLUDE_DIR})
    target_link_libraries(${PROJECT_NAME} PRIVATE "${TORCH_LIBRARIES}" ${CUDNN_LIBRARY} PNG::PNG)
else()
    target_include_directories(${PROJECT_NAME} PRIVATE ${PNG_PNG_INCLUDE_DIR})

    target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}" PNG::PNG)

endif(USE_CUDNN)





if(MSVC)
    file(GLOB TORCH_DLLS "${LIBTORCHPATH}/lib/*.dll")
    add_custom_command(TARGET ${PROJECT_NAME}
                      POST_BUILD
                      COMMAND ${CMAKE_COMMAND} -E copy_if_different
                      ${TORCH_DLLS}
                      $<TARGET_FILE_DIR:${PROJECT_NAME}>)

    if(EXISTS "${CUDA_TOOLKIT_ROOT_DIR}/bin/nvToolsExt64_1.dll")
        add_custom_command(TARGET ${PROJECT_NAME}
                          POST_BUILD
                          COMMAND ${CMAKE_COMMAND} -E copy_if_different
                          "${CUDA_TOOLKIT_ROOT_DIR}/bin/nvToolsExt64_1.dll"
                          $<TARGET_FILE_DIR:${PROJECT_NAME}>)
    endif()
    if(USE_CUDNN)
        file(GLOB CUDNN_DLLS "${CUDNN_DLL_DIR}/*.dll")
        add_custom_command(TARGET ${PROJECT_NAME}
                          POST_BUILD
                          COMMAND ${CMAKE_COMMAND} -E copy_if_different
                          ${CUDNN_DLLS}
                          $<TARGET_FILE_DIR:${PROJECT_NAME}>)
    endif()

     if(EXISTS "${PNG_DLL_DIR}/libpng16.dll")
        add_custom_command(TARGET ${PROJECT_NAME}
                          POST_BUILD
                          COMMAND ${CMAKE_COMMAND} -E copy_if_different
                          "${PNG_DLL_DIR}/libpng16.dll"
                          $<TARGET_FILE_DIR:${PROJECT_NAME}>)
    endif()

    add_custom_command(TARGET ${PROJECT_NAME}
                      POST_BUILD
                      COMMAND ${CMAKE_COMMAND} -E copy_if_different
                      "${CMAKE_SOURCE_DIR}/cifar-100-binary.zip"
                      "$<TARGET_FILE_DIR:${PROJECT_NAME}>/cifar-100-binary.zip"
                      COMMAND ${CMAKE_COMMAND} -E tar xf
                      "$<TARGET_FILE_DIR:${PROJECT_NAME}>/cifar-100-binary.zip"
                      WORKING_DIRECTORY $<TARGET_FILE_DIR:${PROJECT_NAME}>
                      COMMENT "Extracting CIFAR-100 dataset to binary directory")
endif(MSVC)