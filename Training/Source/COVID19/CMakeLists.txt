cmake_minimum_required(VERSION 3.18)


project(COVID19 LANGUAGES CXX CUDA)

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
    set(DRIVE_LETTER D:/)
    set(OpenCV_DIR "${DRIVE_LETTER}Projects/opencv.ins")

    message(WARNING,${OpenCV_DIR})
 


	
   
endif(WIN32)
  
set(USE_CUDNN ON CACHE BOOL "Use cuDNN" FORCE)
set(USE_CUSPARSELT ON CACHE BOOL "Use cuSPARSELt")
set(USE_CUDSS ON CACHE BOOL "Use cuDSS")
set(USE_CUFILE ON CACHE BOOL "Use cuFile")
set(CAFFE2_USE_CUDNN ON CACHE BOOL "CAFFE2_USE_CUDNN")


set(CMAKE_CUDA_ARCHITECTURES 86)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler /permissive-")
find_package(Torch REQUIRED)
find_package(TorchVision REQUIRED)

find_package(CUDA REQUIRED)
find_package(PNG REQUIRED)
find_package(OpenCV REQUIRED)
if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV not found. Please specify OpenCV_DIR")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
if(USE_CUDNN)
    find_library(CUDNN_LIBRARY
        NAMES cudnn
        PATHS ${CUDNN_LIBRARY_DIR}
        NO_DEFAULT_PATH
        REQUIRED)
endif()
set(COMMON 
    ../Common/IDataSet.h 
    ../Common/ProgressBar.h 
    ../Common/FileSaver.h 
    ../Common/FileSaver.cpp 
    ../Common/ResidualBlock.h 
    ../Common/ReduceLROnPlateauScheduler.h 
    ../Common/ReduceLROnPlateauScheduler.cpp
    ../Common/CosineAnnealingScheduler.h 
    ../Common/CosineAnnealingScheduler.cpp 
    ../Common/CutMixTransform.h 
    ../Common/BetaDistribution.h 
    ../Common/OneHot.h 
    ../Common/OneHot.cpp 
    ../Common/KullbackLieblerDivergenceLoss.h
    ../Common/KullbackLieblerDivergenceLoss.cpp 
    ../Common/HybridOptimizer.h
)

set(SOURCE 
 
    COVID19DataSet.h 
    COVID19Module.h
    TrainCovid19Module.h 
    TrainCovid19Module.cpp 
    main.cpp
)


add_executable(${PROJECT_NAME} 
  	${SOURCE}
    ${COMMON}
)


if(USE_CUDNN)
    target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${OpenCV_INCLUDE_DIRS} ${COMMON} ${TorchVision_INCLUDE_DIR} ${CUDNN_INCLUDE_DIR} ${PNG_PNG_INCLUDE_DIR})
    target_link_libraries(${PROJECT_NAME} PRIVATE "${TORCH_LIBRARIES}" ${CUDNN_LIBRARY} ${TorchVision_LIBRARIES} ${OpenCV_LIBRARIES} PNG::PNG)
else()
    target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}  ${OpenCV_INCLUDE_DIRS} ${COMMON} {TorchVision_INCLUDE_DIR} ${PNG_PNG_INCLUDE_DIR})
    target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}"  ${TorchVision_LIBRARIES} ${OpenCV_LIBRARIES} PNG::PNG)

endif(USE_CUDNN)




if(MSVC)
    set(OpenCV_ROOT_DIR "D:/Projects/opencv.ins")
    file(GLOB OPENCV_DLLS "${OpenCV_ROOT_DIR}/x64/vc17/bin/*.dll")

    add_custom_command(TARGET ${PROJECT_NAME}
                  POST_BUILD
                  COMMAND ${CMAKE_COMMAND} -E copy_if_different
                  ${OPENCV_DLLS}
                  $<TARGET_FILE_DIR:${PROJECT_NAME}>)


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
                       "${CMAKE_SOURCE_DIR}/../PreTrainedModels/resnet18_model.pt"
                       "$<TARGET_FILE_DIR:${PROJECT_NAME}>"
                       COMMENT "Copying resnet18_model.pt to binary directory")

     add_custom_command(TARGET ${PROJECT_NAME}
                      POST_BUILD
                      COMMAND ${CMAKE_COMMAND} -E copy_if_different
                      "${CMAKE_SOURCE_DIR}/COVID-19_Radiography_Dataset.zip"
                      "$<TARGET_FILE_DIR:${PROJECT_NAME}>/COVID-19_Radiography_Dataset.zip"
                      COMMAND ${CMAKE_COMMAND} -E tar xf
                      "$<TARGET_FILE_DIR:${PROJECT_NAME}>/COVID-19_Radiography_Dataset.zip"
                      WORKING_DIRECTORY $<TARGET_FILE_DIR:${PROJECT_NAME}>
                      COMMENT "Extracting COVID-19_Radiography_Dataset dataset to binary directory")

endif(MSVC)