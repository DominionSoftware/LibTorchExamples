cmake_minimum_required(VERSION 3.18)


project(PANCREAS-CT LANGUAGES CXX CUDA)

if(MSVC)
    add_compile_options(/std:c++20 /Zc:__cplusplus)
endif()


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
    set(TorchVision_DIR "D:/Projects/vision.ins/share/cmake/TorchVision")
 
	set(Torch_DIR ${LIBTORCHPATH}/share/cmake/Torch/)
	set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6")
     set(nvtx3_dir ${CUDA_TOOLKIT_ROOT_DIR}/include/nvtx3)
    set(USE_NVTX ON CACHE BOOL "Use NVTX" FORCE)
    set(CUDA_nvToolsExt_LIBRARY "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64/nvToolsExt64_1.dll")
    set(CMAKE_CUDA_HOST_IMPLICIT_LINK_LIBRARIES "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64/cuda.lib")
    set(DRIVE_LETTER D:/)
    set(OpenCV_DIR "${DRIVE_LETTER}Projects/opencv.ins")

    message(WARNING,${OpenCV_DIR})
    set(nlohmann_json_DIR "${DRIVE_LETTER}Projects/json.ins/share/cmake/nlohmann_json")
    set(DCMTK_DIR D:/Projects/dcmtk.ins/cmake)
    set(DCMTK_DLL_DIR D:/Projects/dcmtk.ins/bin)
    set(ITK_DIR "D:/Projects/itk540.ins/lib/cmake/ITK-5.4")

    set(VTK_DIR "D:/Projects/vtk941.ins/lib/cmake/vtk-9.4")
 
    find_package(VTK 9.4 COMPONENTS 
                  CommonColor
                  CommonCore
                  IOImage
                  FiltersSources
                  InteractionStyle
                  InteractionWidgets
                  RenderingContextOpenGL2
                  RenderingCore
                  RenderingFreeType
                  RenderingGL2PSOpenGL2
                  RenderingOpenGL2
                ) 


	
   
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
 
find_package(nlohmann_json REQUIRED)

set(ITK_COMPONENTS
  ITKCommon
  ITKImageIO
  ITKIOImageBase
  ITKIOBMP       # For BMP format
  ITKIOJPEG      # For JPEG format
  ITKIOPNG       # For PNG format
  ITKIOTIFF      # For TIFF format
  ITKIONIFTI     # For NIFTI format
  ITKIONRRD      # For NRRD format
  ITKIOVTK       # For VTK integration
  ITKIOTransformBase
  ITKImageGrid
)


find_package(ITK REQUIRED COMPONENTS ${ITK_COMPONENTS})

find_package(VTK REQUIRED)


if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV not found. Please specify OpenCV_DIR")
endif()
 

find_package(DCMTK REQUIRED)

 
if(DCMTK_FOUND)
  message(STATUS "Found DCMTK:")
  message(STATUS "  Includes: ${DCMTK_INCLUDE_DIRS}")
  message(STATUS "  Libraries: ${DCMTK_LIBRARIES}")
else()
  message(FATAL_ERROR "DCMTK not found!")
endif()

set(Eigen3_DIR "D:/Projects/eigen.ins/share/eigen3/cmake")
find_package(Eigen3 REQUIRED)



 
 
include(${ITK_USE_FILE})



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
    ../Common/ReduceLROnPlateauScheduler.h 
    ../Common/ReduceLROnPlateauScheduler.cpp
    ../Common/Transforms.h 
    ../Common/DicomLoader.h 
    ../Common/DicomLoader.cpp 
    ../Common/JSONLoader.h 
    ../Common/JSONLoader.cpp 
    ../Common/MonaiDataLoader.h 
    ../Common/MonaiDataLoader.cpp 
    ../Common/DicomMetaData.h 
    ../Common/DicomImageSlice.h 
    ../Common/DicomImageSlice.cpp 
)

set(SOURCE 
 
    PancreasCTDataSet.h 
    PancreasCTDataSet.cpp

    PancreasCTModule.h
    TrainPancreasCTModule.h 
    TrainPancreasCTModule.cpp 
    main.cpp
)


add_executable(${PROJECT_NAME} 
  	${SOURCE}
    ${COMMON}
)


set(DCMTK_LIBS
    DCMTK::ofstd
    DCMTK::oflog
    DCMTK::oficonv
    DCMTK::dcmdata
    DCMTK::i2d
    DCMTK::dcmxml
    DCMTK::dcmimgle
    DCMTK::dcmimage
    DCMTK::dcmjpeg
    DCMTK::ijg8
    DCMTK::ijg12
    DCMTK::ijg16
    DCMTK::dcmjpls
    DCMTK::dcmtkcharls
    DCMTK::dcmtls
    DCMTK::dcmnet
    DCMTK::dcmsr
    DCMTK::cmr
    DCMTK::dcmdsig
    DCMTK::dcmwlm
    DCMTK::dcmqrdb
    DCMTK::dcmpstat
    DCMTK::dcmrt
    DCMTK::dcmiod
    DCMTK::dcmfg
    DCMTK::dcmseg
    DCMTK::dcmtract
    DCMTK::dcmpmap
    DCMTK::dcmect

)

if(USE_CUDNN)
    target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${OpenCV_INCLUDE_DIRS} ${COMMON} ${TorchVision_INCLUDE_DIR} ${CUDNN_INCLUDE_DIR} ${PNG_PNG_INCLUDE_DIR} ${DCMTK_INCLUDE_DIRS} ${nlohmann_json_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIRS})
    target_link_libraries(${PROJECT_NAME} PRIVATE "${TORCH_LIBRARIES}" ${CUDNN_LIBRARY} ${TorchVision_LIBRARIES} ${OpenCV_LIBRARIES} PNG::PNG ${DCMTK_LIBS} nlohmann_json::nlohmann_json ${VTK_LIBRARIES} ${ITK_LIBRARIES})
else()
    target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}  ${OpenCV_INCLUDE_DIRS} ${COMMON} {TorchVision_INCLUDE_DIR} ${PNG_PNG_INCLUDE_DIR} ${DCMTK_INCLUDE_DIRS} ${nlohmann_json_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIRS})
    target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}"  ${TorchVision_LIBRARIES} ${OpenCV_LIBRARIES} PNG::PNG ${DCMTK_LIBS} nlohmann_json::nlohmann_json ${VTK_LIBRARIES} ${ITK_LIBRARIES})

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

    file(GLOB DCMTK_DLLS "${DCMTK_DLL_DIR}/*.dll")
        add_custom_command(TARGET ${PROJECT_NAME}
                          POST_BUILD
                          COMMAND ${CMAKE_COMMAND} -E copy_if_different
                          ${DCMTK_DLLS}
                          $<TARGET_FILE_DIR:${PROJECT_NAME}>)


     if(EXISTS "${PNG_DLL_DIR}/libpng16.dll")
        add_custom_command(TARGET ${PROJECT_NAME}
                          POST_BUILD
                          COMMAND ${CMAKE_COMMAND} -E copy_if_different
                          "${PNG_DLL_DIR}/libpng16.dll"
                          $<TARGET_FILE_DIR:${PROJECT_NAME}>)
    endif()


set(DEST_DIR "$<TARGET_FILE_DIR:${PROJECT_NAME}>/pathology_nuclei_segmentation_classification")

if(NOT EXISTS ${DEST_DIR})

    install(DIRECTORY "${CMAKE_SOURCE_DIR}/../PreTrainedModels/pathology_nuclei_segmentation_classification"
        DESTINATION "$<TARGET_FILE_DIR:${PROJECT_NAME}>"
        OPTIONAL) 

 else()
   message(STATUS "pathology_nuclei_segmentation_classification models installed")
endif()


endif(MSVC)