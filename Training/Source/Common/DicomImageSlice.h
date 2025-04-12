#ifndef DICOM_IMAGE_SLICE_
#define DICOM_IMAGE_SLICE_
#include <string>
#include <vector>
#include <dcmtk/dcmdata/dctk.h>
#include <dcmtk/dcmimgle/dcmimage.h>
#include <dcmtk/dcmdata/dcdatset.h>

#include "DicomMetaData.h"

namespace  torch_explorer
{
    class DicomImageSlice
    {
    public:

        DicomImageSlice();
        ~DicomImageSlice();
        DicomImageSlice(const DicomImageSlice&) = delete;
        DicomImageSlice(const DicomImageSlice&&) = delete;

        DicomImageSlice& operator=(const DicomImageSlice&) = delete;
        DicomImageSlice& operator=(const DicomImageSlice&&) = delete;
        void loadImage(DicomMetaData& metaData, const std::string& filePath, DcmFileFormat& ff);

    private:

        static char* allocatePixels(const DiPixel* interData, size_t& outSize);


        std::vector<double> imagePositionPatient_;
        std::vector<double> imageOrientationPatient_;
        std::vector<double> normal_;
        std::string orientationRows_;
        std::string orientationCols_;
        std::string orientationNormal_;
        double positionOnNormal_;
        double rescaleSlope_;
        double rescaleIntercept_;
        double spacingBetweenSlices_;
        double sliceThickness_;
        std::vector<double> pixelSpacing_;
        int isSigned_;
        int width_;
        int height_;
        uint16_t bitsPerPixel_;
        uint16_t componentsPerPixel_;
        uint16_t bitsUsedWhenWriting_;
        uint16_t highBitWhenWriting_;
        void* pixelData_;
        size_t pixelDataSizeInBytes_;
    };

   
}



#endif