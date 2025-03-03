#ifndef COVID19_DATASET_
#define COVID19_DATASET_

#include <random>
#include <filesystem>
#include <vector>
#include <iostream>
#include <algorithm>
#include <numeric>
#include "opencv2/opencv.hpp"
#include "../Common/IDataSet.h"
#include "../Common/FileSaver.h"
#include <torch/torch.h>
#include "../Common/CutMixTransform.h"

namespace torch_explorer
{
    // Custom dataset class for Covid19 Radiography dataset
    class Covid19 : public torch::data::Dataset<Covid19>
    {
    public:
        enum class Mode { kTrain, kTest };
        
        Covid19() : mode_(Mode::kTrain), image_size_(224) {}
        
        Covid19(const std::filesystem::path& root_path, Mode mode, float train_ratio = 0.8, 
                int image_size = 224)
            : mode_(mode), image_size_(image_size)
        {
            loadDataset(root_path, train_ratio);
        }
        
        void loadDataset(const std::filesystem::path& root_path, float train_ratio = 0.8)
        {
            // Define the class folders to look for
            std::vector<std::string> classes = {"Normal", "COVID", "Lung_Opacity", "Viral Pneumonia"};
            
            // Map class names to indices for future reference
            for (size_t i = 0; i < classes.size(); ++i) {
                class_to_idx_[classes[i]] = i;
            }
            
            // Clear existing data if any
            image_paths_.clear();
            labels_.clear();
            
            // Load all image paths and their corresponding labels
            std::vector<std::string> all_image_paths;
            std::vector<int> all_labels;
            
            int class_idx = 0;
            for (const auto& class_name : classes) {
                std::filesystem::path class_dir = root_path / class_name / "images";
                
                if (!std::filesystem::exists(class_dir)) {
                    std::cerr << "Warning: Directory not found: " << class_dir << std::endl;
                    continue;
                }
                
                for (const auto& entry : std::filesystem::directory_iterator(class_dir)) {
                    if (entry.path().extension() == ".png" || 
                        entry.path().extension() == ".jpg" || 
                        entry.path().extension() == ".jpeg") {
                        all_image_paths.push_back(entry.path().string());
                        all_labels.push_back(class_idx);
                    }
                }
                
                class_idx++;
            }
            
            // Shuffle the data with a fixed seed for reproducibility
            auto seed = 42;
            auto rng = std::default_random_engine{seed};
            std::vector<size_t> indices(all_image_paths.size());
            std::iota(indices.begin(), indices.end(), 0);
            std::shuffle(indices.begin(), indices.end(), rng);
            
            // Split into train and test sets
            size_t num_train = static_cast<size_t>(all_image_paths.size() * train_ratio);
            
            for (size_t i = 0; i < indices.size(); i++) {
                size_t idx = indices[i];
                if ((mode_ == Mode::kTrain && i < num_train) || 
                    (mode_ == Mode::kTest && i >= num_train)) {
                    image_paths_.push_back(all_image_paths[idx]);
                    labels_.push_back(all_labels[idx]);
                }
            }
            
            std::cout << "Loaded " << image_paths_.size() << " images for " 
                     << (mode_ == Mode::kTrain ? "training" : "testing") << std::endl;
        }
        
        torch::data::Example<> get(size_t index) override {
            std::string image_path = image_paths_[index];
            int label = labels_[index];
            
            // Load and preprocess the image using OpenCV
            cv::Mat image = cv::imread(image_path);
            if (image.empty()) {
                std::cerr << "Could not read image: " << image_path << std::endl;
                // Return an empty tensor if image can't be read
                return {torch::zeros({3, image_size_, image_size_}), 
                        torch::tensor(label, torch::kLong)};
            }
            
            // Resize the image to the desired size
            cv::resize(image, image, cv::Size(image_size_, image_size_));
            
            // Convert from BGR to RGB
            cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
            
            // Convert to tensor
            torch::Tensor tensor_image = torch::from_blob(image.data, 
                                                         {image_size_, image_size_, 3}, 
                                                         torch::kByte).clone();
            
            // Permute dimensions from HWC to CHW format
            tensor_image = tensor_image.permute({2, 0, 1});
            
            // Convert to float and normalize to [0, 1]
            tensor_image = tensor_image.to(torch::kFloat32).div(255);
            
            // Create and return the example
            return {tensor_image, torch::tensor(label, torch::kLong)};
        }
        
        torch::optional<size_t> size() const override {
            return image_paths_.size();
        }

    private:
        Mode mode_;
        int image_size_;
        std::vector<std::string> image_paths_;
        std::vector<int> labels_;
        std::map<std::string, int> class_to_idx_;
    };

    class Covid19DataSet : public IDataSet
    {
    public:
        Covid19DataSet() : is_train(true), options(32) {
            options.workers(2);
        }

        explicit Covid19DataSet(bool is_training) : is_train(is_training), options(32) {
            options.workers(2);
        }

        void load(const std::filesystem::path& root_path, std::shared_ptr<FileSaver> fileSaver = nullptr) override {
            auto mode = is_train ? Covid19::Mode::kTrain : Covid19::Mode::kTest;
            
            // Load the dataset
            raw_dataset = Covid19(root_path, mode);
            
            // Define normalization transform
            auto normalize_transform = torch::data::transforms::Normalize<>({0.485, 0.456, 0.406}, 
                                                                           {0.229, 0.224, 0.225});
            
            // Apply normalization to dataset
            dataset = raw_dataset.map(normalize_transform);
            
            std::cout << "Covid19DataSet loaded with " << dataset.size().value() 
                     << " samples for " << (is_train ? "training" : "testing") << std::endl;
        }

        torch::data::Example<> get(size_t index) override {
            return dataset.get(index);
        }

        torch::optional<size_t> size() const override {
            return dataset.size();
        }

        size_t getBatchSize() const override {
            return options.batch_size();
        }

        void setBatchSize(size_t batch_size) override {
            options.batch_size(batch_size);
        }

        size_t getNumWorkers() const override {
            return options.workers();
        }

        void setNumWorkers(size_t num_workers) override {
            options.workers(num_workers);
        }

        bool isTraining() const override {
            return is_train;
        }

        std::vector<int64_t> getInputShape() const override {
            return { 3, 224, 224 };  // Covid19 images are resized to 224x224 RGB
        }

        size_t getNumClasses() const override {
            return 4;  // Normal, COVID, Lung_Opacity, Viral Pneumonia
        }

        auto getDataLoader() -> std::unique_ptr<torch::data::StatelessDataLoader<torch::data::datasets::MapDataset<
                                                        Covid19,
                                                        torch::data::transforms::Normalize<>>,
                                                        torch::data::samplers::RandomSampler>> override {
            // For compatibility with your CIFAR100DataSet
            using RandomSampler = torch::data::samplers::RandomSampler;
            
            if (is_train) {
                // For training, use a random sampler
                return std::make_unique<torch::data::StatelessDataLoader<decltype(dataset), RandomSampler>>(
                    dataset, RandomSampler(dataset.size().value()), options);
            } else {
                // For testing, still using RandomSampler as required by interface, but with sequential behavior
                return std::make_unique<torch::data::StatelessDataLoader<decltype(dataset), RandomSampler>>(
                    dataset, RandomSampler(dataset.size().value()), options);
            }
        }

        void enableCutMix(float alpha = 1.0, float prob = 0.5) {
            use_cutmix_ = true;
            cutmix_ = CutMixTransform(alpha, prob);
        }

        void disableCutMix() {
            use_cutmix_ = false;
        }

        // Get CutMix transformer for use in training loop
        CutMixTransform& getCutMixTransform() {
            return cutmix_;
        }

        bool isCutMixEnabled() const {
            return use_cutmix_;
        }

    private:
        Covid19 raw_dataset;
        torch::data::datasets::MapDataset<Covid19, torch::data::transforms::Normalize<>> dataset{
            Covid19(),
            torch::data::transforms::Normalize<>({0.485, 0.456, 0.406}, {0.229, 0.224, 0.225})
        };
        
        torch::data::DataLoaderOptions options;
        bool is_train;
        bool use_cutmix_ = false;
        CutMixTransform cutmix_{ 1.0, 0.5 };
        std::mt19937 gen_{ std::random_device{}() };
    };
}

#endif // COVID19_DATASET_
