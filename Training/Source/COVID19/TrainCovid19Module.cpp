#include "TrainCovid19Module.h"
#include "Covid19Module.h"
#include "../Common/IDataSet.h"
#include "../Common/ReduceLROnPlateauScheduler.h"
#include <iomanip>
#include <iostream>
#include <torch/torch.h>

// Helper function to print tensor statistics - adapted from TrainModel.cpp
void printTensorStats(const torch::Tensor& tensor, const std::string& name)
{
    auto cpu_tensor = tensor.cpu(); // Move to CPU for printing
    std::cout << name << " stats:" << std::endl
        << "  Shape: " << cpu_tensor.sizes() << std::endl
        << "  Range: [" << cpu_tensor.min().item<float>() << ", "
        << cpu_tensor.max().item<float>() << "]" << std::endl
        << "  Mean: " << cpu_tensor.mean().item<float>() << std::endl
        << "  Std: " << cpu_tensor.std().item<float>() << std::endl;
    if (cpu_tensor.isnan().any().item<bool>())
    {
        std::cout << "  WARNING: Contains NaN values!" << std::endl;
    }
    if (cpu_tensor.isinf().any().item<bool>())
    {
        std::cout << "  WARNING: Contains Inf values!" << std::endl;
    }
}

namespace torch_explorer
{
    void TrainCovid19Module(
        std::shared_ptr<Covid19Module> model,
        std::shared_ptr<IDataSet<Covid19>> trainData,
        std::shared_ptr<IDataSet<Covid19>> testData,
        size_t num_epochs,
        double learningRate,
        size_t logInterval)
    {
        try
        {
            // Check for CUDA availability
            torch::Device device(torch::kCPU);
            if (torch::cuda::is_available())
            {
                std::cout << "CUDA is available! Training on GPU." << std::endl;
                device = torch::Device(torch::kCUDA);
            }

            auto img_dims = trainData->getInputShape();
            std::cout << "Starting training with:" << std::endl
                << "Device: " << (device.is_cuda() ? "GPU" : "CPU") << std::endl
                << "Learning rate: " << learningRate << std::endl
                << "Number of epochs: " << num_epochs << std::endl
                << "Image dimensions: [" << img_dims[0] << ", "
                << img_dims[1] << ", " << img_dims[2] << "]" << std::endl
                << "Number of classes: " << trainData->getNumClasses() << std::endl;

            // Move model to GPU if available
            model->to(device);

            // Create optimizer - only optimize the final layer parameters
            // (this ensures we're only fine-tuning the final layer of ResNet)
            auto optimizer = torch::optim::Adam(model->parameters(), learningRate);
            ReduceLROnPlateauScheduler scheduler(optimizer);

            auto trainLoader = trainData->getDataLoader();
            auto testLoader = testData->getDataLoader();

            // Print initial parameter stats
            std::cout << "\nInitial model parameters:" << std::endl;
            for (const auto& p : model->parameters())
            {
                printTensorStats(p, "Parameter");
            }

            model->train();
            for (size_t epoch = 0; epoch < num_epochs; ++epoch)
            {
                size_t batch_idx = 0;
                float epoch_loss = 0.0f;
                size_t num_samples = 0;
                size_t num_correct = 0;

                std::cout << "\nStarting epoch " << epoch << std::endl;

                for (auto& batch : *trainLoader)
                {
                    std::vector<torch::Tensor> data_vec, target_vec;
                    for (const auto& example : batch)
                    {
                        data_vec.push_back(example.data);
                        target_vec.push_back(example.target);
                    }

                    auto data = stack(data_vec).to(device);
                    auto target = stack(target_vec).to(torch::kInt64).to(device);

                    if (batch_idx == 0)
                    {
                        printTensorStats(data, "Input batch");
                        std::cout << "Target values: " << target.cpu() << std::endl;
                    }

                    optimizer.zero_grad();
                    auto output = model->forward(data);

                    if (batch_idx == 0)
                    {
                        printTensorStats(output, "Model output");
                    }

                    auto loss = torch::nn::functional::cross_entropy(output, target);

                    if (loss.isnan().any().item<bool>())
                    {
                        std::cout << "WARNING: Loss is NaN!" << std::endl;
                        continue;
                    }

                    loss.backward();
                    optimizer.step();

                    // Compute accuracy
                    auto pred = output.argmax(1);
                    num_correct += pred.eq(target).sum().item<int64_t>();
                    num_samples += target.size(0);
                    epoch_loss += loss.item<float>();

                    if (batch_idx % logInterval == 0)
                    {
                        std::cout << "Train Epoch: " << epoch
                            << " [" << batch_idx * target.size(0) << "/"
                            << trainData->size().value() << "] "
                            << "Loss: " << std::fixed << std::setprecision(4)
                            << loss.item<float>() << std::endl;

                        if (batch_idx == 0)
                        {
                            std::cout << "Gradient statistics:" << std::endl;
                            for (const auto& p : model->parameters())
                            {
                                if (p.grad().defined())
                                {
                                    printTensorStats(p.grad(), "Gradient");
                                }
                            }
                        }
                    }
                    batch_idx++;
                }

                float accuracy = static_cast<float>(num_correct) / num_samples;
                epoch_loss /= batch_idx;

                std::cout << "Epoch: " << epoch
                    << " Average loss: " << std::fixed << std::setprecision(5)
                    << epoch_loss
                    << " Accuracy: " << accuracy * 100.0f << "%" << std::endl;

                // Validation phase
                model->eval();
                torch::NoGradGuard no_grad;

                float test_loss = 0.0f;
                num_correct = 0;
                num_samples = 0;
                batch_idx = 0;

                for (const auto& batch : *testLoader)
                {
                    std::vector<torch::Tensor> data_vec, target_vec;
                    for (const auto& example : batch)
                    {
                        data_vec.push_back(example.data);
                        target_vec.push_back(example.target);
                    }

                    auto data = stack(data_vec).to(device);
                    auto target = stack(target_vec).to(torch::kInt64).to(device);

                    auto output = model->forward(data);
                    test_loss += torch::nn::functional::cross_entropy(
                        output, target, {}, torch::Reduction::Sum).item<float>();

                    auto pred = output.argmax(1);
                    num_correct += pred.eq(target).sum().item<int64_t>();
                    num_samples += target.size(0);
                    batch_idx++;
                }

                test_loss /= num_samples;
                accuracy = static_cast<float>(num_correct) / num_samples;

                std::cout << "Test set: Average loss: " << test_loss
                    << " Accuracy: " << accuracy * 100.0f << "%" << std::endl;

                scheduler.doStep(test_loss);

                std::cout << "Current learning rate: "
                    << scheduler.getLearningRates() << std::endl;

                model->train();
            }

            // Save the model
            torch::save(model, "covid19_model.pt");
            std::cout << "Model saved to covid19_model.pt" << std::endl;
        }
        catch (const std::exception& ex)
        {
            std::cout << "Error: " << ex.what() << std::endl;
        }
    }
}
