#include "TrainCovid19Module.h"
#include "Covid19Module.h"
#include "../Common/IDataSet.h"
#include "../Common/ReduceLROnPlateauScheduler.h"
#include <iomanip>
#include <iostream>
#include <torch/torch.h>

// Helper function to print tensor statistics
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
                device = torch::Device(torch::kCUDA);
                std::cout << "CUDA is available. Using GPU:" << std::endl;
            }
            else
            {
                std::cout << "CUDA is not available. Using CPU." << std::endl;
            }

            auto img_dims = trainData->getInputShape();


            std::cout << "Starting training with:" << std::endl
                << "Device: " << (device.is_cuda() ? "GPU" : "CPU") << std::endl
                << "Learning rate: " << learningRate << std::endl
                << "Number of epochs: " << num_epochs << std::endl
                << "Image dimensions: [" << img_dims[0] << ", "
                << img_dims[1] << ", " << img_dims[2] << "]" << std::endl
                << "Number of classes: " << trainData->getNumClasses() << std::endl;

             model->to(device);

             bool is_on_cuda = false;
             for (const auto& p : model->parameters()) {
                 if (p.device().is_cuda()) {
                     is_on_cuda = true;
                     break;
                 }
             }
             std::cout << "Model is on GPU: " << (is_on_cuda ? "Yes" : "No") << std::endl;

             auto optimizer = torch::optim::Adam(model->parameters(), learningRate);
            ReduceLROnPlateauScheduler scheduler(optimizer);

            auto trainLoader = trainData->getDataLoader();
            auto testLoader = testData->getDataLoader();

            std::cout << "\nModel parameters:" << std::endl;
            for (const auto& p : model->parameters())
            {
                printTensorStats(p, "Parameter");
                std::cout << "  Device: " << p.device() << std::endl;

            }

            model->train();
            for (size_t epoch = 0; epoch < num_epochs; ++epoch)
            {
                size_t batch_idx{ 0 };
                float epoch_loss{ 0.0f };
                size_t num_samples{ 0 };
                size_t num_correct{ 0 };

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
                    std::cout << "Data on CUDA: " << data.is_cuda() << std::endl;
                    std::cout << "Target on CUDA: " << target.is_cuda() << std::endl;

                    if (batch_idx == 0)
                    {
                        printTensorStats(data, "Input batch");
                        std::cout << "Target values: " << target.cpu() << std::endl;
                    }

                    optimizer.zero_grad();
                    torch::Tensor output;
                    try {
                        output = model->forward(data);

                        std::cout << "Output on CUDA: " << output.is_cuda() << ", device: " << output.device() << std::endl;

                        if (batch_idx == 0) {
                            std::cout << "Output is a tensor: " << output.defined() << std::endl;
                            std::cout << "Output shape: " << output.sizes() << std::endl;
                            printTensorStats(output, "Model output");
                        }
                    }
                    catch (const std::exception& e) {
                        std::cerr << "Error in forward pass: " << e.what() << std::endl;
                        continue; // Skip this batch if forward fails
                    }

                    torch::Tensor loss;
                    try {
                        loss = torch::nn::functional::cross_entropy(output, target);
                        std::cout << "Loss on CUDA: " << loss.is_cuda() << ", device: " << loss.device() << std::endl;

                    }
                    catch (const std::exception& e) {
                        std::cerr << "Error in loss calculation: " << e.what() << std::endl;
                        std::cerr << "Output shape: " << output.sizes() << ", Target shape: " << target.sizes() << std::endl;
                        continue; // Skip this batch if loss calculation fails
                    }
                    std::cout << "step 1" << std::endl;
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
                                    std::cout << "  Gradient on CUDA: " << p.grad().is_cuda() << ", device: " << p.grad().device() << std::endl;
                                }
                            }
                        }
                    }
                    batch_idx++;
                }
                std::cout << "step 2" << std::endl;

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
                    std::cout << "step 1-1" << std::endl;

                    auto data = stack(data_vec).to(device);
                    auto target = stack(target_vec).to(torch::kInt64).to(device);

                    torch::Tensor output = model->forward(data);

                    // Create options with reduction=sum
                    auto cross_entropy_options = torch::nn::functional::CrossEntropyFuncOptions().reduction(torch::kSum);

                    // Use options instead of positional arguments
                    test_loss += torch::nn::functional::cross_entropy(
                        output, target, cross_entropy_options).item<float>();

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
                std::cout << "Current learning rate: " << scheduler.getLearningRates() << std::endl;

                model->train();
                if (accuracy > 0.95)
                {
                    std::cout << "Desired accuracy met: " << accuracy << std::endl;

                    break;
                }
            }

            
            torch::save(model, "covid19_model.pt");
            std::cout << "Model saved to covid19_model.pt" << std::endl;

        }
        catch (const std::exception& ex)
        {
            std::cout << "Error: " << ex.what() << std::endl;
        }
    }
}