#include "TrainModel.h"
#include "IDataSet.h"

#include <iomanip>

void printTensorStats(const torch::Tensor& tensor, const std::string& name) {
    auto cpu_tensor = tensor.cpu();  // Move to CPU for printing
    std::cout << name << " stats:" << std::endl
        << "  Shape: " << cpu_tensor.sizes() << std::endl
        << "  Range: [" << cpu_tensor.min().item<float>() << ", "
        << cpu_tensor.max().item<float>() << "]" << std::endl
        << "  Mean: " << cpu_tensor.mean().item<float>() << std::endl
        << "  Std: " << cpu_tensor.std().item<float>() << std::endl;
    if (cpu_tensor.isnan().any().item<bool>()) {
        std::cout << "  WARNING: Contains NaN values!" << std::endl;
    }
    if (cpu_tensor.isinf().any().item<bool>()) {
        std::cout << "  WARNING: Contains Inf values!" << std::endl;
    }
}

void TrainModel(std::shared_ptr<IDataSet> trainData, std::shared_ptr<IDataSet> testData,
    size_t num_epochs, double learningRate, size_t logInterval)
{
    // Check for CUDA availability
    torch::Device device(torch::kCPU);
    if (torch::cuda::is_available()) {
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

    torch::nn::Sequential model(
        torch::nn::Conv2d(torch::nn::Conv2dOptions(img_dims[0], 32, 3).padding(1)),
        torch::nn::Functional(torch::relu),
        torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2)),

        torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, 3).padding(1)),
        torch::nn::Functional(torch::relu),
        torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2)),

        torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 3).padding(1)),
        torch::nn::Functional(torch::relu),
        torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2)),

        torch::nn::Flatten(),
        torch::nn::Linear(64 * (img_dims[1] / 8) * (img_dims[2] / 8), 512),
        torch::nn::Functional(torch::relu),
        torch::nn::Dropout(0.5),
        torch::nn::Linear(512, trainData->getNumClasses())
    );

    // Move model to GPU if available
    model->to(device);
    torch::optim::AdamW optimizer(model->parameters(), learningRate);
   // torch::optim::Adam optimizer(model->parameters(), learningRate);
    auto trainLoader = trainData->getDataLoader();
    auto testLoader = testData->getDataLoader();

    // Print initial parameter stats
    std::cout << "\nInitial model parameters:" << std::endl;
    for (const auto& p : model->parameters()) {
        printTensorStats(p, "Parameter");
    }

    model->train();
    for (size_t epoch = 0; epoch < num_epochs; ++epoch) {
        size_t batch_idx = 0;
        float epoch_loss = 0.0f;
        size_t num_correct = 0;
        size_t num_samples = 0;

        std::cout << "\nStarting epoch " << epoch << std::endl;

        for (auto& batch : *trainLoader) {
            std::vector<torch::Tensor> data_vec, target_vec;
            for (const auto& example : batch) {
                data_vec.push_back(example.data);
                target_vec.push_back(example.target);
            }

            auto data = torch::stack(data_vec).to(device);
            auto target = torch::stack(target_vec).to(device);

            if (batch_idx == 0) {
                printTensorStats(data, "Input batch");
                std::cout << "Target values: " << target.cpu() << std::endl;
                // Count unique values manually
                auto acc = std::set<int64_t>();
                for (int64_t i = 0; i < target.numel(); ++i) {
                    acc.insert(target[i].item<int64_t>());
                }
                std::cout << "Unique target values: ";
                for (auto v : acc) {
                    std::cout << v << " ";
                }
                std::cout << std::endl;
            }
            optimizer.zero_grad();
            auto output = model->forward(data);

            if (batch_idx == 0) {
                printTensorStats(output, "Model output");
                std::cout << "Output for first example:\n" << output[0].cpu() << std::endl;
            }

            auto loss = torch::nn::functional::cross_entropy(output, target);

            if (loss.isnan().any().item<bool>()) {
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

            if (batch_idx % logInterval == 0) {
                std::cout << "Train Epoch: " << epoch
                    << " [" << batch_idx * target.size(0) << "/"
                    << trainData->size().value() << "] "
                    << "Loss: " << std::fixed << std::setprecision(4)
                    << loss.item<float>() << std::endl;

                if (batch_idx == 0) {
                    std::cout << "Gradient statistics:" << std::endl;
                    for (const auto& p : model->parameters()) {
                        if (p.grad().defined()) {
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

        for (const auto& batch : *testLoader) {
            std::vector<torch::Tensor> data_vec, target_vec;
            for (const auto& example : batch) {
                data_vec.push_back(example.data);
                target_vec.push_back(example.target);
            }

            auto data = torch::stack(data_vec).to(device);
            auto target = torch::stack(target_vec).to(device);

            auto output = model->forward(data);
            test_loss += torch::nn::functional::cross_entropy(output, target).item<float>();

            auto pred = output.argmax(1);
            num_correct += pred.eq(target).sum().item<int64_t>();
            num_samples += target.size(0);
            batch_idx++;
        }

        test_loss /= batch_idx;
        accuracy = static_cast<float>(num_correct) / num_samples;

        std::cout << "Test set: Average loss: " << test_loss
            << " Accuracy: " << accuracy * 100.0f << "%" << std::endl;

        model->train();
    }
}