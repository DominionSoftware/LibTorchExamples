#include "TrainModel.h"
#include "IDataSet.h"


void TrainModel(std::shared_ptr<IDataSet> trainData, std::shared_ptr<IDataSet> testData,
				size_t num_epochs,double learningRate,size_t logInterval)
{
	auto img_dims = trainData->getInputShape();
	std::cout << "Image dimensions: [" << img_dims[0] << ", "
		<< img_dims[1] << ", " << img_dims[2] << "]" << std::endl;


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

    torch::optim::Adam optimizer(model->parameters(), learningRate);

    auto trainLoader = trainData->getDataLoader();
    auto testLoader = testData->getDataLoader();


    model->train();


    for (size_t epoch = 0; epoch < num_epochs; ++epoch) {
        size_t batch_idx = 0;
        float epoch_loss = 0.0f;
        size_t num_correct = 0;
        size_t num_samples = 0;

        for (auto& batch : *trainLoader) {
            auto data = batch[0];
            auto target = batch[1].target.squeeze();

            // Forward pass
            optimizer.zero_grad();
            auto output = model->forward(data);
            auto loss = torch::nn::functional::cross_entropy(output, target);

            // Backward pass
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
                    << "Loss: " << loss.item<float>() << std::endl;
            }
            batch_idx++;
        }

        float accuracy = static_cast<float>(num_correct) / num_samples;
        epoch_loss /= batch_idx;

        std::cout << "Epoch: " << epoch
            << " Average loss: " << epoch_loss
            << " Accuracy: " << accuracy * 100.0f << "%" << std::endl;

        // Validation phase
        model->eval();
        torch::NoGradGuard no_grad;

        float test_loss = 0.0f;
        num_correct = 0;
        num_samples = 0;
        batch_idx = 0;
        for (const auto& batch : *testLoader) {
            auto data = batch[0];    // First element is the input data
            auto target = batch[1];  // Second element is the target/label

            auto output = model->forward(data);
            test_loss += torch::nn::functional::cross_entropy(output, target.target).item<float>();

            auto pred = output.argmax(1);
            num_correct += pred.eq(target.target).sum().item<int64_t>();
            num_samples += target.target.size(0);
            batch_idx++;
        }

        test_loss /= batch_idx;
        accuracy = static_cast<float>(num_correct) / num_samples;

        std::cout << "Test set: Average loss: " << test_loss
            << " Accuracy: " << accuracy * 100.0f << "%" << std::endl;

        model->train();
    }
}
