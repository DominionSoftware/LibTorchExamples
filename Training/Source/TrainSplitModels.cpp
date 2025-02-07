#include "TrainSplitModels.h"
#include "IDataSet.h"
#include <iomanip>
#include "CIFAR100FineModule.h"
#include "CIFAR100CoarseModule.h"
#include "ReduceLROnPlateauScheduler.h"
#include "CutMixTransform.h"
#include <memory>
#include "OneHot.h"
#include "KullbackLieblerDivergenceLoss.h"


namespace torch_explorer
{
    

    void TrainSplitModels(std::shared_ptr<CIFAR100CoarseModule> coarse_model,
                         std::shared_ptr<CIFAR100FineModule> fine_model,
                         std::shared_ptr<IDataSet> trainData,
                         std::shared_ptr<IDataSet> testData,
                         size_t num_epochs,
                         double coarse_lr,
                         double fine_lr,
                         size_t logInterval)
    {
        try
        {
            CutMixTransform cutmix(1.0, 0.5);  // alpha=1.0, prob=0.5
            bool use_cutmix = true;
            const int64_t num_fine_classes = 100;    // CIFAR100 fine classes
            const int64_t num_coarse_classes = 20;   // CIFAR100 coarse classes

            torch::Device device(torch::kCPU);
            if (torch::cuda::is_available()) {
                std::cout << "CUDA is available! Training on GPU." << std::endl;
                device = torch::Device(torch::kCUDA);
            }

            // Move models to device
            coarse_model->to(device);
            fine_model->to(device);

            torch::optim::Adam coarse_optimizer(coarse_model->parameters(), coarse_lr);
            torch::optim::Adam fine_optimizer(fine_model->parameters(), fine_lr);

            ReduceLROnPlateauScheduler coarse_scheduler(coarse_optimizer);
            ReduceLROnPlateauScheduler fine_scheduler(fine_optimizer);

            auto trainLoader = trainData->getDataLoader();
            auto testLoader = testData->getDataLoader();

            // Create mapping tensor
            const auto& mapping_data = CIFAR100ClassNames::instance().FineToCoarse();
            auto mapping_tensor = torch::from_blob(
                (void*)mapping_data.data(),
                { static_cast<int64_t>(mapping_data.size()) },
                torch::TensorOptions().dtype(torch::kInt64)
            ).clone().to(device);

            for (size_t epoch = 0; epoch < num_epochs; ++epoch) {
                size_t batch_idx = 0;
                float coarse_epoch_loss = 0.0f;
                float fine_epoch_loss = 0.0f;
                size_t num_samples = 0;
                size_t num_correct_fine = 0;
                size_t num_correct_coarse = 0;

                coarse_model->train();
                fine_model->train();

                for (auto& batch : *trainLoader) {
                    std::vector<torch::Tensor> data_vec, target_vec;
                    for (const auto& example : batch) {
                        data_vec.push_back(example.data);
                        target_vec.push_back(example.target);
                    }

                    auto data = stack(data_vec).to(device);
                    auto fine_target = stack(target_vec).to(torch::kInt64).to(device);
                    
                    torch::Tensor fine_target_one_hot;
                    torch::Tensor coarse_target_one_hot;

                    if (use_cutmix && trainData->isTraining()) {
                        auto example1 = torch::data::Example<>(data, fine_target);
                        auto example2 = torch::data::Example<>(data.clone(), fine_target.clone());
                        auto mixed = cutmix.apply(example1, example2);
                        data = mixed.data;
                        
                        // Convert mixed labels to one-hot
                        auto lambda = mixed.target.max(); // Extract lambda from mixed target
                        auto target1 = to_one_hot(example1.target, num_fine_classes);
                        auto target2 = to_one_hot(example2.target, num_fine_classes);
                        fine_target_one_hot = lambda * target1 + (1 - lambda) * target2;
                        
                        // Handle coarse labels
                        auto coarse_target1 = to_one_hot(mapping_tensor.index_select(0, example1.target), num_coarse_classes);
                        auto coarse_target2 = to_one_hot(mapping_tensor.index_select(0, example2.target), num_coarse_classes);
                        coarse_target_one_hot = lambda * coarse_target1 + (1 - lambda) * coarse_target2;
                    } else {
                        fine_target_one_hot = to_one_hot(fine_target, num_fine_classes);
                        coarse_target_one_hot = to_one_hot(mapping_tensor.index_select(0, fine_target), num_coarse_classes);
                    }

                    // Train coarse model
                    {
                        coarse_optimizer.zero_grad();
                        auto coarse_out = coarse_model->forward(data);
                        auto coarse_loss = kl_divergence_loss(coarse_out, coarse_target_one_hot);
                        coarse_loss.backward();
                        coarse_optimizer.step();
                        coarse_epoch_loss += coarse_loss.item<float>();

                        auto pred_coarse = coarse_out.argmax(1);
                        auto true_coarse = coarse_target_one_hot.argmax(1);
                        num_correct_coarse += pred_coarse.eq(true_coarse).sum().item<int64_t>();
                    }

                    // Train fine model
                    {
                        fine_optimizer.zero_grad();
                        auto fine_out = fine_model->forward(data);
                        auto fine_loss = kl_divergence_loss(fine_out, fine_target_one_hot);
                        fine_loss.backward();
                        fine_optimizer.step();
                        fine_epoch_loss += fine_loss.item<float>();

                        auto pred_fine = fine_out.argmax(1);
                        auto true_fine = fine_target_one_hot.argmax(1);
                        num_correct_fine += pred_fine.eq(true_fine).sum().item<int64_t>();
                    }

                    num_samples += fine_target.size(0);

                    if (batch_idx % logInterval == 0) {
                        std::cout << "Train Epoch: " << epoch
                            << " [" << batch_idx * fine_target.size(0) << "/"
                            << trainData->size().value() << "] "
                            << "Coarse Loss: " << std::fixed << std::setprecision(4)
                            << coarse_epoch_loss / (batch_idx + 1)
                            << " Fine Loss: "
                            << fine_epoch_loss / (batch_idx + 1) << std::endl;
                    }
                    batch_idx++;
                }

                // Rest of the code (validation loop) remains similar but uses one-hot encoding
                // for validation metrics...
                
                // Validation phase
                coarse_model->eval();
                fine_model->eval();
                torch::NoGradGuard no_grad;

                float test_coarse_loss = 0.0f;
                float test_fine_loss = 0.0f;
                num_correct_fine = 0;
                num_correct_coarse = 0;
                num_samples = 0;
                batch_idx = 0;

                for (const auto& batch : *testLoader) {
                    std::vector<torch::Tensor> data_vec, target_vec;
                    for (const auto& example : batch) {
                        data_vec.push_back(example.data);
                        target_vec.push_back(example.target);
                    }

                    auto data = stack(data_vec).to(device);
                    auto fine_target = stack(target_vec).to(torch::kInt64).to(device);
                    
                    // Convert to one-hot for validation
                    auto fine_target_one_hot = to_one_hot(fine_target, num_fine_classes);
                    auto coarse_target_one_hot = to_one_hot(
                        mapping_tensor.index_select(0, fine_target),
                        num_coarse_classes
                    );

                    auto coarse_out = coarse_model->forward(data);
                    auto fine_out = fine_model->forward(data);

                    test_coarse_loss += kl_divergence_loss(coarse_out, coarse_target_one_hot).item<float>();
                    test_fine_loss += kl_divergence_loss(fine_out, fine_target_one_hot).item<float>();

                    auto pred_fine = fine_out.argmax(1);
                    auto pred_coarse = coarse_out.argmax(1);
                    auto true_fine = fine_target_one_hot.argmax(1);
                    auto true_coarse = coarse_target_one_hot.argmax(1);
                    
                    num_correct_fine += pred_fine.eq(true_fine).sum().item<int64_t>();
                    num_correct_coarse += pred_coarse.eq(true_coarse).sum().item<int64_t>();
                    num_samples += fine_target.size(0);
                    batch_idx++;
                }

                test_coarse_loss /= batch_idx;
                test_fine_loss /= batch_idx;
                float accuracy_fine = static_cast<float>(num_correct_fine) / num_samples;
                float accuracy_coarse = static_cast<float>(num_correct_coarse) / num_samples;

                std::cout << "Test set: "
                    << "Coarse Loss: " << test_coarse_loss
                    << " Fine Loss: " << test_fine_loss
                    << " Coarse Accuracy: " << accuracy_coarse * 100.0f << "%"
                    << " Fine Accuracy: " << accuracy_fine * 100.0f << "%" << std::endl;

                coarse_scheduler.doStep(test_coarse_loss);
                fine_scheduler.doStep(test_fine_loss);

                std::cout << "Current learning rates - "
                    << "Coarse: " << coarse_optimizer.defaults().get_lr()
                    << " Fine: " << fine_optimizer.defaults().get_lr() << std::endl;
            }
        }
        catch (const std::exception& ex) {
            std::cout << ex.what() << std::endl;
        }
    }
}
