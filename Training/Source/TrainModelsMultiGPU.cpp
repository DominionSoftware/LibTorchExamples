#include "TrainModelsMultiGPU.h"
#include <torch/torch.h>
#include <iostream>
#include <torch/cuda.h>
#include "CIFAR100ClassNames.h"
#include "CIFAR100CoarseModule.h"
#include "CIFAR100FineModule.h"
#include "IDataSet.h"
#include "ReduceLROnPlateauScheduler.h"
#include "CutMixTransform.h"

namespace torch_explorer
{
   void TrainSplitModelsMultiGPU(
       std::shared_ptr<CIFAR100CoarseModule> coarse_model,
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

           if (!torch::cuda::is_available())
           {
               throw std::runtime_error("CUDA is not available");
           }

           const auto device_count = torch::cuda::device_count();
           if (device_count < 2)
           {
               throw std::runtime_error("Need at least 2 GPUs, found " +
                   std::to_string(device_count));
           }

           torch::Device coarse_device(torch::kCUDA, 0);
           torch::Device fine_device(torch::kCUDA, 1);

           std::cout << "Using " << device_count << " GPUs\n"
               << "Coarse model on GPU 0\n"
               << "Fine model on GPU 1\n"
               << "Learning rates - Coarse: " << coarse_lr
               << ", Fine: " << fine_lr 
               << "\nCutMix enabled: " << (use_cutmix ? "yes" : "no") << std::endl;

           coarse_model->to(coarse_device);
           fine_model->to(fine_device);

           torch::optim::Adam coarse_optimizer(coarse_model->parameters(), coarse_lr);
           torch::optim::Adam fine_optimizer(fine_model->parameters(), fine_lr);

           // Create learning rate schedulers
           ReduceLROnPlateauScheduler coarse_scheduler(coarse_optimizer);
           ReduceLROnPlateauScheduler fine_scheduler(fine_optimizer);

           auto trainLoader = trainData->getDataLoader();
           auto testLoader = testData->getDataLoader();

           const auto& mapping_data = CIFAR100ClassNames::instance().FineToCoarse();
           auto mapping_tensor = torch::tensor(
               std::vector<int64_t>(mapping_data.begin(), mapping_data.end()),
               torch::TensorOptions().dtype(torch::kInt64).device(coarse_device)
           );

           for (size_t epoch = 0; epoch < num_epochs; ++epoch)
           {
               coarse_model->train();
               fine_model->train();

               size_t batch_idx = 0;
               float coarse_epoch_loss = 0.0f;
               float fine_epoch_loss = 0.0f;
               size_t num_correct_coarse = 0;
               size_t num_correct_fine = 0;
               size_t num_samples = 0;

               for (auto& batch : *trainLoader)
               {
                   std::vector<torch::Tensor> data_vec, target_vec;
                   for (const auto& example : batch)
                   {
                       data_vec.push_back(example.data);
                       target_vec.push_back(example.target.to(torch::kInt64));
                   }

                   auto data = stack(data_vec);
                   auto target = stack(target_vec);
                   
                   if (use_cutmix && trainData->isTraining())
                   {
                       auto example1 = torch::data::Example<>(data, target);
                       auto example2 = torch::data::Example<>(data.clone(), target.clone());
                       auto mixed = cutmix.apply(example1, example2);
                       data = mixed.data;
                       target = mixed.target;
                   }

                   auto data_coarse = data.to(coarse_device);
                   auto data_fine = data.to(fine_device);
                   auto target_coarse = target.to(coarse_device);
                   auto target_fine = target.to(fine_device);

                   auto coarse_target = mapping_tensor.index_select(0, target_coarse.to(torch::kInt64));

                   // Train coarse model on GPU 0
                   {
                       coarse_optimizer.zero_grad();
                       auto coarse_out = coarse_model->forward(data_coarse);
                       auto coarse_loss = torch::nn::functional::cross_entropy(
                           coarse_out, coarse_target);
                       coarse_loss.backward();
                       coarse_optimizer.step();

                       coarse_epoch_loss += coarse_loss.item<float>();
                       auto pred_coarse = coarse_out.argmax(1);
                       num_correct_coarse += pred_coarse.eq(coarse_target)
                                                    .sum().item<int64_t>();
                   }

                   // Train fine model on GPU 1
                   {
                       fine_optimizer.zero_grad();
                       auto fine_out = fine_model->forward(data_fine);
                       auto fine_loss = torch::nn::functional::cross_entropy(
                           fine_out, target_fine);
                       fine_loss.backward();
                       fine_optimizer.step();

                       fine_epoch_loss += fine_loss.item<float>();
                       auto pred_fine = fine_out.argmax(1);
                       num_correct_fine += pred_fine.eq(target_fine)
                                                .sum().item<int64_t>();
                   }

                   num_samples += target_fine.size(0);

                   if (batch_idx % logInterval == 0)
                   {
                       std::cout << "Train Epoch: " << epoch
                           << " [" << batch_idx * target_fine.size(0) << "/"
                           << trainData->size().value() << "]\n"
                           << "Coarse Loss: " << std::fixed << std::setprecision(4)
                           << coarse_epoch_loss / (batch_idx + 1)
                           << " Fine Loss: "
                           << fine_epoch_loss / (batch_idx + 1) << std::endl;
                   }
                   batch_idx++;
               }

               float train_coarse_acc = static_cast<float>(num_correct_coarse) / num_samples;
               float train_fine_acc = static_cast<float>(num_correct_fine) / num_samples;
               coarse_epoch_loss /= batch_idx;
               fine_epoch_loss /= batch_idx;

               // Validation phase
               coarse_model->eval();
               fine_model->eval();
               float test_coarse_loss = 0.0f;
               float test_fine_loss = 0.0f;
               num_correct_coarse = 0;
               num_correct_fine = 0;
               num_samples = 0;
               batch_idx = 0;

               {
                   torch::NoGradGuard no_grad;

                   for (const auto& batch : *testLoader)
                   {
                       std::vector<torch::Tensor> data_vec, target_vec;
                       for (const auto& example : batch)
                       {
                           data_vec.push_back(example.data);
                           target_vec.push_back(example.target.to(torch::kInt64));
                       }

                       auto data = stack(data_vec);
                       auto target = stack(target_vec);

                       auto data_coarse = data.to(coarse_device);
                       auto data_fine = data.to(fine_device);
                       auto target_coarse = target.to(coarse_device);
                       auto target_fine = target.to(fine_device);

                       auto coarse_target = mapping_tensor.index_select(0, target_coarse.to(torch::kInt64));

                       auto coarse_out = coarse_model->forward(data_coarse);
                       auto fine_out = fine_model->forward(data_fine);

                       test_coarse_loss += torch::nn::functional::cross_entropy(
                           coarse_out, coarse_target).item<float>();
                       test_fine_loss += torch::nn::functional::cross_entropy(
                           fine_out, target_fine).item<float>();

                       auto pred_coarse = coarse_out.argmax(1);
                       auto pred_fine = fine_out.argmax(1);

                       num_correct_coarse += pred_coarse.eq(coarse_target)
                                                    .sum().item<int64_t>();
                       num_correct_fine += pred_fine.eq(target_fine)
                                                .sum().item<int64_t>();

                       num_samples += target_fine.size(0);
                       batch_idx++;
                   }
               }

               test_coarse_loss /= batch_idx;
               test_fine_loss /= batch_idx;
               float test_coarse_acc = static_cast<float>(num_correct_coarse) / num_samples;
               float test_fine_acc = static_cast<float>(num_correct_fine) / num_samples;

               coarse_scheduler.doStep(test_coarse_loss);
               fine_scheduler.doStep(test_fine_loss);

               std::cout << "\nEpoch " << epoch << " Summary:\n"
                   << "Training - Coarse: loss=" << coarse_epoch_loss
                   << ", acc=" << train_coarse_acc * 100.0f << "%\n"
                   << "Training - Fine: loss=" << fine_epoch_loss
                   << ", acc=" << train_fine_acc * 100.0f << "%\n"
                   << "Validation - Coarse: loss=" << test_coarse_loss
                   << ", acc=" << test_coarse_acc * 100.0f << "%\n"
                   << "Validation - Fine: loss=" << test_fine_loss
                   << ", acc=" << test_fine_acc * 100.0f << "%\n"
                   << "Learning rates - Coarse: " << coarse_scheduler.getLearningRates()
                   << ", Fine: " << fine_scheduler.getLearningRates()
                   << std::endl;
           }
       }
       catch (const std::exception& ex)
       {
           std::cout << "Error: " << ex.what() << std::endl;
       }
   }
} // namespace torch_explorer
