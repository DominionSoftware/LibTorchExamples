#include "TrainModelsMultiGPU.h"
#include <torch/torch.h>
#include <iostream>
#include <torch/cuda.h>
#include "CIFAR100ClassNames.h"
#include "CIFAR100CoarseModule.h"
#include "CIFAR100FineModule.h"
#include "CosineAnnealingScheduler.h"
#include "IDataSet.h"
#include "CutMixTransform.h"
#include "OneHot.h"
#include "KullbackLieblerDivergenceLoss.h"
#include "HybridOptimizer.h"

namespace torch_explorer
{

	void OptimizerSwitch(HybridOptimizer& optimizer,CosineAnnealingScheduler& scheduler,double new_lr = 0.001)
	{
		// Reset scheduler with new learning rate
		scheduler.resetScheduler(new_lr);

		// Set initial learning rate for new optimizer
		for (auto& group : optimizer.current().param_groups()) 
		{
			group.options().set_lr(new_lr);
		}
	}
	

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
			CutMixTransform cutmix(1.0, 0.5); 
			bool use_cutmix = true;
			constexpr int64_t num_fine_classes = 100; // CIFAR100 fine classes
			constexpr int64_t num_coarse_classes = 20; // CIFAR100 coarse classes

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

			HybridOptimizer coarse_optimizer(coarse_model, coarse_lr,2);

			HybridOptimizer fine_optimizer(fine_model, fine_lr, 2);
				

			// Create learning rate schedulers
			CosineAnnealingScheduler coarse_scheduler(coarse_optimizer,5,1e-4,2);
			CosineAnnealingScheduler fine_scheduler(fine_optimizer,5,1e-4,2);

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
						target_vec.push_back(example.target);
					}

					auto data = stack(data_vec);
					auto fine_target = stack(target_vec).to(torch::kInt64);

					torch::Tensor fine_target_one_hot;
					torch::Tensor coarse_target_one_hot;

                    if (use_cutmix && trainData->isTraining())
                    {
                        auto example1 = torch::data::Example<>(data, fine_target);
                        auto example2 = torch::data::Example<>(data.clone(), fine_target.clone());
                        auto mixed = cutmix.apply(example1, example2);
                        data = mixed.data;
                        
                        // Move targets to appropriate device before operations
                        auto target1_gpu = example1.target.to(coarse_device);
                        auto target2_gpu = example2.target.to(coarse_device);
                        
                        // Convert mixed labels to one-hot
                        auto lambda = mixed.target.max(); // Extract lambda from mixed target
                        auto target1 = to_one_hot(target1_gpu, num_fine_classes);
                        auto target2 = to_one_hot(target2_gpu, num_fine_classes);
                        fine_target_one_hot = lambda * target1 + (1 - lambda) * target2;
                        
                        // Handle coarse labels
                        auto coarse_target1 = to_one_hot(mapping_tensor.index_select(0, target1_gpu), num_coarse_classes);
                        auto coarse_target2 = to_one_hot(mapping_tensor.index_select(0, target2_gpu), num_coarse_classes);
                        coarse_target_one_hot = lambda * coarse_target1 + (1 - lambda) * coarse_target2;
                    }
                    else
                    {
                        // Move target to appropriate device before operations
                        auto fine_target_gpu = fine_target.to(coarse_device);
                        fine_target_one_hot = to_one_hot(fine_target_gpu, num_fine_classes);
                        coarse_target_one_hot = to_one_hot(mapping_tensor.index_select(0, fine_target_gpu), num_coarse_classes);
                    }

					auto data_coarse = data.to(coarse_device);
					auto data_fine = data.to(fine_device);
					coarse_target_one_hot = coarse_target_one_hot.to(coarse_device);
					fine_target_one_hot = fine_target_one_hot.to(fine_device);

					// Train coarse model on GPU 0
					{
						coarse_optimizer.zero_grad();
						auto coarse_out = coarse_model->forward(data_coarse);
						auto coarse_loss = kl_divergence_loss(coarse_out, coarse_target_one_hot);
						coarse_loss.backward();
						coarse_optimizer.step();

						coarse_epoch_loss += coarse_loss.item<float>();
						auto pred_coarse = coarse_out.argmax(1);
						auto true_coarse = coarse_target_one_hot.argmax(1);
						num_correct_coarse += pred_coarse.eq(true_coarse).sum().item<int64_t>();
					}

					// Train fine model on GPU 1
					{
						fine_optimizer.zero_grad();
						auto fine_out = fine_model->forward(data_fine);
						auto fine_loss = kl_divergence_loss(fine_out, fine_target_one_hot);
						fine_loss.backward();
						fine_optimizer.step();

						fine_epoch_loss += fine_loss.item<float>();
						auto pred_fine = fine_out.argmax(1);
						auto true_fine = fine_target_one_hot.argmax(1);
						num_correct_fine += pred_fine.eq(true_fine).sum().item<int64_t>();
					}

					num_samples += fine_target.size(0);

					if (batch_idx % logInterval == 0)
					{
						std::cout << "Train Epoch: " << epoch
							<< " [" << batch_idx * fine_target.size(0) << "/"
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
							target_vec.push_back(example.target);
						}

						auto data = stack(data_vec);
						auto fine_target = stack(target_vec).to(torch::kInt64);

						auto data_coarse = data.to(coarse_device);
						auto data_fine = data.to(fine_device);

                        // Convert to one-hot for validation
                        // Move target to appropriate device before operations
                        auto fine_target_gpu = fine_target.to(coarse_device);
                        auto fine_target_one_hot = to_one_hot(fine_target_gpu, num_fine_classes);
                        auto coarse_target_one_hot = to_one_hot(
                            mapping_tensor.index_select(0, fine_target_gpu),
                            num_coarse_classes
                        );

						coarse_target_one_hot = coarse_target_one_hot.to(coarse_device);
						fine_target_one_hot = fine_target_one_hot.to(fine_device);

						auto coarse_out = coarse_model->forward(data_coarse);
						auto fine_out = fine_model->forward(data_fine);

						test_coarse_loss += kl_divergence_loss(coarse_out, coarse_target_one_hot).item<float>();
						test_fine_loss += kl_divergence_loss(fine_out, fine_target_one_hot).item<float>();

						auto pred_coarse = coarse_out.argmax(1);
						auto pred_fine = fine_out.argmax(1);
						auto true_coarse = coarse_target_one_hot.argmax(1);
						auto true_fine = fine_target_one_hot.argmax(1);

						num_correct_coarse += pred_coarse.eq(true_coarse).sum().item<int64_t>();
						num_correct_fine += pred_fine.eq(true_fine).sum().item<int64_t>();

						num_samples += fine_target.size(0);
						batch_idx++;
					}
				}

				test_coarse_loss /= batch_idx;
				test_fine_loss /= batch_idx;
				float test_coarse_acc = static_cast<float>(num_correct_coarse) / num_samples;
				float test_fine_acc = static_cast<float>(num_correct_fine) / num_samples;


				bool coarse_changed = coarse_optimizer.epoch_step(coarse_epoch_loss);
				bool fine_changed  = fine_optimizer.epoch_step(fine_epoch_loss);
				if (coarse_changed)
				{
					OptimizerSwitch(coarse_optimizer, coarse_scheduler);
				}

				if (fine_changed)
				{
					OptimizerSwitch(fine_optimizer, fine_scheduler);
				}


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
