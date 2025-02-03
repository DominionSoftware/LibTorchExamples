#include "TrainModel.h"
#include "IDataSet.h"
#include <iomanip>
#include "CIFAR100Module.h"
#include "CIFAR100ClassNames.h"
#include "HeirarchicalLoss.h"

class ReduceLROnPlateau : public torch::optim::LRScheduler
{
public:
	ReduceLROnPlateau(torch::optim::Optimizer& optimizer,
	                  double factor = 0.1,
	                  size_t patience = 10,
	                  double min_lr = 1e-6,
	                  double threshold = 1e-4)
		: LRScheduler(optimizer),
		  factor_(factor),
		  patience_(patience),
		  min_lr_(min_lr),
		  threshold_(threshold),
		  best_loss_(std::numeric_limits<double>::max()),
		  bad_epochs_(0)
	{
	}

protected:
	std::vector<double> get_lrs() override
	{
		auto current_lrs = get_current_lrs();
		std::vector<double> new_lrs;

		// If we've waited long enough with no improvement
		if (bad_epochs_ >= patience_)
		{
			for (double lr : current_lrs)
			{
				// Reduce learning rate but don't go below min_lr
				new_lrs.push_back(std::max(lr * factor_, min_lr_));
			}
			bad_epochs_ = 0; // Reset counter
		}
		else
		{
			new_lrs = current_lrs; // Keep same learning rates
		}

		return new_lrs;
	}

public:
	// Call this instead of step() to track loss
	void stepWithLoss(double loss)
	{
		if (loss < best_loss_ - threshold_)
		{
			best_loss_ = loss;
			bad_epochs_ = 0;
		}
		else
		{
			bad_epochs_++;
		}
		step();
	}

private:
	double factor_;
	size_t patience_;
	double min_lr_;
	double threshold_;
	double best_loss_;
	size_t bad_epochs_;
};

class CosineAnnealingLR : public torch::optim::LRScheduler
{
public:
	CosineAnnealingLR(torch::optim::Optimizer& optimizer,
	                  size_t T_max,
	                  double eta_min = 0)
		: LRScheduler(optimizer),
		  T_max_(T_max),
		  eta_min_(eta_min)
	{
	}

protected:
	std::vector<double> get_lrs() override
	{
		std::vector<double> new_lrs;
		auto current_lrs = get_current_lrs();

		double factor = 0.5 * (1 + std::cos(M_PI * step_count_ / T_max_));

		for (double lr : current_lrs)
		{
			new_lrs.push_back(eta_min_ + (lr - eta_min_) * factor);
		}
		return new_lrs;
	}

private:
	size_t T_max_;
	double eta_min_;
};

class StepLRScheduler : public torch::optim::LRScheduler
{
public:
	StepLRScheduler(torch::optim::Optimizer& optimizer,
	                const std::vector<size_t>& milestones,
	                double gamma)
		: LRScheduler(optimizer),
		  milestones_(milestones),
		  gamma_(gamma)
	{
	}

protected:
	std::vector<double> get_lrs() override
	{
		auto current_lrs = get_current_lrs();
		double factor = 1.0;

		for (const auto milestone : milestones_)
		{
			if (step_count_ >= milestone)
			{
				factor *= gamma_;
			}
		}

		std::vector<double> new_lrs;
		for (double lr : current_lrs)
		{
			new_lrs.push_back(lr * factor);
		}
		return new_lrs;
	}

private:
	std::vector<size_t> milestones_;
	double gamma_;
};

class OneCycleLR : public torch::optim::LRScheduler
{
public:
	OneCycleLR(torch::optim::Optimizer& optimizer,
	           size_t total_steps,
	           double max_lr,
	           double div_factor = 25.0)
		: LRScheduler(optimizer),
		  total_steps_(total_steps),
		  max_lr_(max_lr),
		  initial_lr_(max_lr / div_factor)
	{
	}

protected:
	std::vector<double> get_lrs() override
	{
		std::vector<double> new_lrs;
		double progress = static_cast<double>(step_count_) / total_steps_;

		for (double _ : get_current_lrs())
		{
			if (progress < 0.3)
			{
				// First 30%: linear warmup
				double lr = initial_lr_ + (max_lr_ - initial_lr_) * (progress / 0.3);
				new_lrs.push_back(lr);
			}
			else
			{
				// Remaining 70%: cosine annealing
				progress = (progress - 0.3) / 0.7;
				double lr = initial_lr_ + (max_lr_ - initial_lr_) *
					0.5 * (1 + std::cos(M_PI * progress));
				new_lrs.push_back(lr);
			}
		}
		return new_lrs;
	}

private:
	size_t total_steps_;
	double max_lr_;
	double initial_lr_;
};

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
	void TrainModel(std::shared_ptr<CIFAR100Module> model,
	                std::shared_ptr<IDataSet> trainData,
	                std::shared_ptr<IDataSet> testData,
	                size_t num_epochs, double learningRate, size_t logInterval)
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

			torch::optim::Adam optimizer(model->parameters(), learningRate);
			ReduceLROnPlateau scheduler(optimizer);

			auto trainLoader = trainData->getDataLoader();
			auto testLoader = testData->getDataLoader();

			// Create mapping tensor
			const auto& mapping_data = CIFAR100ClassNames::FineToCoarse();

			// Create initial tensor on CPU and clone to make it independent
			auto mapping_tensor = torch::from_blob(
				(void*)mapping_data.data(),
				{static_cast<int64_t>(mapping_data.size())},
				{1},
				torch::TensorOptions().dtype(torch::kInt64)
			).clone();

			// Move to the appropriate device
			mapping_tensor = mapping_tensor.to(device);

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
				size_t num_correct_fine = 0;
				size_t num_correct_coarse = 0;

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
					auto coarse_target = mapping_tensor.index_select(0, target);

					if (batch_idx == 0)
					{
						printTensorStats(data, "Input batch");
						std::cout << "Fine target values: " << target.cpu() << std::endl;
						std::cout << "Coarse target values: " << coarse_target.cpu() << std::endl;
					}

					optimizer.zero_grad();
					auto output = model->forward(data);
					auto [coarse_out, fine_out] = output;

					if (batch_idx == 0)
					{
						printTensorStats(coarse_out, "Coarse output");
						printTensorStats(fine_out, "Fine output");
					}

					auto loss = HeirachicalLoss(output, std::make_tuple(coarse_target, target));

					if (loss.isnan().any().item<bool>())
					{
						std::cout << "WARNING: Loss is NaN!" << std::endl;
						continue;
					}

					loss.backward();
					optimizer.step();

					// Compute accuracy
					auto pred_fine = fine_out.argmax(1);
					auto pred_coarse = coarse_out.argmax(1);
					num_correct_fine += pred_fine.eq(target).sum().item<int64_t>();
					num_correct_coarse += pred_coarse.eq(coarse_target).sum().item<int64_t>();
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

				float accuracy_fine = static_cast<float>(num_correct_fine) / num_samples;
				float accuracy_coarse = static_cast<float>(num_correct_coarse) / num_samples;
				epoch_loss /= batch_idx;

				std::cout << "Epoch: " << epoch
					<< " Average loss: " << std::fixed << std::setprecision(5)
					<< epoch_loss
					<< " Fine Accuracy: " << accuracy_fine * 100.0f << "%"
					<< " Coarse Accuracy: " << accuracy_coarse * 100.0f << "%" << std::endl;

				// Validation phase
				model->eval();
				torch::NoGradGuard no_grad;

				float test_loss = 0.0f;
				num_correct_fine = 0;
				num_correct_coarse = 0;
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
					auto coarse_target = mapping_tensor.index_select(0, target);

					auto output = model->forward(data);
					test_loss += HeirachicalLoss(output,
					                             std::make_tuple(coarse_target, target)).item<float>();

					auto [coarse_out, fine_out] = output;
					auto pred_fine = fine_out.argmax(1);
					auto pred_coarse = coarse_out.argmax(1);
					num_correct_fine += pred_fine.eq(target).sum().item<int64_t>();
					num_correct_coarse += pred_coarse.eq(coarse_target).sum().item<int64_t>();
					num_samples += target.size(0);
					batch_idx++;
				}

				test_loss /= batch_idx;
				accuracy_fine = static_cast<float>(num_correct_fine) / num_samples;
				accuracy_coarse = static_cast<float>(num_correct_coarse) / num_samples;

				std::cout << "Test set: Average loss: " << test_loss
					<< " Fine Accuracy: " << accuracy_fine * 100.0f << "%"
					<< " Coarse Accuracy: " << accuracy_coarse * 100.0f << "%" << std::endl;

				scheduler.stepWithLoss(epoch_loss);

				std::cout << "Current learning rate: "
					<< optimizer.defaults().get_lr() << std::endl;

				model->train();
			}
		}
		catch (const std::exception& ex)
		{
			std::cout << ex.what() << std::endl;
		}
	}
}
