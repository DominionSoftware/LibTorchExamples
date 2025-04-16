#ifndef HYBRID_OPTIMIZER_
#define HYBRID_OPTIMIZER_


#include <torch/torch.h>
#include <memory>
#include "IOptimizer.h"


namespace torch_explorer
{
	class HybridOptimizer : public IOptimizer
	{
	public:
		HybridOptimizer(std::shared_ptr<torch::nn::Module> model,
		                double initial_lr,
		                size_t epoch_tolerance = 10,
		                double threshold = 1e-4)
			: IOptimizer(),
			model_(model)
		    , epoch_tolerance_(epoch_tolerance)
			, threshold_(threshold)
			, best_loss_(std::numeric_limits<double>::max())
			, epochs_without_improvement_(0)
			, current_epoch_(0)
			, switched_(false)
		{
			// Start with Adam
			adam_optimizer_ = std::make_unique<torch::optim::Adam>(
				model->parameters(),
				torch::optim::AdamOptions(initial_lr).weight_decay(1e-4)
			);

			// Initialize SGD (won't be used until switch)
			sgd_optimizer_ = std::make_unique<torch::optim::SGD>(
				model->parameters(),
				torch::optim::SGDOptions(initial_lr)
				.momentum(0.9)
				.weight_decay(1e-4)
			);

			current_optimizer_ = adam_optimizer_.get();
		}

		void step() override
		{
			current_optimizer_->step();
		}

		void zero_grad() override
		{
			current_optimizer_->zero_grad();
		}

 		
		bool epoch_step(float current_loss) override
		{
			 
			current_epoch_++;

			// Track loss improvement
			if (current_loss < best_loss_ - threshold_) 
			{
				epochs_without_improvement_ = 0;
				best_loss_ = current_loss;
			}
			else 
			{
				epochs_without_improvement_++;
			}

			// Switch if stuck for too long and still using Adam
			if (epochs_without_improvement_ >= epoch_tolerance_ &&
				current_optimizer_ == adam_optimizer_.get())
			{
				std::cout << "Switching optimizer from Adam to SGD at epoch "
					<< current_epoch_ << " due to stagnation." << std::endl;
				std::cout << "No improvement for " << epochs_without_improvement_
					<< " epochs. Best loss: " << best_loss_
					<< ", Current loss: " << current_loss << std::endl;

				current_optimizer_ = sgd_optimizer_.get();
				
				epochs_without_improvement_ = 0;  // Reset counter after switch
				return true;
			}
			return false;
		}
		

		torch::optim::Optimizer& current() override
		{
			return *current_optimizer_;
		}

		size_t getCurrentEpoch() const
		{
			return current_epoch_;
		}
		float getBestLoss() const
		{
			return best_loss_;
		}
		size_t getEpochsWithoutImprovement() const
		{
			return epochs_without_improvement_;
		}

	private:
		std::shared_ptr<torch::nn::Module> model_;
		std::unique_ptr<torch::optim::Adam> adam_optimizer_;
		std::unique_ptr<torch::optim::SGD> sgd_optimizer_;
		torch::optim::Optimizer* current_optimizer_;
		size_t epoch_tolerance_;
		double threshold_;
		float best_loss_;
		size_t epochs_without_improvement_;
		size_t current_epoch_;
		bool switched_;
 
	};
}

#endif
