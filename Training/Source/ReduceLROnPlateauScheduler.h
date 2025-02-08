#ifndef REDUCE_LR_ON_PLATAEU_SCHEDULER_
#define REDUCE_LR_ON_PLATAEU_SCHEDULER_
#include "ISchedulerStep.h"
#include <torch/optim/schedulers/lr_scheduler.h>



namespace torch_explorer
{


	class ReduceLROnPlateauScheduler : public torch::optim::LRScheduler, public ISchedulerStep
	{
	public:
		ReduceLROnPlateauScheduler(torch::optim::Optimizer& optimizer,
			double factor = 0.1,
			size_t epoch_tolerance = 10,
			double min_lr = 1e-6,
			double threshold = 1e-4);



		void doStep(double param) override;


		std::vector<double> getLearningRates()
		{
			return get_lrs();
		}


	protected:

		void stepWithLoss(double loss);

		std::vector<double> get_lrs() override
		{
			auto current_lrs = get_current_lrs();
			std::vector<double> new_lrs;

			// If we've waited long enough with no improvement
			if (bad_epochs_ >= epoch_tolerance_)
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



	private:
		double factor_;
		size_t epoch_tolerance_;
		double min_lr_;
		double threshold_;
		double best_loss_;
		size_t bad_epochs_;
	};
}

#endif
