#include "ReduceLROnPlateauScheduler.h"


using namespace torch_explorer;



ReduceLROnPlateauScheduler::ReduceLROnPlateauScheduler(torch::optim::Optimizer& optimizer,
	double factor,
	size_t epoch_tolerance_,
	double min_lr,
	double threshold) : LRScheduler(optimizer),
								factor_(factor),
								epoch_tolerance_(epoch_tolerance_),
								min_lr_(min_lr),
								threshold_(threshold),
								best_loss_(std::numeric_limits<double>::max()),
								bad_epochs_(0)
{
}


void ReduceLROnPlateauScheduler::doStep(double param)
{
	stepWithLoss(param);
}


 
void ReduceLROnPlateauScheduler::stepWithLoss(double loss)
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