#ifndef IOPTIMIZER_
#define IOPTIMIZER_

#include <torch/torch.h>



namespace torch_explorer
{

	class IOptimizer
	{
	public:

		virtual ~IOptimizer() {}


		virtual void step() = 0;


		virtual void zero_grad() = 0;

		virtual bool epoch_step(float current_loss = 0) = 0;

		virtual torch::optim::Optimizer& current() = 0;


	};
}

#endif