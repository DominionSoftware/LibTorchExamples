#ifndef HEIRARCHICAL_LOSS_
#define HEIRARCHICAL_LOSS_
#include <torch/torch.h>

namespace torch_explorer
{


	torch::Tensor HeirachicalLoss(
		const std::tuple<torch::Tensor, torch::Tensor>& outputs,
		const std::tuple<torch::Tensor, torch::Tensor>& targets,
		float coarse_weight = 0.3);
}

#endif
