
#include "HeirarchicalLoss.h"
#include <torch/torch.h>


namespace torch_explorer
{


	torch::Tensor HeirachicalLoss(const std::tuple<torch::Tensor, torch::Tensor>& outputs,
								const std::tuple<torch::Tensor, torch::Tensor>& targets,float coarse_weight)
	{
		auto [coarse_out, fine_out] = outputs;
		auto [coarse_target, fine_target] = targets;

		auto coarse_loss = torch::nn::functional::cross_entropy(coarse_out, coarse_target);
		auto fine_loss = torch::nn::functional::cross_entropy(fine_out, fine_target);

		// Weighted combination of both losses
		return coarse_weight * coarse_loss + (1.0 - coarse_weight) * fine_loss;
	}
}