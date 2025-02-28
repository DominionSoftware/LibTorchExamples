
#include "KullbackLieblerDivergenceLoss.h"


namespace torch_explorer
{

	torch::Tensor kl_divergence_loss(const torch::Tensor& pred_logits, const torch::Tensor& target_probs)
	{
		auto log_softmax = torch::log_softmax(pred_logits, 1);
		return mean(sum(target_probs * (torch::log(target_probs + 1e-10) - log_softmax), 1));
	}
}



