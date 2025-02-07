#ifndef KULLBACK_LIEBLER_DIVERGENCE_LOSS_

#define KULLBACK_LIEBLER_DIVERGENCE_LOSS_

#include <torch/torch.h>

namespace torch_explorer
{

	torch::Tensor kl_divergence_loss(const torch::Tensor& pred_logits, const torch::Tensor& target_probs);
}

#endif

