 
#include "OneHot.h"

namespace torch_explorer
{
	torch::Tensor to_one_hot(const torch::Tensor& labels, int64_t num_classes)
	{
		auto one_hot = torch::zeros({ labels.size(0), num_classes }, labels.device());
		return one_hot.scatter_(1, labels.unsqueeze(1), 1);
	}
}

 
