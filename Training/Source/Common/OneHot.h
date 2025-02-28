#ifndef ONEHOT_
#define ONEHOT_
#include <torch/torch.h>



namespace torch_explorer
{
	torch::Tensor to_one_hot(const torch::Tensor& labels, int64_t num_classes);
}


#endif
