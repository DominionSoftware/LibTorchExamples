#include "CosineAnnealingScheduler.h"

 
using namespace torch_explorer;

 

void CosineAnnealingScheduler::doStep(double /* param */)
{
    T_cur_++;

    // Calculate new learning rate
    double factor = 0.5 * (1 + std::cos(M_PI * T_cur_ / T_max_));
    double new_lr = eta_min_ + (initial_lr_ - eta_min_) * factor;

    // Apply to current optimizer
    for (auto& group : optimizer_.current().param_groups()) {
        group.options().set_lr(new_lr);
    }

    // Check for restart
    if (T_cur_ >= T_max_) {
        cycle_++;
        T_cur_ = 0;
        T_max_ *= T_mult_;
    }
}

 




