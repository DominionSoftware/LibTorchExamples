#ifndef COSINE_ANNEALING_SCHEDULER_
#define COSINE_ANNEALING_SCHEDULER_
#define _USE_MATH_DEFINES
#include "ISchedulerStep.h"
#include <torch/optim/schedulers/lr_scheduler.h>
#include <cmath>

namespace torch_explorer 
{

class CosineAnnealingScheduler : public torch::optim::LRScheduler, public ISchedulerStep 
{
public:
    CosineAnnealingScheduler(torch::optim::Optimizer& optimizer,
                            size_t T_max,
                            double eta_min = 0,
                            size_t T_mult = 2)
        : torch::optim::LRScheduler(optimizer)
        , T_max_(T_max)
        , eta_min_(eta_min)
        , T_mult_(T_mult)
        , T_cur_(0)
        , cycle_(0)
    {
    }

    void doStep(double /* param */ = 0) override;
   

    std::vector<double> getLearningRates() 
    {
        return get_lrs();
    }

    CosineAnnealingScheduler(const CosineAnnealingScheduler&) = delete;
    CosineAnnealingScheduler& operator=(const CosineAnnealingScheduler&) = delete;

protected:
    std::vector<double> get_lrs() override 
    {
        std::vector<double> new_lrs;
        auto current_lrs = get_current_lrs();


        double factor = 0.5 * (1 + std::cos(M_PI * T_cur_ / T_max_));

        for (double lr : current_lrs) 
        {
            new_lrs.push_back(eta_min_ + (lr - eta_min_) * factor);
        }
        
        return new_lrs;
    }

private:
    size_t T_max_;      // Period before first restart
    double eta_min_;    // Minimum learning rate
    size_t T_mult_;     // Factor to increase T_max after each restart
    size_t T_cur_;      // Current number of steps in cycle
    size_t cycle_;      // Current cycle number
};

} 

#endif