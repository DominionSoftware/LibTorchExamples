#include "CosineAnnealingScheduler.h"

 
using namespace torch_explorer;

 

void CosineAnnealingScheduler::doStep(double /* param */)
{
    step();
    T_cur_++;
        
        // Check if we need to restart
   if (T_cur_ >= T_max_)
   {
        cycle_++;
        T_cur_ = 0;
        T_max_ *= T_mult_;  // Increase period for next cycle
   }
}

 




