#ifndef ISCHEDULER_STEP_
#define ISCHEDULER_STEP_




class ISchedulerStep

{
public:

	virtual ~ISchedulerStep() {}


	virtual void doStep(double param = 0) = 0;


};

#endif
