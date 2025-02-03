#ifndef PROGRESS_BAR_
#define PROGRESS_BAR_


namespace torch_explorer
{


	template<typename T>
	class ProgressBar
	{
	public:



		void progress(T current, T total)
		{
			std::cout << " progress = %" << (static_cast<double>(current) / static_cast<double>(total)) * 100 << std::endl;
		}
	};
}

#endif