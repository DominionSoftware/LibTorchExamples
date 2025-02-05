#ifndef CIFAR100CLASS_NAMES_
#define CIFAR100CLASS_NAMES_

#include <filesystem>
#include <fstream>

namespace torch_explorer
{


	class CIFAR100ClassNames
	{
	public:

		static  CIFAR100ClassNames& instance()
		{
			static CIFAR100ClassNames theInstance;

			return theInstance;
		}


		std::string getFineClassName(int index) const
		{
			
			return fine_classes[index];
		}

		// Get superclass name from index
		std::string getSuperclassName(int index) const
		{
			return superclasses[index];
		}

		// Get number of superclasses
		size_t getNumSuperclasses() const
		{
			return superclasses.size();
		}

		// Get number of fine classes
		size_t getNumFineClasses() const
		{
			return FineToCoarse().size();
		}

		std::vector<int64_t> FineToCoarse() const
		{
			return fine_to_coarse_;
		}

		void loadClassNames(const std::filesystem::path& path, const std::string& coarseFileName,const std::string& fineFileName)
		{

			std::filesystem::path localPath = path;
			std::ifstream file;
			localPath.replace_filename(coarseFileName);

			file.open(localPath.string(), std::fstream::in);

			if (!file.is_open())
			{
				throw std::runtime_error("Unable to open coarse file names file.");
			}

			while(!file.eof())
			{
				std::string line;
				file >> line;
				superclasses.push_back(line);
 			}

			file.close();

			localPath.replace_filename(fineFileName);

			file.open(localPath.string(), std::fstream::in);

			if (!file.is_open())
			{
				throw std::runtime_error("Unable to open fine file names file.");
			}

			while (!file.eof())
			{
				std::string line;
				file >> line;
				fine_classes.push_back(line);
			}
			file.close();
		}


	private:
		
		std::vector<std::string> superclasses;
		std::vector<std::string> fine_classes;

		CIFAR100ClassNames()
		{
			fine_to_coarse_.reserve(100);  
			for (int i = 0; i < 20; i++)
			{
				for (int j = 0; j < 5; j++)
				{
					fine_to_coarse_.push_back(i);
				}
			}
		}

		CIFAR100ClassNames(CIFAR100ClassNames& rhs) = delete;
		CIFAR100ClassNames operator = (CIFAR100ClassNames&& rhs) = delete;
		CIFAR100ClassNames operator = (CIFAR100ClassNames& rhs) = delete;
		std::vector<int64_t> fine_to_coarse_;

		
	};
}
#endif
