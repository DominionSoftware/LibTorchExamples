#ifndef CIFAR100CLASS_NAMES_
#define CIRAR100CLASS_NAMES_
namespace torch_explorer
{


	struct CIFAR100ClassNames
	{
		const std::vector<std::string> superclasses = {
			"aquatic_mammals", "fish", "flowers", "food_containers",
			"fruit_and_vegetables", "household_electrical_devices",
			"household_furniture", "insects", "large_carnivores",
			"large_man-made_outdoor_things", "large_natural_outdoor_scenes",
			"large_omnivores_and_herbivores", "medium-sized_mammals",
			"non-insect_invertebrates", "people", "reptiles",
			"small_mammals", "trees", "vehicles_1", "vehicles_2"
		};


		const std::vector<std::vector<std::string>> fine_classes = {
			// 0: aquatic mammals
			{"beaver", "dolphin", "otter", "seal", "whale"},
			// 1: fish
			{"aquarium_fish", "flatfish", "ray", "shark", "trout"},
			// 2: flowers
			{"orchids", "poppies", "roses", "sunflowers", "tulips"},
			// 3: food containers
			{"bottles", "bowls", "cans", "cups", "plates"},
			// 4: fruit and vegetables
			{"apples", "mushrooms", "oranges", "pears", "sweet_peppers"},
			// 5: household electrical devices
			{"clock", "computer_keyboard", "lamp", "telephone", "television"},
			// 6: household furniture
			{"bed", "chair", "couch", "table", "wardrobe"},
			// 7: insects
			{"bee", "beetle", "butterfly", "caterpillar", "cockroach"},
			// 8: large carnivores
			{"bear", "leopard", "lion", "tiger", "wolf"},
			// 9: large man-made outdoor things
			{"bridge", "castle", "house", "road", "skyscraper"},
			// 10: large natural outdoor scenes
			{"cloud", "forest", "mountain", "plain", "sea"},
			// 11: large omnivores and herbivores
			{"camel", "cattle", "chimpanzee", "elephant", "kangaroo"},
			// 12: medium-sized mammals
			{"fox", "porcupine", "possum", "raccoon", "skunk"},
			// 13: non-insect invertebrates
			{"crab", "lobster", "snail", "spider", "worm"},
			// 14: people
			{"baby", "boy", "girl", "man", "woman"},
			// 15: reptiles
			{"crocodile", "dinosaur", "lizard", "snake", "turtle"},
			// 16: small mammals
			{"hamster", "mouse", "rabbit", "shrew", "squirrel"},
			// 17: trees
			{"maple", "oak", "palm", "pine", "willow"},
			// 18: vehicles 1
			{"bicycle", "bus", "motorcycle", "pickup truck", "train"},
			// 19: vehicles 2
			{"lawn-mower", "rocket", "streetcar", "tank", "tractor"}
		};


		static const std::vector<int64_t>& FineToCoarse()
		{
			static const std::vector<int64_t> mapping = []()
				{
				std::vector<int64_t> m;
				m.reserve(100);  // We know the size
				for (int i = 0; i < 20; i++) 
				{
					for (int j = 0; j < 5; j++) 
					{
						m.push_back(i);
					}
				}
				return m;
				}();
			return mapping;
		}


		std::string getFineClassName(int index) const
		{
			int superclass_idx = FineToCoarse()[index];
			int local_idx = index % 5;
			return fine_classes[superclass_idx][local_idx];
		}

		// Get superclass name from index
		std::string getSuperclassName(int index) const
		{
			return superclasses[index];
		}

		// Get number of superclasses
		size_t getNumSuperclasses() const { return superclasses.size(); }

		// Get number of fine classes
		size_t getNumFineClasses() const { return FineToCoarse().size(); }
	};
}
#endif
