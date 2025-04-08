#include "JSONLoader.h"
#include <regex>
#include <stdexcept>

namespace torch_explorer {

    nlohmann::json JSONLoader::loadJSON(const std::vector<std::filesystem::path>& paths,
        const std::string& requiredField) {
        for (const auto& path : paths) {
            if (std::filesystem::exists(path)) {
                try {
                    std::ifstream file(path);
                    nlohmann::json json;
                    file >> json;

                    // Check if the required field exists if specified
                    if (!requiredField.empty() && !json.contains(requiredField)) {
                        std::cerr << "Warning: Required field '" << requiredField
                            << "' not found in " << path.string() << std::endl;
                        continue;
                    }

                    std::cout << "Loaded JSON from: " << path.string() << std::endl;
                    return json;
                }
                catch (const std::exception& e) {
                    std::cerr << "Error parsing " << path.string() << ": " << e.what() << std::endl;
                }
            }
        }

        // If no valid JSON found, return empty JSON object
        return nlohmann::json{};
    }

    float JSONLoader::evaluateExpression(const nlohmann::json& config, const std::string& expr) {
        // Remove the $ prefix
        std::string expression = expr.substr(1);

        // Special case: "1.0 - float(@out_size) / float(@patch_size)"
        std::regex outSizeDivPatchSize(R"(1\.0\s*-\s*float\(@out_size\)\s*/\s*float\(@patch_size\))");
        if (std::regex_match(expression, outSizeDivPatchSize)) {
            int outSize = resolveReference<int>(config, "@out_size");
            int patchSize = resolveReference<int>(config, "@patch_size");
            return 1.0f - static_cast<float>(outSize) / static_cast<float>(patchSize);
        }

        // Handle integer division with // (Python-style)
        std::regex intDivision(R"(\(\(@([a-zA-Z0-9_]+)\s*-\s*@([a-zA-Z0-9_]+)\)\s*\/\/\s*(\d+)\))");
        std::smatch match;
        if (std::regex_search(expression, match, intDivision)) {
            std::string var1 = "@" + match[1].str();
            std::string var2 = "@" + match[2].str();
            int divisor = std::stoi(match[3].str());

            int val1 = resolveReference<int>(config, var1);
            int val2 = resolveReference<int>(config, var2);

            return static_cast<float>((val1 - val2) / divisor);
        }

        // For complex expressions like "((@patch_size - @out_size) // 2,) * 4"
        if (expression.find("@patch_size") != std::string::npos &&
            expression.find("@out_size") != std::string::npos) {
            int patchSize = resolveReference<int>(config, "@patch_size");
            int outSize = resolveReference<int>(config, "@out_size");

            // For the specific expression "((@patch_size - @out_size) // 2,) * 4"
            if (expression.find("// 2") != std::string::npos && expression.find("* 4") != std::string::npos) {
                return ((patchSize - outSize) / 2.0); // The * 4 is handled separately
            }
        }

        // Default case if we can't parse
        std::cerr << "Warning: Unable to evaluate expression: " << expr << ", using 0.0" << std::endl;
        return 0.0f;
    }

} // namespace torch_explorer