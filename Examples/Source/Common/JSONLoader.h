#ifndef JSON_LOADER_
#define JSON_LOADER_

#include <string>
#include <vector>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

namespace torch_explorer {

    class JSONLoader {
    public:
        /**
         * @brief Load a JSON file from one of multiple possible paths.
         *
         * @param paths Vector of possible file paths
         * @param requiredField Optional field that must exist in the JSON
         * @return nlohmann::json The loaded JSON object, or empty JSON if not found
         */
        nlohmann::json loadJSON(const std::vector<std::filesystem::path>& paths,
            const std::string& requiredField = "");

        /**
         * @brief Resolve a variable reference in a JSON configuration.
         *
         * @tparam T The expected type of the reference value
         * @param config The JSON configuration object
         * @param refName The reference name (starting with @)
         * @return T The resolved value, or a default value if not found
         */
        template <typename T>
        T resolveReference(const nlohmann::json& config, const std::string& refName);

        /**
         * @brief Evaluate a simple expression in a JSON configuration.
         *
         * @param config The JSON configuration object
         * @param expr The expression string (starting with $)
         * @return float The evaluated expression result
         */
        float evaluateExpression(const nlohmann::json& config, const std::string& expr);

        /**
         * @brief Get a value from JSON, resolving references and expressions.
         *
         * @tparam T The expected type of the value
         * @param config The JSON configuration object
         * @param value The JSON value (which might be a reference or expression)
         * @return T The resolved value
         */
        template <typename T>
        T getValue(const nlohmann::json& config, const nlohmann::json& value);
    };

    // Template implementation needs to be in the header file
    template <typename T>
    T JSONLoader::resolveReference(const nlohmann::json& config, const std::string& refName) {
        // Remove the @ prefix
        std::string varName = refName.substr(1);

        // Check if the variable exists in the config
        if (config.contains(varName)) {
            return config[varName].get<T>();
        }

        // If not found, throw an exception or return a default value
        std::cerr << "Warning: Reference not found: " << refName << ", using default value." << std::endl;
        if constexpr (std::is_same_v<T, int>) return 0;
        else if constexpr (std::is_same_v<T, float>) return 0.0f;
        else if constexpr (std::is_same_v<T, std::string>) return "";
        else return T{};
    }

    template <typename T>
    T JSONLoader::getValue(const nlohmann::json& config, const nlohmann::json& value) {
        // Check if it's a string that might be a reference or expression
        if (value.is_string()) {
            auto strValue = value.get<std::string>();

            // Handle variable references (@variable)
            if (strValue.size() > 1 && strValue[0] == '@') {
                return resolveReference<T>(config, strValue);
            }
            // Handle expressions ($expression)
            if (strValue.size() > 1 && strValue[0] == '$') {
                if constexpr (std::is_floating_point_v<T> || std::is_same_v<T, float>) {
                    return evaluateExpression(config, strValue);
                }
                else if constexpr (std::is_integral_v<T> || std::is_same_v<T, int>) {
                    return static_cast<T>(evaluateExpression(config, strValue));
                }
                else {
                    std::cerr << "Warning: Expression evaluation to this type not supported, using default" << std::endl;
                    return T{};
                }
            }
        }

        // Direct conversion for non-string or non-reference values
        return value.get<T>();
    }

} // namespace torch_explorer

#endif // JSON_LOADER_H