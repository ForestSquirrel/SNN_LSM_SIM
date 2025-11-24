#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <cstdint>
#include <numeric>
#include <map>
#include <bit>

/**
 * Swaps byte order for 32-bit integers (big-endian <-> little-endian).
 * @param val Value to swap.
 * @return Reordered integer.
 * 
 * @ingroup datasets
 */
inline uint32_t swap_endian(uint32_t val) {
    return ((val << 24) & 0xFF000000) |
        ((val << 8) & 0x00FF0000) |
        ((val >> 8) & 0x0000FF00) |
        ((val >> 24) & 0x000000FF);
}

/**
 * Minimal MNIST dataset loader supporting sorting and splitting utilities.
 * @tparam T Numeric type used to store pixel values.
 * 
 * @ingroup datasets
 */
template <typename T>
class MNIST {
    static_assert(std::is_arithmetic<T>::value, "Template parameter T must be numeric.");

public:
    static constexpr int IMAGE_DIM = 28;
    static constexpr int PIXEL_COUNT = IMAGE_DIM * IMAGE_DIM;

    enum class PartitionType : uint8_t {
        TEST = 0,
        TRAIN = 1,
        ALL = 2
    };

    enum class SortingType : uint8_t {
        LABELS = 0,
        REPEATING = 1 // cyclic label order: 0,1,...,9,0,1,...
    };

    enum class Labels : uint8_t {
        DIGIT_0 = 0, DIGIT_1 = 1, DIGIT_2 = 2, DIGIT_3 = 3, DIGIT_4 = 4,
        DIGIT_5 = 5, DIGIT_6 = 6, DIGIT_7 = 7, DIGIT_8 = 8, DIGIT_9 = 9
    };

private:
    std::vector<T> m_images;
    std::vector<Labels> m_labels;

    std::string m_train_images_path;
    std::string m_train_labels_path;
    std::string m_test_images_path;
    std::string m_test_labels_path;

    void read_file(const std::string& path, bool is_image) {
        std::ifstream file(path, std::ios::binary);
        std::cout << "Trying to open: [" << path << "] -> "
            << (file.is_open() ? "OK" : "FAILED") << std::endl;
        if (!file.is_open()) {
            std::cerr << "Cannot open file: [" << path << "]\n";
            throw std::runtime_error("Error opening file: " + path);
        }

        uint32_t magic, count;
        file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        file.read(reinterpret_cast<char*>(&count), sizeof(count));
        magic = swap_endian(magic);
        count = swap_endian(count);

        if (is_image) {
            if (magic != 2051) {
                throw std::runtime_error("Image file magic number mismatch: " + std::to_string(magic));
            }

            uint32_t rows, cols;
            file.read(reinterpret_cast<char*>(&rows), sizeof(rows));
            file.read(reinterpret_cast<char*>(&cols), sizeof(cols));
            rows = swap_endian(rows);
            cols = swap_endian(cols);

            if (rows != IMAGE_DIM || cols != IMAGE_DIM)
                throw std::runtime_error("Image dimension mismatch.");

            size_t total_pixels = static_cast<size_t>(count) * PIXEL_COUNT;
            std::vector<uint8_t> buffer(total_pixels);
            file.read(reinterpret_cast<char*>(buffer.data()), buffer.size());

            m_images.reserve(m_images.size() + total_pixels);
            for (auto b : buffer)
                m_images.push_back(static_cast<T>(b));
        }
        else {
            if (magic != 2049) {
                throw std::runtime_error("Label file magic number mismatch: " + std::to_string(magic));
            }

            std::vector<uint8_t> buffer(count);
            file.read(reinterpret_cast<char*>(buffer.data()), buffer.size());

            m_labels.reserve(m_labels.size() + count);
            for (auto b : buffer)
                m_labels.push_back(static_cast<Labels>(b));
        }
    }

    template <typename... Args>
    void get_split_counts(std::map<Labels, int>& counts, Labels label, int n_images, Args... args) const {
        if (counts.count(label))
            throw std::runtime_error("Duplicate label in split arguments.");
        counts[label] = n_images;
        get_split_counts(counts, args...);
    }

    void get_split_counts(std::map<Labels, int>&) const {}

public:
    MNIST(const std::string& train_images, const std::string& train_labels,
        const std::string& test_images, const std::string& test_labels)
        : m_train_images_path(train_images),
        m_train_labels_path(train_labels),
        m_test_images_path(test_images),
        m_test_labels_path(test_labels) {}

    MNIST(const MNIST& other) = default;
    MNIST(MNIST&& other) noexcept = default;

    void load(PartitionType type) {
        destroy();

        if (type == PartitionType::TRAIN || type == PartitionType::ALL) {
            read_file(m_train_images_path, true);
            read_file(m_train_labels_path, false);
        }
        if (type == PartitionType::TEST || type == PartitionType::ALL) {
            read_file(m_test_images_path, true);
            read_file(m_test_labels_path, false);
        }

        if (m_images.size() / PIXEL_COUNT != m_labels.size()) {
            throw std::runtime_error("Loaded image and label counts do not match.");
        }
    }

    void sort(SortingType type) {
        if (m_labels.empty()) return;

        std::vector<size_t> indices(size());
        std::iota(indices.begin(), indices.end(), 0);

        if (type == SortingType::LABELS) {
            std::sort(indices.begin(), indices.end(), [this](size_t i, size_t j) {
                return m_labels[i] < m_labels[j];
                });
        }
        else if (type == SortingType::REPEATING) {
            std::map<Labels, std::vector<size_t>> indices_by_label;
            for (size_t i = 0; i < size(); ++i) {
                indices_by_label[m_labels[i]].push_back(i);
            }

            std::vector<size_t> new_indices;
            new_indices.reserve(size());

            size_t max_size = 0;
            for (const auto& pair : indices_by_label) {
                if (pair.second.size() > max_size) {
                    max_size = pair.second.size();
                }
            }

            for (size_t i = 0; i < max_size; ++i) {
                for (int label_val = 0; label_val <= 9; ++label_val) {
                    Labels lbl = static_cast<Labels>(label_val);
                    if (indices_by_label.count(lbl)) {
                        const std::vector<size_t>& current_indices = indices_by_label.at(lbl);
                        if (i < current_indices.size()) {
                            new_indices.push_back(current_indices[i]);
                        }
                    }
                }
            }
            indices = std::move(new_indices);
        }

        std::vector<T> new_images;
        std::vector<Labels> new_labels;
        new_images.reserve(m_images.size());
        new_labels.reserve(m_labels.size());

        for (size_t i : indices) {
            size_t start_idx = i * PIXEL_COUNT;
            new_images.insert(new_images.end(),
                m_images.begin() + start_idx,
                m_images.begin() + start_idx + PIXEL_COUNT);
            new_labels.push_back(m_labels[i]);
        }

        m_images = std::move(new_images);
        m_labels = std::move(new_labels);
    }

    void destroy() noexcept {
        m_images.clear();
        m_images.shrink_to_fit();
        m_labels.clear();
        m_labels.shrink_to_fit();
    }

    size_t size() const noexcept { return m_labels.size(); }

    struct ImageAccessor {
    private:
        const MNIST* m_mnist;
    public:
        explicit ImageAccessor(const MNIST* mnist) : m_mnist(mnist) {}

        std::vector<T> at(size_t idx) const {
            if (idx >= m_mnist->size())
                throw std::out_of_range("Image index out of range.");
            size_t start = idx * PIXEL_COUNT;
            return std::vector<T>(
                m_mnist->m_images.begin() + start,
                m_mnist->m_images.begin() + start + PIXEL_COUNT);
        }

        std::vector<T> operator[](size_t idx) const { return at(idx); }
    };

    struct LabelAccessor {
    private:
        const MNIST* m_mnist;
    public:
        explicit LabelAccessor(const MNIST* mnist) : m_mnist(mnist) {}

        Labels at(size_t idx) const {
            if (idx >= m_mnist->size())
                throw std::out_of_range("Label index out of range.");
            return m_mnist->m_labels[idx];
        }

        Labels operator[](size_t idx) const { return at(idx); }
    };

    ImageAccessor images() const { return ImageAccessor{ this }; }
    LabelAccessor labels() const { return LabelAccessor{ this }; }

    template <typename... Args>
    MNIST<T> split(Args... args) const {
        if (sizeof...(Args) % 2 != 0)
            throw std::runtime_error("split() requires even number of arguments (label, count pairs).");

        std::map<Labels, int> desired_counts;
        get_split_counts(desired_counts, args...);

        // compute total images needed
        size_t total_needed = 0;
        for (typename std::map<Labels, int>::const_iterator it = desired_counts.begin(); it != desired_counts.end(); ++it)
            total_needed += it->second;

        MNIST<T> new_mnist(m_train_images_path, m_train_labels_path,
            m_test_images_path, m_test_labels_path);

        new_mnist.m_images.reserve(total_needed * PIXEL_COUNT);
        new_mnist.m_labels.reserve(total_needed);

        std::map<Labels, int> current_counts;
        for (typename std::map<Labels, int>::const_iterator it = desired_counts.begin(); it != desired_counts.end(); ++it)
            current_counts[it->first] = 0;

        for (size_t i = 0; i < size(); ++i) {
            Labels lbl = m_labels[i];
            if (desired_counts.count(lbl)) {
                if (current_counts[lbl] < desired_counts[lbl]) {
                    size_t start = i * PIXEL_COUNT;
                    new_mnist.m_images.insert(new_mnist.m_images.end(),
                        m_images.begin() + start,
                        m_images.begin() + start + PIXEL_COUNT);
                    new_mnist.m_labels.push_back(lbl);
                    current_counts[lbl]++;
                }
            }
        }

        new_mnist.m_images.shrink_to_fit();
        new_mnist.m_labels.shrink_to_fit();

        return new_mnist;
    }
};
