#pragma once
/**
 * @file progresBar.h
 * @brief Console helpers for rendering single and nested progress bars.
 */
#include <iostream>
#include <iomanip>

/**
 * @brief Render a simple progress bar to stdout.
 */
inline void print_progress(int current, int total, int width = 50) {
    static int last_progress_chars = -1;
    static int last_percent = -1;

    float percentage = static_cast<float>(current) / total;
    int progress_chars = static_cast<int>(percentage * width);
    int percent = static_cast<int>(percentage * 100);

    // Only redraw if something changed
    if (progress_chars != last_progress_chars || percent != last_percent) {
        std::cout << "\r[";
        for (int i = 0; i < width; ++i)
            std::cout << (i < progress_chars ? '#' : ' ');
        std::cout << "] " << std::setw(3) << percent << "%";
        std::cout.flush();

        last_progress_chars = progress_chars;
        last_percent = percent;
    }

    // Print newline at the end
    if (current >= total) {
        std::cout << std::endl;
        last_progress_chars = last_percent = -1; // reset static vars
    }
}

/**
 * @brief Display a two-level progress indicator where an outer loop contains
 *        an inner batch loop.
 */
inline void print_nested_progress(
    int i_current, int i_total,
    int j_current, int j_total,
    int width = 30,
    const std::string& outer_text = "Overall",
    const std::string& inner_text = "Batch"
) {
    float i_percentage = static_cast<float>(i_current) / i_total;
    int i_progress_chars = static_cast<int>(i_percentage * width);
    int i_percent = static_cast<int>(i_percentage * 100);

    std::cout << "\r[";
    for (int i = 0; i < width; ++i)
        std::cout << (i < i_progress_chars ? '#' : ' ');
    std::cout << "] " << std::setw(3) << i_percent << "% " << outer_text;

    std::cout << "\033[1A";

    float j_percentage = static_cast<float>(j_current) / j_total;
    int j_progress_chars = static_cast<int>(j_percentage * width);
    int j_percent = static_cast<int>(j_percentage * 100);

    std::cout << "\r[";
    for (int j = 0; j < width; ++j)
        std::cout << (j < j_progress_chars ? '#' : ' ');
    std::cout << "] " << std::setw(3) << j_percent << "% " << inner_text;

    std::cout << "\033[1B";

    std::cout.flush();

    if (i_current >= i_total) {
        std::cout << "\033[1A";
        std::cout << "\r[";
        for (int j = 0; j < width; ++j) std::cout << '#';
        std::cout << "] 100% " << inner_text;

        std::cout << "\033[1B";
        std::cout << "\r[";
        for (int i = 0; i < width; ++i) std::cout << '#';
        std::cout << "] 100% " << outer_text << std::endl;
    }
}
