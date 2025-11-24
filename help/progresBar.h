#pragma once
#include <iostream>
#include <iomanip>

/**
 * Renders a single-line progress bar.
 * @param current Current progress count.
 * @param total Total count representing 100%.
 * @param width Width of the bar in characters.
 * 
 * @ingroup utilities
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
 * Renders two-level nested progress bars in-place.
 * @param i_current Outer loop current iteration.
 * @param i_total Outer loop total iterations.
 * @param j_current Inner loop current iteration.
 * @param j_total Inner loop total iterations.
 * @param width Width of each bar in characters.
 * @param outer_text Label for the outer bar.
 * @param inner_text Label for the inner bar.
 * 
 * @ingroup utilities
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
