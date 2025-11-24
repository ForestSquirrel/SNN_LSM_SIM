#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <numeric>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>

using std::string;
using std::cout;
using std::cerr;
using std::endl;

bool save_device_vector_to_file(std::fstream& fs, const thrust::device_vector<float>& vec) {
    size_t size = vec.size();

    fs.write(reinterpret_cast<const char*>(&size), sizeof(size_t));
    if (!fs.good()) {
        cerr << "Error writing vector size." << endl;
        return false;
    }

    if (size > 0) {
        thrust::host_vector<float> h_vec = vec;

        fs.write(reinterpret_cast<const char*>(h_vec.data()), size * sizeof(float));
        if (!fs.good()) {
            cerr << "Error writing vector data." << endl;
            return false;
        }
    }
    return true;
}

bool load_device_vector_from_file(std::fstream& fs, thrust::device_vector<float>& vec) {
    size_t size = 0;

    fs.read(reinterpret_cast<char*>(&size), sizeof(size_t));
    if (!fs.good() && !fs.eof()) { 
        cerr << "Error reading vector size." << endl;
        return false;
    }

    if (size > 0) {
        thrust::host_vector<float> h_vec(size);
        fs.read(reinterpret_cast<char*>(h_vec.data()), size * sizeof(float));
        if (!fs.good()) {
            cerr << "Error reading vector data." << endl;
            return false;
        }

        vec = h_vec; 
    }
    else {
        vec.clear();
    }
    return true;
}


/**
 * @brief Saves six Thrust device vectors to a binary configuration file.
 *
 * @param fileName The path to the file.
 * @param ILSM_Xpre Input vector 1.
 * @param ILSM_Xpost Input vector 2.
 * @param ILSM_W Input vector 3.
 * @param LSM_Xpre Input vector 4.
 * @param LSM_Xpost Input vector 5.
 * @param LSM_W Input vector 6.
 * @return true if successful, false otherwise.
 */
bool saveLSM(
    string fileName,
    thrust::device_vector<float> ILSM_Xpre,
    thrust::device_vector<float> ILSM_Xpost,
    thrust::device_vector<float> ILSM_W,
    thrust::device_vector<float> LSM_Xpre,
    thrust::device_vector<float> LSM_Xpost,
    thrust::device_vector<float> LSM_W
) {
    if (ILSM_Xpre.empty() || ILSM_Xpost.empty() || ILSM_W.empty() ||
        LSM_Xpre.empty() || LSM_Xpost.empty() || LSM_W.empty()) {
        cerr << "Sanity Check failed: One or more vectors are empty." << endl;
        return false;
    }

    if (!(ILSM_Xpre.size() == ILSM_Xpost.size() && ILSM_Xpost.size() == ILSM_W.size())) {
        cerr << "Sanity Check failed: ILSM vector sizes do not match." << endl;
        return false;
    }

    if (!(LSM_Xpre.size() == LSM_Xpost.size() && LSM_Xpost.size() == LSM_W.size())) {
        cerr << "Sanity Check failed: LSM vector sizes do not match." << endl;
        return false;
    }

    std::fstream fs(fileName, std::ios::out | std::ios::binary | std::ios::trunc);
    if (!fs.is_open()) {
        cerr << "Error: Could not open file for writing: " << fileName << endl;
        return false;
    }

    bool success = true;
    success &= save_device_vector_to_file(fs, ILSM_Xpre);
    success &= save_device_vector_to_file(fs, ILSM_Xpost);
    success &= save_device_vector_to_file(fs, ILSM_W);
    success &= save_device_vector_to_file(fs, LSM_Xpre);
    success &= save_device_vector_to_file(fs, LSM_Xpost);
    success &= save_device_vector_to_file(fs, LSM_W);

    fs.close();
    return success;
}

/**
 * @brief Loads six Thrust device vectors from a binary configuration file.
 *
 * @param fileName The path to the file.
 * @param ILSM_Xpre Output vector 1 (modified by reference).
 * @param ILSM_Xpost Output vector 2 (modified by reference).
 * @param ILSM_W Output vector 3 (modified by reference).
 * @param LSM_Xpre Output vector 4 (modified by reference).
 * @param LSM_Xpost Output vector 5 (modified by reference).
 * @param LSM_W Output vector 6 (modified by reference).
 * @return true if successful, false otherwise.
 */
bool loadLSM(
    string fileName,
    thrust::device_vector<float>& ILSM_Xpre,
    thrust::device_vector<float>& ILSM_Xpost,
    thrust::device_vector<float>& ILSM_W,
    thrust::device_vector<float>& LSM_Xpre,
    thrust::device_vector<float>& LSM_Xpost,
    thrust::device_vector<float>& LSM_W
) {
    std::fstream fs(fileName, std::ios::in | std::ios::binary);
    if (!fs.is_open()) {
        cerr << "Error: Could not open file for reading: " << fileName << endl;
        return false;
    }

    bool success = true;
    success &= load_device_vector_from_file(fs, ILSM_Xpre);
    success &= load_device_vector_from_file(fs, ILSM_Xpost);
    success &= load_device_vector_from_file(fs, ILSM_W);
    success &= load_device_vector_from_file(fs, LSM_Xpre);
    success &= load_device_vector_from_file(fs, LSM_Xpost);
    success &= load_device_vector_from_file(fs, LSM_W);

    fs.close();


    if (!(ILSM_Xpre.size() == ILSM_Xpost.size() && ILSM_Xpost.size() == ILSM_W.size())) {
        cerr << "Sanity Check failed on loaded data: ILSM vector sizes do not match." << endl;
        return false;
    }

    if (!(LSM_Xpre.size() == LSM_Xpost.size() && LSM_Xpost.size() == LSM_W.size())) {
        cerr << "Sanity Check failed on loaded data: LSM vector sizes do not match." << endl;
        return false;
    }
    if (ILSM_Xpre.empty() || LSM_Xpre.empty()) { 
        cerr << "Sanity Check failed on loaded data: Loaded vectors are unexpectedly empty." << endl;
        return false;
    }

    return success;
}

void print_device_vector(const thrust::device_vector<float>& vec, const string& name) {
    if (vec.empty()) {
        cout << name << " (Size: 0) - Empty" << endl;
        return;
    }
    cout << name << " (Size: " << vec.size() << ") - First element: " << vec[0] << ", Last element: " << vec[vec.size() - 1] << endl;
}
