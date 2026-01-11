#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <random>
#include <chrono>

using namespace std;

// g++ -O3 -std=c++17 cords_phi2_full_sampled.cpp -o cords_phi2_full_sampled

/* =========================
 * Parameters
 * ========================= */
struct Params {
    size_t sample_size = 0; // 0 = full table
    string out_file;        // empty = auto-generate
};

/* =========================
 * CSV parsing
 * ========================= */
static inline vector<string> split_csv_line(const string& line) {
    vector<string> out;
    string cell;
    stringstream ss(line);
    while (getline(ss, cell, ',')) out.push_back(cell);
    return out;
}

/* =========================
 * Reservoir Sampling
 * ========================= */
struct Reservoir {
    vector<vector<string>> rows;
    size_t seen = 0;
    size_t k;
    mt19937_64 rng;

    Reservoir(size_t k_, uint64_t seed = 0xC0FFEEULL)
        : k(k_), rng(seed) {
        if (k > 0) rows.reserve(k);
    }

    void consider(const vector<string>& row) {
        ++seen;
        if (k == 0) {
            rows.push_back(row);
            return;
        }
        if (rows.size() < k) {
            rows.push_back(row);
            return;
        }
        uniform_int_distribution<size_t> dist(1, seen);
        size_t r = dist(rng);
        if (r <= k) rows[r - 1] = row;
    }
};

/* =========================
 * χ² computation
 * ========================= */
static double compute_chi2(
    const vector<vector<size_t>>& nij,
    const vector<size_t>& ni,
    const vector<size_t>& nj,
    size_t n
) {
    double chi2 = 0.0;
    for (size_t i = 0; i < nij.size(); ++i) {
        if (ni[i] == 0) continue;
        for (size_t j = 0; j < nij[i].size(); ++j) {
            if (nj[j] == 0) continue;
            double expected =
                (double)ni[i] * (double)nj[j] / (double)n;
            if (expected <= 0.0) continue;
            double diff = (double)nij[i][j] - expected;
            chi2 += (diff * diff) / expected;
        }
    }
    return chi2;
}

/* =========================
 * φ² for one column pair
 * ========================= */
static double compute_phi2(
    const vector<vector<string>>& rows,
    size_t col_i,
    size_t col_j
) {
    unordered_map<string, size_t> map_i, map_j;
    size_t next_i = 0, next_j = 0;

    for (const auto& r : rows) {
        const string& vi = r[col_i];
        const string& vj = r[col_j];
        if (!map_i.count(vi)) map_i[vi] = next_i++;
        if (!map_j.count(vj)) map_j[vj] = next_j++;
    }

    size_t d1 = map_i.size();
    size_t d2 = map_j.size();
    size_t min_d = min(d1, d2);
    if (min_d <= 1) return 0.0;

    vector<vector<size_t>> nij(d1, vector<size_t>(d2, 0));
    vector<size_t> ni(d1, 0), nj(d2, 0);

    for (const auto& r : rows) {
        size_t i = map_i[r[col_i]];
        size_t j = map_j[r[col_j]];
        nij[i][j]++;
        ni[i]++;
        nj[j]++;
    }

    size_t n = rows.size();
    if (n == 0) return 0.0;

    double chi2 = compute_chi2(nij, ni, nj, n);
    return chi2 / ((double)n * (double)(min_d - 1));
}

/* =========================
 * Utils
 * ========================= */
static string basename_noext(const string& path) {
    size_t pos = path.find_last_of("/\\");
    string name = (pos == string::npos) ? path : path.substr(pos + 1);
    if (name.size() >= 4 && name.substr(name.size() - 4) == ".csv") {
        name = name.substr(0, name.size() - 4);
    }
    return name;
}

/* =========================
 * CLI
 * ========================= */
static bool parse_args(int argc, char** argv, Params& P, string& csv_path) {
    if (argc < 2) return false;
    csv_path = argv[1];

    for (int i = 2; i < argc; ++i) {
        string a = argv[i];
        if (a == "--sample" && i + 1 < argc) {
            P.sample_size = stoull(argv[++i]);
        } else if (a == "--out" && i + 1 < argc) {
            P.out_file = argv[++i];
        } else {
            cerr << "Unknown option: " << a << "\n";
            return false;
        }
    }
    return true;
}


/* =========================
 * Main
 * ========================= */
int main(int argc, char** argv) {
    using Clock = chrono::steady_clock;
    auto t0 = Clock::now();

    Params P;
    string csv_path;
    if (!parse_args(argc, argv, P, csv_path)) {
        cerr << "Usage: " << argv[0]
             << " data.csv [--sample N]\n";
        return 1;
    }

    ifstream fin(csv_path);
    if (!fin.is_open()) {
        cerr << "Cannot open file: " << csv_path << "\n";
        return 1;
    }

    string header;
    getline(fin, header);
    vector<string> col_names = split_csv_line(header);
    size_t ncols = col_names.size();

    Reservoir R(P.sample_size);

    string line;
    while (getline(fin, line)) {
        if (line.empty()) continue;
        vector<string> row = split_csv_line(line);
        if (row.size() == ncols) R.consider(row);
    }
    fin.close();

    if (R.rows.empty()) {
        cerr << "No data rows.\n";
        return 1;
    }

    string out_path;
    if (!P.out_file.empty()) {
        out_path = P.out_file;
    } else {
        string base = basename_noext(csv_path);
        ostringstream oss;
        oss << "correlation_" << base
            << "_n" << P.sample_size << ".txt";
        out_path = oss.str();
    }

    ofstream fout(out_path);

    fout << "column1,column2,correlation\n";
    for (size_t i = 0; i < ncols; ++i) {
        for (size_t j = i + 1; j < ncols; ++j) {
            double phi2 = compute_phi2(R.rows, i, j);
            fout << col_names[i] << ","
                 << col_names[j] << ","
                 << phi2 << "\n";
        }
    }
    fout.close();

    cerr << "[φ²] Output file: " << out_path << "\n";


    auto t1 = Clock::now();
    double sec =
        chrono::duration_cast<chrono::duration<double>>(t1 - t0).count();
    cerr << "[φ²] Total elapsed time: " << sec << " sec\n";
    return 0;
}
