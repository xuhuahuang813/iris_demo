#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <chrono>
#include <iomanip>


using namespace std;

/* =========================
 * Parameters (CLI overridable)
 * ========================= */
struct Params {
    size_t sample_size = 8000;     // typical cords sample size (few thousand) 
    size_t dmax = 50;              // d_max (paper mentions 50 used in implementation) 
    size_t mcv_k = 100;             // emulate catalog MCV list size
    double skew_cover = 0.90;      // (1 - ε4) coverage threshold in Step 4 
    bool output_header = true;     // print "列1 列2 关联程度"
};

/* =========================
 * Utility: CSV parsing (simple, no quoted commas)
 * ========================= */
static inline vector<string> split_csv_line(const string& line) {
    vector<string> out;
    out.reserve(64);
    string cell;
    stringstream ss(line);
    while (getline(ss, cell, ',')) out.push_back(cell);
    return out;
}

static inline size_t hashValue(const string& v) {
    return std::hash<string>{}(v);
}

/* =========================
 * Reservoir Sampling over rows
 * ========================= */
struct Reservoir {
    vector<vector<string>> rows;
    size_t seen = 0;
    size_t k;
    std::mt19937_64 rng;

    Reservoir(size_t k_, uint64_t seed = 0xC0FFEEULL) : k(k_), rng(seed) {
        rows.reserve(k);
    }

    void consider(const vector<string>& row) {
        ++seen;
        if (rows.size() < k) {
            rows.push_back(row);
            return;
        }
        std::uniform_int_distribution<size_t> dist(1, seen);
        size_t r = dist(rng);
        if (r <= k) {
            rows[r - 1] = row;
        }
    }
};

/* =========================
 * ColumnStats computed from sample (since no DB catalog available)
 * In a DBMS, |Ci|_R and MCVs come from catalog. Here we approximate from sample.
 * ========================= */
struct ColumnStats {
    size_t distinct_R = 0;                 // approximated by sample distinct
    size_t table_rows_R = 0;               // total rows in CSV file (counted in streaming)
    vector<string> frequent_values;        // top-K values
    vector<size_t> freq_counts;            // their frequencies in sample
};

static ColumnStats computeColumnStatsFromSample(
    const vector<vector<string>>& sample_rows,
    size_t col_idx,
    size_t total_rows_in_file,
    const Params& P
) {
    unordered_map<string, size_t> freq;
    freq.reserve(sample_rows.size() * 2);

    for (const auto& r : sample_rows) {
        if (col_idx < r.size()) freq[r[col_idx]]++;
    }

    vector<pair<string, size_t>> items;
    items.reserve(freq.size());
    for (auto& kv : freq) items.push_back(kv);

    sort(items.begin(), items.end(),
         [](const auto& a, const auto& b) { return a.second > b.second; });

    ColumnStats st;
    st.table_rows_R = total_rows_in_file;
    st.distinct_R = freq.size();

    // take top mcv_k
    size_t take = min(P.mcv_k, items.size());
    st.frequent_values.reserve(take);
    st.freq_counts.reserve(take);
    for (size_t i = 0; i < take; ++i) {
        st.frequent_values.push_back(items[i].first);
        st.freq_counts.push_back(items[i].second);
    }

    return st;
}

/* =========================
 * Category function (CORDS Fig.3 semantics)
 * If skew = true: only keep MCV categories, others filtered out (return d)
 * Else: hashing into d buckets.
 * 
 * ========================= */
static inline size_t Category(
    const string& x,
    const vector<string>& frequentValues,
    size_t d,
    bool skew
) {
    if (skew) {
        for (size_t i = 0; i < frequentValues.size(); ++i) {
            if (x == frequentValues[i]) return i;
        }
        return d; // filtered out (not in MCV list)
    } else {
        return hashValue(x) % d;
    }
}

/* =========================
 * Correct Chi-square (paper Eq. (1))
 * chi2 = Σ (nij - Eij)^2 / Eij,  Eij = ni. * n.j / n
 * 
 * ========================= */
static double computeChiSquare(
    const vector<vector<size_t>>& nij,
    const vector<size_t>& ni_dot,
    const vector<size_t>& n_dot_j,
    size_t total_n
) {
    if (total_n == 0) return 0.0;

    double chi2 = 0.0;
    size_t d1 = nij.size();
    size_t d2 = nij[0].size();

    for (size_t i = 0; i < d1; ++i) {
        if (ni_dot[i] == 0) continue;
        for (size_t j = 0; j < d2; ++j) {
            if (n_dot_j[j] == 0) continue;
            double expected = (double)ni_dot[i] * (double)n_dot_j[j] / (double)total_n;
            if (expected <= 0.0) continue;
            double diff = (double)nij[i][j] - expected;
            chi2 += (diff * diff) / expected;
        }
    }
    return chi2;
}

/* =========================
 * Compute φ^2 = chi2 / (n * (min(d1,d2)-1))
 * with n = effective_n (rows that entered contingency table).
 * 
 * ========================= */
static double computePhi2ForPair(
    const vector<vector<string>>& sample_rows,
    size_t col_i,
    size_t col_j,
    const ColumnStats& Ci,
    const ColumnStats& Cj,
    const Params& P
) {
    // Determine skew based on whether top MCVs cover enough mass (approximated on sample).
    auto cover_ratio = [](const vector<size_t>& cnts, size_t sample_n) -> double {
        size_t s = 0;
        for (auto c : cnts) s += c;
        return sample_n == 0 ? 0.0 : (double)s / (double)sample_n;
    };

    size_t sample_n_raw = sample_rows.size();

    bool skew_i = cover_ratio(Ci.freq_counts, sample_n_raw) >= P.skew_cover;
    bool skew_j = cover_ratio(Cj.freq_counts, sample_n_raw) >= P.skew_cover;

    size_t d1 = skew_i ? Ci.frequent_values.size() : min(Ci.distinct_R, P.dmax);
    size_t d2 = skew_j ? Cj.frequent_values.size() : min(Cj.distinct_R, P.dmax);

    if (d1 == 0 || d2 == 0) return 0.0;
    size_t min_d = min(d1, d2);
    if (min_d <= 1) return 0.0;

    vector<vector<size_t>> nij(d1, vector<size_t>(d2, 0));
    vector<size_t> ni_dot(d1, 0), n_dot_j(d2, 0);
    size_t effective_n = 0;

    for (const auto& r : sample_rows) {
        if (col_i >= r.size() || col_j >= r.size()) continue;

        const string& xi = r[col_i];
        const string& xj = r[col_j];

        size_t bi = Category(xi, Ci.frequent_values, d1, skew_i);
        size_t bj = Category(xj, Cj.frequent_values, d2, skew_j);

        // If skew mode filters out non-MCV values, only count those that remain in table
        if (bi < d1 && bj < d2) {
            nij[bi][bj]++;
            ni_dot[bi]++;
            n_dot_j[bj]++;
            effective_n++;
        }
    }

    if (effective_n == 0) return 0.0;

    double chi2 = computeChiSquare(nij, ni_dot, n_dot_j, effective_n);

    // φ^2 as defined in cords: chi2 / (n * (d - 1)), d = min(d1,d2)
    double phi2 = chi2 / ((double)effective_n * (double)(min_d - 1));
    return phi2;
}

/* =========================
 * CLI
 * ========================= */
static void print_usage(const char* prog) {
    cerr << "Usage: " << prog << " data.csv [options]\n"
         << "Options:\n"
         << "  --sample N        reservoir sample size (default 4000)\n"
         << "  --dmax D          d_max buckets (default 50)\n"
         << "  --mcv K           top-K frequent values for skew mode (default 30)\n"
         << "  --skew_cover X    coverage threshold in [0,1] (default 0.90)\n"
         << "  --no_header       do not output header line\n";
}

static bool parse_args(int argc, char** argv, Params& P, string& csv_path) {
    if (argc < 2) return false;
    csv_path = argv[1];

    for (int i = 2; i < argc; ++i) {
        string a = argv[i];
        if (a == "--sample" && i + 1 < argc) P.sample_size = (size_t)stoull(argv[++i]);
        else if (a == "--dmax" && i + 1 < argc) P.dmax = (size_t)stoull(argv[++i]);
        else if (a == "--mcv" && i + 1 < argc) P.mcv_k = (size_t)stoull(argv[++i]);
        else if (a == "--skew_cover" && i + 1 < argc) P.skew_cover = stod(argv[++i]);
        else if (a == "--no_header") P.output_header = false;
        else {
            cerr << "Unknown option: " << a << "\n";
            return false;
        }
    }

    if (P.sample_size == 0) {
        cerr << "--sample must be > 0\n";
        return false;
    }
    if (P.dmax == 0) {
        cerr << "--dmax must be > 0\n";
        return false;
    }
    if (P.skew_cover < 0.0 || P.skew_cover > 1.0) {
        cerr << "--skew_cover must be in [0,1]\n";
        return false;
    }
    return true;
}

static string basename_noext(const string& path) {
    // 去掉目录
    size_t pos = path.find_last_of("/\\");
    string name = (pos == string::npos) ? path : path.substr(pos + 1);

    // 去掉 .csv 后缀（如果有）
    if (name.size() >= 4 && name.substr(name.size() - 4) == ".csv") {
        name = name.substr(0, name.size() - 4);
    }
    return name;
}

static string format_double(double x) {
    ostringstream oss;
    oss << fixed << setprecision(2) << x;
    return oss.str();
}

/* =========================
 * Main
 * ========================= */
int main(int argc, char** argv) {
    using Clock = std::chrono::steady_clock;
    auto t_start = Clock::now();

    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    Params P;
    string csv_path;
    if (!parse_args(argc, argv, P, csv_path)) {
        print_usage(argv[0]);
        return 1;
    }

    string csv_base = basename_noext(csv_path);

    ostringstream fname;
    fname << "correlation_"
        << csv_base
        << "_n" << P.sample_size
        << "_d" << P.dmax
        << "_mcv" << P.mcv_k
        << "_sk" << format_double(P.skew_cover)
        << ".txt";

    string output_path = fname.str();


    ifstream fin(csv_path);
    if (!fin.is_open()) {
        cerr << "Cannot open file: " << csv_path << "\n";
        return 1;
    }

    string header;
    if (!getline(fin, header)) {
        cerr << "Empty CSV\n";
        return 1;
    }
    vector<string> colNames = split_csv_line(header);
    size_t nCols = colNames.size();
    if (nCols < 2) {
        cerr << "Need at least 2 columns\n";
        return 1;
    }

    Reservoir R(P.sample_size);
    string line;
    size_t total_rows = 0;

    while (getline(fin, line)) {
        if (line.empty()) continue;
        vector<string> row = split_csv_line(line);
        if (row.size() != nCols) continue; // skip malformed lines
        total_rows++;
        R.consider(row);
    }
    fin.close();

    if (R.rows.empty()) {
        cerr << "No valid rows read.\n";
        return 1;
    }

    // Compute per-column stats from sample
    vector<ColumnStats> stats;
    stats.reserve(nCols);
    for (size_t c = 0; c < nCols; ++c) {
        stats.push_back(computeColumnStatsFromSample(R.rows, c, total_rows, P));
    }

    // Output
    ofstream fout(output_path);
    if (!fout.is_open()) {
        cerr << "Cannot write "<< output_path << "\n";;
        return 1;
    }

    if (P.output_header) {
        fout << "column1,column2,correlation\n";
    }

    for (size_t i = 0; i < nCols; ++i) {
        for (size_t j = i + 1; j < nCols; ++j) {
            double phi2 = computePhi2ForPair(R.rows, i, j, stats[i], stats[j], P);
            fout << colNames[i] << "," << colNames[j] << "," << phi2 << "\n";
        }
    }
    fout.close();
    cerr << "[CORDS] Output file: " << output_path << "\n";

    auto t_end = Clock::now();
    double elapsed_sec =
        std::chrono::duration_cast<std::chrono::duration<double>>(t_end - t_start).count();

    std::cerr << "[CORDS] Total elapsed time: "
              << elapsed_sec << " seconds\n";
    return 0;
}
