//
// Created by xhhuang on 25-2-20.
//
#include <bits/stdc++.h>
#include <omp.h>  // 确保编译器支持，并使用 -fopenmp 选项编译
using namespace std;

// ---------------------- 全局或工具部分 ----------------------
static vector<int> primes{307, 311, 313, 317, 331, 337, 347, 349,
                          353, 359, 367, 373, 379, 383, 389, 397,
                          401, 409, 419, 421, 431};
static int nbuckets = 16;
static float keyratio_1d = 0.85;
static float keyratio_2d = 0.75;

// choice 结构：为演示方便，这里假设它可以容纳至多 5 个列的 dv
struct choice {
    vector<int> id;   // 存储列下标
    int dvs;          // 不同组合值的数量
    string ch;        // 存储 AVI / Hist / Key 等
    double corr;      // 相关度量值
    vector<int> dv;   // 存储每个列的离散取值数

    choice() : dvs(0), ch(""), corr(0.0) {}
    choice(const vector<int>& _id, int _dvs, const string& _ch,
           double _corr, const vector<int>& _dv)
            : id(_id), dvs(_dvs), ch(_ch), corr(_corr), dv(_dv) {}
};

// ---------------------- 2D 计算函数 ----------------------
void cords2d(vector<vector<string>>& tab,
             vector<choice>& ch,
             int n_rows,
             map<int, map<string, float>>& p,
             map<int, vector<string>>& d)
{
    // 让外层循环并行
#pragma omp parallel for
    for (int t = 0; t < (int)ch.size(); t++)
    {
        int i = ch[t].id[0];
        int j = ch[t].id[1];
        double phi = 0;
        auto di = d[i];
        auto dj = d[j];
        auto pi = p[i];
        auto pj = p[j];
        int sdi = (int)di.size(), sdj = (int)dj.size();

        // 计算所有 (i, j) 取值组合出现情况
        // 这里哈希并去重
        vector<size_t> dvs;  // 哈希值
        dvs.reserve(n_rows);
        for (int r = 0; r < n_rows; r++) {
            // 简单哈希
            size_t hv = std::hash<string>()(tab[r][i]) * primes[0]
                        + std::hash<string>()(tab[r][j]) * primes[1];
            dvs.push_back(hv);
        }
        sort(dvs.begin(), dvs.end());
        dvs.erase(unique(dvs.begin(), dvs.end()), dvs.end());

        // prune
        if ((sdi >= n_rows * keyratio_1d || sdj >= n_rows * keyratio_1d) ||
            (sdi >= n_rows * keyratio_2d &&
             sdj >= n_rows * keyratio_2d &&
             (int)dvs.size() >= n_rows * keyratio_2d))
        {
            ch[t] = choice({i, j}, dvs.size(), "AVI", -1.0, {sdi, sdj});
            continue;
        }

        // 是否倾向于 Hist
        bool ishist = (max(sdi / sdj, sdj / sdi) >= 128);

        // 初始化
        ch[t] = choice({i, j}, dvs.size(), "-", phi, {sdi, sdj});

        // bucketing：如果组合值过大，进行分桶（与 2D 的 nbuckets*nbuckets 比较）
        if ((int)dvs.size() > nbuckets * nbuckets)
        {
            // 对第 i 列分桶
            {
                vector<string> dtmp;
                map<string, float> ptmp;
                dtmp.reserve(nbuckets + 1); // 多加一个边界
                for (int n = 0; n < nbuckets; n++)
                {
                    int startPos = n * sdi / nbuckets;
                    int endPos   = (n + 1) * sdi / nbuckets;
                    dtmp.push_back(di[startPos]);
                    float sumProb = 0.0;
                    for (int t2 = startPos; t2 < endPos; t2++) {
                        sumProb += pi[di[t2]];
                    }
                    ptmp[di[startPos]] = sumProb;
                }
                // 末尾
                dtmp.push_back(di[sdi - 1]);
                di = dtmp;
                pi = ptmp;
            }
            // 对第 j 列分桶
            {
                vector<string> dtmp;
                map<string, float> ptmp;
                dtmp.reserve(nbuckets + 1); // 多加一个边界
                for (int n = 0; n < nbuckets; n++)
                {
                    int startPos = n * sdj / nbuckets;
                    int endPos   = (n + 1) * sdj / nbuckets;
                    dtmp.push_back(dj[startPos]);
                    float sumProb = 0.0;
                    for (int t2 = startPos; t2 < endPos; t2++) {
                        sumProb += pj[dj[t2]];
                    }
                    ptmp[dj[startPos]] = sumProb;
                }
                dtmp.push_back(dj[sdj - 1]);
                dj = dtmp;
                pj = ptmp;
            }
        }

        // 重新获取可能更新后的大小
        sdi = (int)di.size();
        sdj = (int)dj.size();

        // 计算 phi (类似卡方或互信息的度量)
        // 双重循环，落在每个 bucket 范围内的条目数统计
        for (int ni = 0; ni < sdi - 1; ni++)
        {
            for (int nj = 0; nj < sdj - 1; nj++)
            {
                float pij = 0.0;
                // 直接遍历 n_rows 统计
                for (int r = 0; r < n_rows; r++)
                {
                    if (tab[r][i] >= di[ni] && tab[r][i] < di[ni + 1] &&
                        tab[r][j] >= dj[nj] && tab[r][j] < dj[nj + 1])
                    {
                        pij += 1.0;
                    }
                }
                pij /= (float)n_rows;

                float expected = pi[di[ni]] * pj[dj[nj]];
                if (expected > 1e-12) {
                    phi += pow(pij - expected, 2.0f) / expected;
                }
            }
        }
        phi /= (float)(min(sdi, sdj) - 1);

        ch[t].corr = phi;
        if (ishist) {
            ch[t].ch = "Hist";
        }
        if (phi < 0.001) {
            ch[t].ch = "AVI";
        }
    }
}

// ---------------------- 3D 计算函数 ----------------------
void cords3d(vector<vector<string>>& tab,
             vector<choice>& ch3, // 存储三列组合的结果
             int n_rows,
             map<int, map<string, float>>& p,
             map<int, vector<string>>& d)
{
#pragma omp parallel for
    for (int t = 0; t < (int)ch3.size(); t++)
    {
        int i = ch3[t].id[0];
        int j = ch3[t].id[1];
        int k = ch3[t].id[2];
        double phi = 0;
        auto di = d[i];
        auto dj = d[j];
        auto dk = d[k];

        auto pi = p[i];
        auto pj = p[j];
        auto pk = p[k];

        int sdi = (int)di.size();
        int sdj = (int)dj.size();
        int sdk = (int)dk.size();

        // 计算 (i,j,k) 组合出现情况
        vector<size_t> dvs;
        dvs.reserve(n_rows);
        for (int r = 0; r < n_rows; r++)
        {
            // 简单拼接哈希
            size_t hv =
                    std::hash<string>()(tab[r][i]) * primes[0] +
                    std::hash<string>()(tab[r][j]) * primes[1] +
                    std::hash<string>()(tab[r][k]) * primes[2];
            dvs.push_back(hv);
        }
        sort(dvs.begin(), dvs.end());
        dvs.erase(unique(dvs.begin(), dvs.end()), dvs.end());

        // prune 类似处理
        // 这里可以根据需要扩展规则，这里仅演示延用 2D prune 的思路
        if ((sdi >= n_rows * keyratio_1d ||
             sdj >= n_rows * keyratio_1d ||
             sdk >= n_rows * keyratio_1d) ||
            ((sdi >= n_rows * keyratio_2d) &&
             (sdj >= n_rows * keyratio_2d) &&
             (sdk >= n_rows * keyratio_2d) &&
             ((int)dvs.size() >= n_rows * keyratio_2d)))
        {
            ch3[t] = choice({i, j, k}, dvs.size(), "AVI", -1.0, {sdi, sdj, sdk});
            continue;
        }

        // 这里不分 Sparse / Hist，直接与 2D 同理
        ch3[t] = choice({i, j, k}, dvs.size(), "-", phi, {sdi, sdj, sdk});

        // bucketing 判断：若组合值过大，比如 > nbuckets^3，就分桶
        if ((int)dvs.size() > nbuckets * nbuckets * nbuckets)
        {
            // 分桶逻辑（演示与 cords2d 相同，实际上可以更灵活）
            // 对 i 列
            {
                vector<string> dtmp;
                map<string, float> ptmp;
                dtmp.reserve(nbuckets + 1);
                for (int n = 0; n < nbuckets; n++)
                {
                    int startPos = n * sdi / nbuckets;
                    int endPos   = (n + 1) * sdi / nbuckets;
                    dtmp.push_back(di[startPos]);
                    float sumProb = 0.0;
                    for (int t2 = startPos; t2 < endPos; t2++) {
                        sumProb += pi[di[t2]];
                    }
                    ptmp[di[startPos]] = sumProb;
                }
                dtmp.push_back(di[sdi - 1]);
                di = dtmp;
                pi = ptmp;
            }
            // j
            {
                vector<string> dtmp;
                map<string, float> ptmp;
                dtmp.reserve(nbuckets + 1);
                for (int n = 0; n < nbuckets; n++)
                {
                    int startPos = n * sdj / nbuckets;
                    int endPos   = (n + 1) * sdj / nbuckets;
                    dtmp.push_back(dj[startPos]);
                    float sumProb = 0.0;
                    for (int t2 = startPos; t2 < endPos; t2++) {
                        sumProb += pj[dj[t2]];
                    }
                    ptmp[dj[startPos]] = sumProb;
                }
                dtmp.push_back(dj[sdj - 1]);
                dj = dtmp;
                pj = ptmp;
            }
            // k
            {
                vector<string> dtmp;
                map<string, float> ptmp;
                dtmp.reserve(nbuckets + 1);
                for (int n = 0; n < nbuckets; n++)
                {
                    int startPos = n * sdk / nbuckets;
                    int endPos   = (n + 1) * sdk / nbuckets;
                    dtmp.push_back(dk[startPos]);
                    float sumProb = 0.0;
                    for (int t2 = startPos; t2 < endPos; t2++) {
                        sumProb += pk[dk[t2]];
                    }
                    ptmp[dk[startPos]] = sumProb;
                }
                dtmp.push_back(dk[sdk - 1]);
                dk = dtmp;
                pk = ptmp;
            }
        }

        // 重新获取大小
        sdi = (int)di.size();
        sdj = (int)dj.size();
        sdk = (int)dk.size();

        // 计算 phi: 三重循环
        for (int ni = 0; ni < sdi - 1; ni++)
        {
            for (int nj = 0; nj < sdj - 1; nj++)
            {
                for (int nk = 0; nk < sdk - 1; nk++)
                {
                    float pijk = 0.0;
                    for (int r = 0; r < n_rows; r++)
                    {
                        bool inBucket =
                                (tab[r][i] >= di[ni] && tab[r][i] < di[ni + 1]) &&
                                (tab[r][j] >= dj[nj] && tab[r][j] < dj[nj + 1]) &&
                                (tab[r][k] >= dk[nk] && tab[r][k] < dk[nk + 1]);
                        if (inBucket) {
                            pijk += 1.0f;
                        }
                    }
                    pijk /= (float)n_rows;

                    float expected = pi[di[ni]] * pj[dj[nj]] * pk[dk[nk]];
                    if (expected > 1e-12) {
                        phi += pow(pijk - expected, 2.0f) / expected;
                    }
                }
            }
        }

        // 这里的归一化可以根据需求调整，这里演示与 2D 类似
        phi /= (float)(min({sdi, sdj, sdk}) - 1);

        ch3[t].corr = phi;
        // 如果 phi < 0.001，就标记为 AVI
        if (phi < 0.001) {
            ch3[t].ch = "AVI";
        }
    }
}

// ---------------------- 4D 计算函数 ----------------------
void cords4d(vector<vector<string>>& tab,
             vector<choice>& ch4,
             int n_rows,
             map<int, map<string, float>>& p,
             map<int, vector<string>>& d)
{
#pragma omp parallel for
    for (int t = 0; t < (int)ch4.size(); t++)
    {
        int i = ch4[t].id[0];
        int j = ch4[t].id[1];
        int k = ch4[t].id[2];
        int m = ch4[t].id[3];
        double phi = 0.0;

        auto di = d[i];
        auto dj = d[j];
        auto dk = d[k];
        auto dm = d[m];

        auto pi = p[i];
        auto pj = p[j];
        auto pk = p[k];
        auto pm = p[m];

        int sdi = (int)di.size();
        int sdj = (int)dj.size();
        int sdk = (int)dk.size();
        int sdm = (int)dm.size();

        // hash 组合
        vector<size_t> dvs;
        dvs.reserve(n_rows);
        for (int r = 0; r < n_rows; r++)
        {
            size_t hv =
                    std::hash<string>()(tab[r][i]) * primes[0] +
                    std::hash<string>()(tab[r][j]) * primes[1] +
                    std::hash<string>()(tab[r][k]) * primes[2] +
                    std::hash<string>()(tab[r][m]) * primes[3];
            dvs.push_back(hv);
        }
        sort(dvs.begin(), dvs.end());
        dvs.erase(unique(dvs.begin(), dvs.end()), dvs.end());

        // prune
        if ((sdi >= n_rows * keyratio_1d ||
             sdj >= n_rows * keyratio_1d ||
             sdk >= n_rows * keyratio_1d ||
             sdm >= n_rows * keyratio_1d) ||
            (sdi >= n_rows * keyratio_2d &&
             sdj >= n_rows * keyratio_2d &&
             sdk >= n_rows * keyratio_2d &&
             sdm >= n_rows * keyratio_2d &&
             (int)dvs.size() >= n_rows * keyratio_2d))
        {
            ch4[t] = choice({i, j, k, m}, dvs.size(), "AVI", -1.0, {sdi, sdj, sdk, sdm});
            continue;
        }

        ch4[t] = choice({i, j, k, m}, (int)dvs.size(), "-", phi, {sdi, sdj, sdk, sdm});

        // bucketing 如果太大，做分桶（示例简单处理）
        if ((int)dvs.size() > nbuckets * nbuckets * nbuckets * nbuckets)
        {
            // 对 i/j/k/m 分桶，这里省略，和上面类似
            // ...
        }

        // 重新获取大小(如果分桶的话)
        sdi = (int)di.size();
        sdj = (int)dj.size();
        sdk = (int)dk.size();
        sdm = (int)dm.size();

        // 4 重循环
        for (int ni = 0; ni < sdi - 1; ni++)
        {
            for (int nj = 0; nj < sdj - 1; nj++)
            {
                for (int nk = 0; nk < sdk - 1; nk++)
                {
                    for (int nm = 0; nm < sdm - 1; nm++)
                    {
                        float pijkm = 0.0;
                        for (int r = 0; r < n_rows; r++)
                        {
                            bool inBucket =
                                    (tab[r][i] >= di[ni] && tab[r][i] < di[ni + 1]) &&
                                    (tab[r][j] >= dj[nj] && tab[r][j] < dj[nj + 1]) &&
                                    (tab[r][k] >= dk[nk] && tab[r][k] < dk[nk + 1]) &&
                                    (tab[r][m] >= dm[nm] && tab[r][m] < dm[nm + 1]);
                            if (inBucket) {
                                pijkm += 1.0f;
                            }
                        }
                        pijkm /= (float)n_rows;
                        float expected = pi[di[ni]] * pj[dj[nj]] *
                                         pk[dk[nk]] * pm[dm[nm]];
                        if (expected > 1e-12) {
                            phi += pow(pijkm - expected, 2.0f) / expected;
                        }
                    }
                }
            }
        }
        phi /= (float)(min({sdi, sdj, sdk, sdm}) - 1);
        ch4[t].corr = phi;

        if (phi < 0.001) {
            ch4[t].ch = "AVI";
        }
    }
}

// ---------------------- 5D 计算函数 ----------------------
void cords5d(vector<vector<string>>& tab,
             vector<choice>& ch5,
             int n_rows,
             map<int, map<string, float>>& p,
             map<int, vector<string>>& d)
{
#pragma omp parallel for
    for (int t = 0; t < (int)ch5.size(); t++)
    {
        int i = ch5[t].id[0];
        int j = ch5[t].id[1];
        int k = ch5[t].id[2];
        int m = ch5[t].id[3];
        int n = ch5[t].id[4];
        double phi = 0.0;

        auto di = d[i];
        auto dj = d[j];
        auto dk = d[k];
        auto dm = d[m];
        auto dn = d[n];

        auto pi = p[i];
        auto pj = p[j];
        auto pk = p[k];
        auto pm = p[m];
        auto pn = p[n];

        int sdi = (int)di.size();
        int sdj = (int)dj.size();
        int sdk = (int)dk.size();
        int sdm = (int)dm.size();
        int sdn = (int)dn.size();

        // hash
        vector<size_t> dvs;
        dvs.reserve(n_rows);
        for (int r = 0; r < n_rows; r++)
        {
            size_t hv =
                    std::hash<string>()(tab[r][i]) * primes[0] +
                    std::hash<string>()(tab[r][j]) * primes[1] +
                    std::hash<string>()(tab[r][k]) * primes[2] +
                    std::hash<string>()(tab[r][m]) * primes[3] +
                    std::hash<string>()(tab[r][n]) * primes[4];
            dvs.push_back(hv);
        }
        sort(dvs.begin(), dvs.end());
        dvs.erase(unique(dvs.begin(), dvs.end()), dvs.end());

        // prune
        if ((sdi >= n_rows * keyratio_1d ||
             sdj >= n_rows * keyratio_1d ||
             sdk >= n_rows * keyratio_1d ||
             sdm >= n_rows * keyratio_1d ||
             sdn >= n_rows * keyratio_1d) ||
            (sdi >= n_rows * keyratio_2d &&
             sdj >= n_rows * keyratio_2d &&
             sdk >= n_rows * keyratio_2d &&
             sdm >= n_rows * keyratio_2d &&
             sdn >= n_rows * keyratio_2d &&
             (int)dvs.size() >= n_rows * keyratio_2d))
        {
            ch5[t] = choice({i, j, k, m, n}, dvs.size(), "AVI", -1.0,
                            {sdi, sdj, sdk, sdm, sdn});
            continue;
        }

        ch5[t] = choice({i, j, k, m, n}, (int)dvs.size(), "-", phi,
                        {sdi, sdj, sdk, sdm, sdn});

        // 如果过大可分桶，这里省略
        // ...

        sdi = (int)di.size();
        sdj = (int)dj.size();
        sdk = (int)dk.size();
        sdm = (int)dm.size();
        sdn = (int)dn.size();

        // 5重循环
        for (int ni = 0; ni < sdi - 1; ni++)
        {
            for (int nj = 0; nj < sdj - 1; nj++)
            {
                for (int nk = 0; nk < sdk - 1; nk++)
                {
                    for (int nm = 0; nm < sdm - 1; nm++)
                    {
                        for (int nn = 0; nn < sdn - 1; nn++)
                        {
                            float pijkmn = 0.0f;
                            for (int r = 0; r < n_rows; r++)
                            {
                                bool inBucket =
                                        (tab[r][i] >= di[ni] && tab[r][i] < di[ni + 1]) &&
                                        (tab[r][j] >= dj[nj] && tab[r][j] < dj[nj + 1]) &&
                                        (tab[r][k] >= dk[nk] && tab[r][k] < dk[nk + 1]) &&
                                        (tab[r][m] >= dm[nm] && tab[r][m] < dm[nm + 1]) &&
                                        (tab[r][n] >= dn[nn] && tab[r][n] < dn[nn + 1]);
                                if (inBucket) {
                                    pijkmn += 1.0f;
                                }
                            }
                            pijkmn /= (float)n_rows;
                            float expected = pi[di[ni]] * pj[dj[nj]] *
                                             pk[dk[nk]] * pm[dm[nm]] *
                                             pn[dn[nn]];
                            if (expected > 1e-12) {
                                phi += pow(pijkmn - expected, 2.0f) / expected;
                            }
                        }
                    }
                }
            }
        }
        phi /= (float)(min({sdi, sdj, sdk, sdm, sdn}) - 1);
        ch5[t].corr = phi;
        if (phi < 0.001) {
            ch5[t].ch = "AVI";
        }
    }
}

// ---------------------- 其他函数与主函数 ----------------------
void transOutput(const string& ifnm,
                 const string& ofnm,
                 const vector<string>& trueColNames)
{
    ifstream in(ifnm.c_str());
    ofstream out(ofnm.c_str());

    if (!in.is_open()) {
        cerr << "Error opening input file: " << ifnm << endl;
        return;
    }
    if (!out.is_open()) {
        cerr << "Error opening output file: " << ofnm << endl;
        return;
    }

    string line;
    while (getline(in, line)) {
        // 使用 \t 拆分
        vector<string> tokens;
        {
            stringstream ss(line);
            string tmp;
            while (getline(ss, tmp, '\t')) {
                tokens.push_back(tmp);
            }
        }

        // 确定要替换的列数
        int replaceCount = 0;
        if (tokens.size() == 3) replaceCount = 1;
        else if (tokens.size() == 7) replaceCount = 2;
        else if (tokens.size() == 9) replaceCount = 3;
        else if (tokens.size() == 11) replaceCount = 4;
        else if (tokens.size() == 13) replaceCount = 5;

        // 替换列索引
        for (int i = 0; i < replaceCount; i++) {
            try {
                int index = stoi(tokens[i]);
                if (index >= 0 && index < (int)trueColNames.size()) {
                    tokens[i] = trueColNames[index];
                } else {
                    cerr << "Index out of range in trueColNames: " << index << endl;
                }
            } catch (...) {
                cerr << "Failed to convert \"" << tokens[i] << "\" to int." << endl;
            }
        }

        // 重新拼接并写入文件
        for (size_t i = 0; i < tokens.size(); ++i) {
            if (i > 0) out << "\t";
            out << tokens[i];
        }
        out << "\n";
    }

    in.close();
    out.close();
}


// 生成所有 k 列组合的工具函数（从 cols 中选取大小为 k 的子集）
static void combineK(const vector<int>& cols, int k,
                     int start, vector<int>& path,
                     vector<vector<int>>& res)
{
    if ((int)path.size() == k) {
        res.push_back(path);
        return;
    }
    for (int i = start; i < (int)cols.size(); i++) {
        path.push_back(cols[i]);
        combineK(cols, k, i + 1, path, res);
        path.pop_back();
    }
}

void CORDS(const string& ifnm,
           const string& ofnm,
           const string& scols)
{
    ofstream ofs(ofnm.c_str());
    vector<vector<string>> tab;
    string line, v;
    ifstream in(ifnm.c_str());

    while (getline(in, line))
    {
        vector<string> ln;
        istringstream ss(line);
        while (getline(ss, v, ','))
            ln.push_back(v);
        tab.push_back(ln);
    }
    in.close();

    vector<int> cols;
    {
        istringstream ss(scols);
        while (getline(ss, v, ','))
            cols.push_back(stoi(v));
    }

    int n_rows = (int)tab.size();
    if (n_rows == 0) {
        cerr << "Error: No data read from input file " << ifnm << endl;
        return;
    }

    // 计算各列每个值出现频率 p[c][value] & 离散取值列表 d[c]
    map<int, map<string, float>> p;
    map<int, vector<string>> d;

#pragma omp parallel for
    for (size_t ci = 0; ci < cols.size(); ci++) {
        int c = cols[ci];
        map<string, float> pi;
        vector<string> di;
        di.reserve(n_rows);

        for (int r = 0; r < n_rows; r++) {
            di.push_back(tab[r][c]);
        }
        sort(di.begin(), di.end());
        di.erase(unique(di.begin(), di.end()), di.end());
        for (auto &val : di) {
            pi[val] = 0.0;
        }
        for (int r = 0; r < n_rows; r++) {
            pi[tab[r][c]] += 1.0f;
        }
        for (auto &kv : pi) {
            kv.second /= (float)n_rows;
        }

#pragma omp critical
        {
            p[c] = pi;
            d[c] = di;
        }
    }

    // 并行计算 2D、3D、4D、5D
    vector<choice> ch2, ch3, ch4, ch5;

#pragma omp parallel sections
    {
#pragma omp section
        {
            vector<vector<int>> comb2;
            vector<int> path;
            combineK(cols, 2, 0, path, comb2);
            ch2.reserve(comb2.size());

#pragma omp parallel for
            for (size_t i = 0; i < comb2.size(); i++) {
                ch2.push_back(choice(comb2[i], 0, "", 0.0, {}));
            }
            cords2d(tab, ch2, n_rows, p, d);
            sort(ch2.begin(), ch2.end(), [](const choice &a, const choice &b) {
                return a.corr > b.corr;
            });
        }

#pragma omp section
        {
            vector<vector<int>> comb3;
            vector<int> path;
            combineK(cols, 3, 0, path, comb3);
            ch3.reserve(comb3.size());

#pragma omp parallel for
            for (size_t i = 0; i < comb3.size(); i++) {
                ch3.push_back(choice(comb3[i], 0, "", 0.0, {}));
            }
            cords3d(tab, ch3, n_rows, p, d);
            sort(ch3.begin(), ch3.end(), [](const choice &a, const choice &b) {
                return a.corr > b.corr;
            });
        }

#pragma omp section
        {
            vector<vector<int>> comb4;
            vector<int> path;
            combineK(cols, 4, 0, path, comb4);
            ch4.reserve(comb4.size());

#pragma omp parallel for
            for (size_t i = 0; i < comb4.size(); i++) {
                ch4.push_back(choice(comb4[i], 0, "", 0.0, {}));
            }
            cords4d(tab, ch4, n_rows, p, d);
            sort(ch4.begin(), ch4.end(), [](const choice &a, const choice &b) {
                return a.corr > b.corr;
            });
        }

#pragma omp section
        {
            vector<vector<int>> comb5;
            vector<int> path;
            combineK(cols, 5, 0, path, comb5);
            ch5.reserve(comb5.size());

#pragma omp parallel for
            for (size_t i = 0; i < comb5.size(); i++) {
                ch5.push_back(choice(comb5[i], 0, "", 0.0, {}));
            }
            cords5d(tab, ch5, n_rows, p, d);
            sort(ch5.begin(), ch5.end(), [](const choice &a, const choice &b) {
                return a.corr > b.corr;
            });
        }
    }

    // 写入输出文件
    ofs << cols.size() << "\n";
    for (auto c : cols)
    {
        if (d[c].size() < n_rows * keyratio_1d)
            ofs << c << "\t" << d[c].size() << "\tAVI\n";
        else
            ofs << c << "\t" << d[c].size() << "\tKey\n";
    }

    // 输出 2D 结果
    ofs << ch2.size() << "\n";
    for (auto &it : ch2) {
        ofs << it.id[0] << "\t" << it.id[1] << "\t"
            << it.dvs << "\t" << it.corr << "\t"
            << it.ch << "\t" << it.dv[0] << "\t"
            << it.dv[1] << "\n";
    }

    // 输出 3D 结果
    ofs << ch3.size() << "\n";
    for (auto &it : ch3) {
        ofs << it.id[0] << "\t" << it.id[1] << "\t" << it.id[2] << "\t"
            << it.dvs << "\t" << it.corr << "\t"
            << it.ch << "\t";
        for (auto dvv : it.dv) ofs << dvv << "\t";
        ofs << "\n";
    }

    // 输出 4D 结果
    ofs << ch4.size() << "\n";
    for (auto &it : ch4) {
        ofs << it.id[0] << "\t" << it.id[1] << "\t" << it.id[2] << "\t" << it.id[3] << "\t"
            << it.dvs << "\t" << it.corr << "\t" << it.ch << "\t";
        for (auto dvv : it.dv) ofs << dvv << "\t";
        ofs << "\n";
    }

    // 输出 5D 结果
    ofs << ch5.size() << "\n";
    for (auto &it : ch5) {
        ofs << it.id[0] << "\t" << it.id[1] << "\t" << it.id[2] << "\t" << it.id[3] << "\t" << it.id[4] << "\t"
            << it.dvs << "\t" << it.corr << "\t" << it.ch << "\t";
        for (auto dvv : it.dv) ofs << dvv << "\t";
        ofs << "\n";
    }

    ofs << "0\n";
    ofs.flush();
    ofs.close();
}

// ---------------------- main 函数示例 ----------------------
int main()
{
    // 以 Census 为例
    // 这里仅示例，你可自行修改数据集路径、输出路径等
    string ifnm = "/home/qswang/iris_demo/dataset_public/Census/census";
    string ofnm = "CORDS_Census_multi.log";
    string transOfnm = "CORDS_Census_multi_trans.log";

    // 假设我们想要处理以下列下标
//    string cols = "0,1,2,3,4,5,6,7,8,9,10,11,12,13";
    string cols = "0,1,2,3,4";


    // 真正的列名（用于 transOutput）
//    vector<string> trueColNames = {
//            "0","1","3","4","5","6","7","8","9","10","11","12","13","14"
//    };
    vector<string> trueColNames = {
            "0","1","3","4","5"
    };

    // 调用
    CORDS(ifnm, ofnm, cols);

    // 可选：把结果文件中的下标替换成真实列名
    transOutput(ofnm, transOfnm, trueColNames);

    return 0;
}
