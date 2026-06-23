#include "cords.h"

static vector<int> primes{ 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431 };

static int nbuckets = 16;
static float keyratio_1d = 0.85;
static float keyratio_2d = 0.75;

void cords2d(vector<vector<string>> tab, vector<choice>& ch, int n_rows, map<int, map<string, float>> p, map<int, vector<string>> d)
{
	//#pragma omp parallel for
	for (int t = 0; t < ch.size(); t++)
	{
		int i = ch[t].id[0];
		int j = ch[t].id[1];
		double phi = 0;
		auto di = d[i], dj = d[j];
		auto pi = p[i], pj = p[j];
		int sdi = di.size(), sdj = dj.size();

		vector<int> dvs;
		for (int r = 0; r < n_rows; r++)
			dvs.push_back(hash<string>()(tab[r][i]) * primes[0] + hash<string>()(tab[r][j]) * primes[1]);
		sort(dvs.begin(), dvs.end());
		dvs.erase(unique(dvs.begin(), dvs.end()), dvs.end());

        //prune cases
        if ((di.size () >= n_rows * keyratio_1d || dj.size() >= n_rows * keyratio_1d) ||
		    (di.size() >= n_rows * keyratio_2d && dj.size() >= n_rows * keyratio_2d && dvs.size() >= n_rows * keyratio_2d))
		{
			ch[t] = choice(vector<int>{i, j}, dvs.size(), "AVI", -1, vector<int>{sdi, sdj});
			continue;
		}

        //conservative - more sparse depending on storage budget
//        hxh 注释掉Sparse类别。改为，无论是否稀疏，都计算phi
//        if(dvs.size() <= 64)
//        {
//            ch[t] = choice(vector<int>{i, j}, dvs.size(), "Sparse", 1, vector<int>{sdi, sdj});
//            continue;
//        }

//        hxh 注释掉Hist类别。改为，无论是否域值空间相差过大，都计算phi
		bool ishist = (max(sdi/sdj, sdj/sdi) >= 128);
//        if(ishist)
//        {
//            ch[t] = choice(vector<int>{i, j}, dvs.size(), "Hist", 0, vector<int>{sdi, sdj});
//            continue;
//        }

		ch[t] = choice(vector<int>{i, j}, dvs.size(), "-", phi, vector<int>{sdi, sdj});

		//bucketing
		if (dvs.size() > nbuckets * nbuckets)
		{
			vector<string> dtmp;
			map<string, float> ptmp;
			for (int n = 0; n < nbuckets; n++)
			{
				dtmp.push_back(di[n * di.size() / nbuckets]);
				ptmp[di[n * di.size() / nbuckets]] = 0;
				for (int t = n * di.size() / nbuckets; t < (n + 1) * di.size() / nbuckets; t++)
					ptmp[di[n * di.size() / nbuckets]] += pi[di[t]];
			}
			pi = ptmp;
			dtmp.push_back(di[di.size() - 1]);
			di = vector<string>(dtmp);
			dtmp.clear();
			ptmp.clear();
			for (int n = 0; n < nbuckets; n++)
			{
				dtmp.push_back(dj[n * dj.size() / nbuckets]);
				ptmp[dj[n * dj.size() / nbuckets]] = 0;
				for (int t = n * dj.size() / nbuckets; t < (n + 1) * dj.size() / nbuckets; t++)
					ptmp[dj[n * dj.size() / nbuckets]] += pj[dj[t]];
			}
			pj = ptmp;
			dtmp.push_back(dj[dj.size() - 1]);
			dj = vector<string>(dtmp);

			//di.erase(unique(di.begin(), di.end()), di.end());
			//dj.erase(unique(dj.begin(), dj.end()), dj.end());
		}

		for (int ni = 0; ni < di.size() - 1; ni++)
		{
			for (int nj = 0; nj < dj.size() - 1; nj++)
			{
				float pij = 0;
				for (int r = 0; r < n_rows; r++)
				{
					if (tab[r][i] >= di[ni] && tab[r][j] >= dj[nj] && tab[r][i] < di[ni + 1] && tab[r][j] < dj[nj + 1])
						pij++;
				}
				pij /= n_rows;
				phi += pow(pij - pi[di[ni]] * pj[dj[nj]], 2) / (pi[di[ni]] * pj[dj[nj]]);
			}
		}
		phi /= (min(di.size(), dj.size()) - 1);

        ch[t].corr = phi;
        if(ishist) ch[t].ch = "Hist";
		if (phi < 0.001)
		{
            ch[t].ch = "AVI";
//          hxh 注释掉ch[t].corr = -1;
//            ch[t].corr = -1;
		}
	}
}

void CORDS(string ifnm, string ofnm, string scols)
{
	ofstream ofs(ofnm.c_str());
	vector<vector<string>> tab;
	string line, v;
	ifstream in(ifnm.c_str());
	while (getline(in, line))
	{
		vector<string> ln;
		istringstream ss(line);
		int id = 0;
		while (getline(ss, v, ','))
			ln.push_back(v);
		tab.push_back(ln);
	}
	vector<int> cols;
	istringstream ss(scols);
	int id = 0;
	while (getline(ss, v, ','))
		cols.push_back(stoi(v)); //stoi()将字符串直接转为整型

	vector<choice> ch;
	for (int i = 0; i < cols.size(); i++)
	{
		for (int j = i + 1; j < cols.size(); j++)
		{
			ch.push_back(choice(vector<int>{cols[i], cols[j]}));
		}
	}
	map<int, map<string, float>> p; // p[c][value] = 该列 c 中某 value 的出现频率
	map<int, vector<string>> d; // d[c] = 该列 c 的所有离散取值
	int n_rows = tab.size();
    if (n_rows == 0) {
        cerr << "Error: No data read from input file " << ifnm << endl;
        return; // 或者直接退出整个程序
    }
	int n_cols = tab[0].size();

	for (int t = 0; t < cols.size(); t++)
	{
		map<string, float> pi;
		vector<string> di;
		for (int j = 0; j < n_rows; j++)
			di.push_back(tab[j][cols[t]]);
		sort(di.begin(), di.end());
		di.erase(unique(di.begin(), di.end()), di.end());
		for (int n = 0; n < di.size(); n++)
		{
			pi[di[n]] = 0;
			for (int j = 0; j < n_rows; j++)
			{
				if (tab[j][cols[t]] == di[n])
					pi[di[n]]++;
			}
			pi[di[n]] /= n_rows;
			p[cols[t]] = pi;
			d[cols[t]] = di;
		}
		//ofs << cols[t] << ": " << di.size() << endl;
	}

	cords2d(tab, ch, n_rows, p, d);
	sort(ch.begin(), ch.end(), [](const choice a, const choice b) {return a.corr > b.corr; });

	//output
	ofs << cols.size() << endl;
	for (int i = 0; i < cols.size(); i++)
	{
        if(d[cols[i]].size() < n_rows * keyratio_1d)
            ofs << cols[i] << "\t" << d[cols[i]].size() << "\tAVI" << endl;
        else
            ofs << cols[i] << "\t" << d[cols[i]].size() << "\tKey" << endl;
	}
	ofs << ch.size() << endl;
	for (auto it = ch.begin(); it != ch.end(); it++)
		ofs << it->id[0] << "\t" << it->id[1] << "\t" << it->dvs << "\t" << it->corr << "\t" << it->ch << "\t" << it->dv[0] << "\t" << it->dv[1] << endl;
	ofs << "0" << endl;
	ofs.flush();
	ofs.close();
}

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
            // 用 stringstream 按照 \t 分隔行
            stringstream ss(line);
            string tmp;
            while (getline(ss, tmp, '\t')) {
                tokens.push_back(tmp);
            }
        }

        // 根据不同大小进行替换
        if (tokens.size() == 3) {
            // 替换第一个字符串
            // 假设 tokens[0] 能转换成整数，用 stoi 解析
            // 并在 trueColNames 中取对应下标
            try {
                int index = stoi(tokens[0]);
                if (index >= 0 && index < (int)trueColNames.size()) {
                    tokens[0] = trueColNames[index];
                } else {
                    cerr << "Index out of range in trueColNames: " << index << endl;
                }
            } catch (...) {
                cerr << "Failed to convert \"" << tokens[0] << "\" to int." << endl;
            }
        }
        else if (tokens.size() == 7) {
            // 替换前两个字符串
            // 假设 tokens[0], tokens[1] 都能转换成整数
            try {
                int index0 = stoi(tokens[0]);
                if (index0 >= 0 && index0 < (int)trueColNames.size()) {
                    tokens[0] = trueColNames[index0];
                } else {
                    cerr << "Index out of range in trueColNames: " << index0 << endl;
                }

                int index1 = stoi(tokens[1]);
                if (index1 >= 0 && index1 < (int)trueColNames.size()) {
                    tokens[1] = trueColNames[index1];
                } else {
                    cerr << "Index out of range in trueColNames: " << index1 << endl;
                }
            } catch (...) {
                cerr << "Failed to convert one of \""
                     << tokens[0] << "\" or \""
                     << tokens[1] << "\" to int." << endl;
            }
        }

        // 将处理后的结果写入输出文件
        for (size_t i = 0; i < tokens.size(); ++i) {
            if (i > 0) out << "\t"; // 用 \t 作为分隔
            out << tokens[i];
        }
        out << "\n";
    }

    in.close();
    out.close();
}

int main()
{
    //  Census
//	string ifnm = "/home/qswang/iris_demo/dataset_public/Census/census";
//	string ofnm = "CORDS_Census_computeAll.log";
//    string transOfnm = "CORDS_Census_trans_computeAll.log";
//	string cols = "0,1,2,3,4,5,6,7,8,9,10,11,12,13";
//    vector<string> trueColNames = {"0","1","3","4","5","6","7","8","9","10","11","12","13","14"};

    //  DMV
//    string ifnm = "/home/qswang/iris_demo/dataset_public/DMV/dmv_nohead_11cols.csv";
//    string ofnm = "CORDS_DMV.log";
//    string transOfnm = "CORDS_DMV_trans.log";
//    string cols = "0,1,2,3,4,5,6,7,8,9,10";
//    vector<string> trueColNames = {"0","2","4","6","9","10","14","16","17","18","19"};

    //  Forest
    string ifnm = "/home/qswang/iris_demo/covtype_nohead.csv";
    string ofnm = "CORDS_Forest_10.log";
    string transOfnm = "CORDS_Forest_trans_10.log";
//    string cols = "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54";
//    vector<string> trueColNames = {"Elevation","Aspect","Slope","Horizontal_Distance_To_Hydrology","Vertical_Distance_To_Hydrology","Horizontal_Distance_To_Roadways","Hillshade_9am","Hillshade_Noon","Hillshade_3pm","Horizontal_Distance_To_Fire_Points","Wilderness_Area1","Wilderness_Area2","Wilderness_Area3","Wilderness_Area4","Soil_Type1","Soil_Type2","Soil_Type3","Soil_Type4","Soil_Type5","Soil_Type6","Soil_Type7","Soil_Type8","Soil_Type9","Soil_Type10","Soil_Type11","Soil_Type12","Soil_Type13","Soil_Type14","Soil_Type15","Soil_Type16","Soil_Type17","Soil_Type18","Soil_Type19","Soil_Type20","Soil_Type21","Soil_Type22","Soil_Type23","Soil_Type24","Soil_Type25","Soil_Type26","Soil_Type27","Soil_Type28","Soil_Type29","Soil_Type30","Soil_Type31","Soil_Type32","Soil_Type33","Soil_Type34","Soil_Type35","Soil_Type36","Soil_Type37","Soil_Type38","Soil_Type39","Soil_Type40","Cover_Type"};
    string cols = "0,1,2,3,4,5,6,7,8,9";
    vector<string> trueColNames = {"Elevation","Aspect","Slope","Horizontal_Distance_To_Hydrology","Vertical_Distance_To_Hydrology","Horizontal_Distance_To_Roadways","Hillshade_9am","Hillshade_Noon","Hillshade_3pm","Horizontal_Distance_To_Fire_Points"};
    CORDS(ifnm, ofnm, cols);
    transOutput(ofnm, transOfnm, trueColNames);
}

