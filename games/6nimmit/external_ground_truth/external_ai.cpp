#include <iostream>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>
#include <numeric>
#include <cassert>

#ifdef SHOW
#define DEBUG(x) x
#else
#define DEBUG(x)
#endif

using namespace std;

const int ROWS = 4;
const int COLS = 5;
const int HAND_CARDS = 10;
const int MAX_CARDS = 104;
const int MAX_ROUNDS = 10;
const double INF = 1e20;
const int NP = 2;

int depth_limit;

template <typename T>
inline bool relax(T &a, const T &b) {
	if (b > a) {
		a = b;
		return true;
	}
	return false;
}
template <typename T>
inline bool tense(T &a, const T &b) {
	if (b < a) {
		a = b;
		return true;
	}
	return false;
}

int heads[MAX_CARDS + 1];
double jcl[MAX_CARDS + 1];
double cmaxprob[MAX_CARDS + 1][HAND_CARDS + 1][MAX_CARDS + 1];

bool small;
int player_id, cur_round;
int player_num, cards_num;
int lastc[NP];

int best_card;

double C_log(int n, int m) {
	return jcl[n] - jcl[m] - jcl[n - m];
}

inline int getint() {
	int x;
	cin >> x;
	DEBUG(cerr << x << " ");
	return x;
}

struct Stack {
	int t, n, h;

    Stack() {
        n = 0;
		t = 0;
		h = 0;
    }
    void init() {
        for (int j = 0; j < COLS; ++j) {
            int x = getint();
            if (x > 0) {
                n = j + 1;
                h += heads[x];
                t = x;
            }
        }
    }
    bool full() const {
        return n == COLS;
    }
    void push(int x) {
        t = x;
        h += heads[x];
		n++;
    }
    int refresh(int x) {
        int res = -h;
        n = 1;
        t = x;
        h = heads[x];
        return res;
    }
    int value() const {
        return -h;
    }

	friend inline bool operator < (const Stack &lhs, const Stack &rhs) {
		return lhs.t < rhs.t;
	}
};

bool ecb[MAX_CARDS + 1];

struct Status {
    Stack s[ROWS];
    int c[HAND_CARDS], n;

    Status() {
        n = 0;
        memset(c, 0, sizeof(c));
    }
    void init_stacks() {
        for (int i = 0; i < ROWS; ++i)
            s[i].init();
    }
    void remove(int mc, int ec) {
        int q = 0;
        for (int i = 0; i < n; ++i) {
            if (c[i] != mc)
				c[q++] = c[i];
        }
        n = q;
    }
    int add(int x) {
        int r = -1;
        for (int i = 0; i < ROWS; ++i) {
            if (s[i].t < x && (r == -1 || s[i].t > s[r].t))
                r = i;
        }
        if (r == -1) {
            int maxv = -9999;
            for (int i = 0; i < ROWS; ++i)
				if (relax(maxv, s[i].value()))
					r = i;
            return s[r].refresh(x);
        } else if (s[r].full()) {
            return s[r].refresh(x);
        } else {
            s[r].push(x);
            return 0;
        }
    }
	int best_small_stack(int x, int y) {
		int maxv = -10000;
		int maxi;
		for (int i = 0; i < ROWS; ++i) {
			Status ns = *this;
			int v = 0;
			v += ns.s[i].refresh(x);
			v -= ns.add(y);
			if (relax(maxv, v))
				maxi = i;
		}
		return maxi;
	}
    int add(int x, int y) {
		if (x > y)
			return -add(y, x);
        int r = -1;
        for (int i = 0; i < ROWS; ++i) {
            if (s[i].t < x && (r == -1 || s[i].t > s[r].t))
                r = i;
        }
        if (r == -1) {
			r = best_small_stack(x, y);
            return s[r].refresh(x) - add(y);
        } else if (s[r].full()) {
            return s[r].refresh(x) - add(y);
        } else {
            s[r].push(x);
            return -add(y);
        }
	}

    double estimate() const {
		const int T = 5;
        double res = 0;
		for (int ti = 0; ti < T; ti++) {
			Status ns = *this;
			static int cm[HAND_CARDS];
			for (int i = 0; i < n; i++)
				cm[i] = c[i];
			random_shuffle(cm, cm + n);

			for (int i = 0; i < n; i++) {
				int d = rand() % 104 + 1;
				res += ns.add(cm[i], d);
			}
		}
		res /= T;
        return res;
    }
} cur_s;

void prepare() {
    for (int i = 1; i <= MAX_CARDS; ++i) {
        int &x = heads[i];
        x = 0;
        if (i % 5 == 0) x += 2;
        if (i % 10 == 0) x += 1;
        if (i % 11 == 0) x += 5;
        if (x == 0) x = 1;
    }
    for (int i = 1; i <= MAX_CARDS; ++i) {
        jcl[i] = jcl[i - 1] + log((double)i);
    }

	for (int nc = 0; nc <= MAX_CARDS; nc++) {
		for (int k = 1; k <= min(HAND_CARDS, nc); k++) {
			for (int i = 0; i < nc; i++) {
				if (nc - i >= k) {
					cmaxprob[nc][k][i] = exp(C_log(nc - i - 1, k - 1) - C_log(nc, k));
				}
			}
		}
	}
}

void data_init(int argc, char *argv[]) {
    if (argc == 2 && argv[1][1] == 's') small = true;
    player_id = getint();
	cur_round = getint();
    cur_s.n = HAND_CARDS - cur_round;
    for (int i = 0; i < cur_s.n; ++i) {
        int x = getint();
        cur_s.c[i] = x;
        ecb[x] = true;
    }
    player_num = getint();
	assert(player_num == NP);

	cards_num = getint();
    if (small) cur_round--;

    for (int t = 0; t <= cur_round; ++t) {
        if (t == cur_round) {
            cur_s.init_stacks();
        } else {
            for (int i = 0; i < ROWS; ++i)
                for (int j = 0; j < COLS; ++j) {
                    int x = getint();
                    if (x > 0 && t == 0) ecb[x] = true;
                }
        }
        if (t < cur_round || small) {
            for (int i = 0; i < player_num; ++i) {
				int x = getint();
				lastc[i] = x;
                if (x > 0) ecb[x] = true;
            }
        }
    }
	DEBUG(cerr << endl);
}

struct CYB {
	int dist_n;
	pair<double, int> dist[MAX_CARDS];

	void get_prob_dist(const Status &s) {
		Stack si[ROWS];
		for (int r = 0; r < ROWS; r++)
			si[r] = Stack(s.s[r]);

		sort(si, si + ROWS);

		static int bp[ROWS + 1];
		for (int r = 0; r < ROWS; r++)
			bp[r] = si[r].t;
		bp[ROWS] = 105;

		double minh = INF;
		for (int r = 0; r < ROWS; r++)
			tense(minh, (double)si[r].h);

		const int NS = 2;
		static double prob[MAX_CARDS + 1];
		static bool cb[MAX_CARDS + 1];

		for (int i = 1; i <= 104; i++)
			cb[i] = ecb[i];
		for (int i = 0; i < s.n; i++)
			cb[s.c[i]] = false;

		int ncb = 0;
		for (int i = 1; i <= 104; i++)
			if (!cb[i])
				ncb++;

		for (int i = 1; i <= 104; i++)
			prob[i] = cb[i] ? 0 : 1.0 / ncb;

		for (int stage = 1; stage <= NS; stage++) {
			if (stage == NS) {
				for (int i = 0; i < s.n; i++)
					cb[s.c[i]] = true;
			}
			for (int i = 1; i <= 104; i++)
				prob[i] += prob[i - 1];
			assert(abs(prob[104] - 1) < 1e-6);

			dist_n = 0;
			for (int i = 1; i < bp[0]; i++)
				if (!cb[i])
					dist[dist_n++] = make_pair(-minh, i);

			for (int r = 0; r < ROWS; r++)
				for (int i = bp[r] + 1; i < bp[r + 1]; i++)
					if (!cb[i]) {
						if (si[r].n == 5) {
							double p = 1 - (prob[i] - prob[bp[r] - 1]);
							dist[dist_n++] = make_pair(-si[r].h * p, i);
						} else if (si[r].n == 4) {
							double p = prob[i] - prob[bp[r] - 1];
							double q = prob[bp[r + 1]] - prob[i];
							dist[dist_n++] = make_pair(si[r].h * (q - p), i);
						} else {
							dist[dist_n++] = make_pair(0, i);
						}
					}

			if (stage == NS) {
				random_shuffle(dist, dist + dist_n);
				dist_n = (int)ceil(dist_n * 0.5);
			}

			sort(dist, dist + dist_n);
			reverse(dist, dist + dist_n);
			for (int i = 0, j = 0; i < dist_n; i = j) {
				while (j < dist_n && abs(dist[i].first - dist[j].first) < 1e-3) {
					j++;
				}

				double p = 0;
				for (int k = i; k < j; k++)
					p += cmaxprob[dist_n][s.n][k];
				for (int k = i; k < j; k++)
					dist[k].first = p / (j - i);
			}
			if (stage == NS) {
				while (dist_n > 0 && dist[dist_n - 1].first < 0.001) {
					dist_n--;
				}
			} else {
				for (int i = 1; i <= 104; i++)
					prob[i] = 0;

				double tp = 0;
				for (int i = 0; i < dist_n; i++) {
					prob[dist[i].second] = dist[i].first;
					tp += dist[i].first;
				}
				for (int i = 1; i <= 104; i++)
					prob[i] /= tp;
			}
		}
	}
};

double dfs(const Status &s, int depth) {
	if (s.n == 0) {
		return 0;
	}
    if (depth == depth_limit) {
        return s.estimate();
    }

    CYB cyb;
    cyb.get_prob_dist(s);

    double res = -INF;

    for (int i = 0; i < s.n; ++i) {
        int mc = s.c[i];

        double cur = 0;
		double tp = 0;

        for (int j = 0; j < cyb.dist_n; ++j) {
            double p = cyb.dist[j].first;
            int ec = cyb.dist[j].second;

            Status ns = s;
            ns.remove(mc, ec);
            int t = ns.add(mc, ec);

			tp += p;

            ecb[ec] = true;
			cur += p * (t + 0.9 * dfs(ns, depth + 1));
            ecb[ec] = false;
        }

		cur /= tp;

        if (relax(res, cur) && !depth) {
            best_card = mc;
        }
    }

    return res;
}

void solve_small() {
	cout << cur_s.best_small_stack(lastc[player_id], lastc[!player_id]);
}

int get_depth_limit() {
	if (cur_s.n > 4)
        return 2;
	else
		return 3;
}

int main(int argc, char *argv[]) {
    prepare();

    data_init(argc, argv);

    if (small) {
        solve_small();
        return 0;
    }

    depth_limit = get_depth_limit();
    dfs(cur_s, 0);

    cout << best_card << endl;
	return 0;
}
