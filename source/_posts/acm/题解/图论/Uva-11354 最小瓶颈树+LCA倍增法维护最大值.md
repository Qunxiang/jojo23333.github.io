---
title: Uva-11354 最小瓶颈树+LCA倍增法维护最大值
date: 2017-8-27 12:03:34
categories: ACM
tags:  [最小瓶颈树,kruskal,倍增法,LCA]
---

# Uva-11354
题意：  
给你一个无向图，N个节点M条边，边权为d，对Q组询问a b,问能取到的从a到b路径上的最小值。

题解：  
首先总是要取最小的值，则可以先用kruskal求最小生成树（也就是最小瓶颈树），即在树上求任意两点之间路径边权值的最小值。可以用倍增求解LCA的方法，**在保存p[i][j]（节点i的向上2^i个祖先） 的同时维护mlen[i][j]（节点i向上2^i条边的最大值）**

```c++
#include <iostream>
#include <cstdio>
#include <cctype>
#include <algorithm>
#include <cstring>
#include <string>
#include <cmath>
#include <vector>
#include <set>
#include <stack>
#include <sstream>
#include <queue>
#include <map>
#include <functional>
#include <bitset>

using namespace std;
#define pb push_back
#define mk make_pair
#define ll long long
#define ull unsigned long long
#define pii pair<int, int>
#define mkp make_pair
#define fst first
#define scd second
#define ALL(A) A.begin(), A.end()
#define REP(i,n) for(int (i)=0;(i)<(int)(n);(i)++)
#define REP1(i, n) for(int (i)=1;(i)<=(int)(n);(i)++)
#define fastio ios::sync_with_stdio(0), cin.tie(0)
#define frein freopen("in.txt", "r", stdin)
#define freout freopen("out.txt", "w", stdout)
#define freout1 freopen("out1.txt", "w", stdout)
#define PI M_PI
#define MAXN 100000
#define xork(a,b) ((b&1)?(a):(0))
#define sc(n) scanf("%d",&(n))

ll mod = 10000;
ll INF = 1LL<<60LL;
const double eps = 1e-8;
template<typename T> T gcd(T a,T b)
{if(!b)return a;return gcd(b,a%b);}
struct edge{
    int from,to;
    int v;
    bool operator<(const edge &a)const{
        return v<a.v;
    }
};

vector<edge> E;
vector<vector<pii> >G(MAXN);
int d[MAXN],len[MAXN];
int p[MAXN][20],mlen[MAXN][20];
int f[MAXN];
int N,M;

int getf(int v){
    if(f[v]==v) return v;
    else return f[v] = getf(f[v]);
}

bool Merge(int v1,int v2){
    int f1 = getf(v1);
    int f2 = getf(v2);
    if(f1 == f2){
        return false;
    }
    f[f1] = f2;
    return true;
}

void kruskal()
{
    for(int i=0; i<=N; i++)
        f[i] = i;
    int cnt = 0;
    for(int i=0; i<E.size(); i++){
        if(cnt>=N-1)    break;
        int f = E[i].from;
        int t = E[i].to;
        if(Merge(f,t)){
            cnt++;
            G[f].pb(mkp(t,E[i].v));
            G[t].pb(mkp(f,E[i].v));
//            cout<<f<<" "<<t<<" "<<E[i].v<<endl;
        }
    }
}

void dfs(int v,int pre,int depth)
{
    d[v] = depth;
    for(int i=0; i<G[v].size(); i++){
        int t = G[v][i].first;
        int val = G[v][i].second;
        if(t==pre)  continue;
        dfs(t,v,depth+1);
        //len[t] = val;
        p[t][0] = v;
        mlen[t][0] = val;
    }
}

void lca_init(int n)
{
//    for(int i=1; i<=n; i++)
//        printf("mlen[%d][0] = %d\n",i,mlen[i][0]);
    for(int j=1; (1<<j)<=n; j++){
        for(int i=1; i<=n; i++){
            p[i][j] = p[p[i][j-1]][j-1];
            mlen[i][j] = max(mlen[i][j-1],mlen[p[i][j-1]][j-1]);
            //printf("mlen[%d][%d] = %d\n",i,j,mlen[i][j]);
        }
    }
}

int query(int a,int b)
{
    //printf("Query a:%d b%d\n",a,b);
    if(d[a]>d[b])  swap(a,b);
    int f = d[b] - d[a];
    int maxe = -1;
    for(int i=0; (1<<i)<=f; i++)
        if(f&(1<<i)){
            maxe = max(maxe,mlen[b][i]);
            b = p[b][i];
        }
    //printf("maxe = %d\n",maxe);
    if(a!=b){
        for(int i=(int)log2(N);i>=0; i--){
            if(p[a][i]!=p[b][i]){
                maxe = max(maxe,max(mlen[b][i],mlen[a][i]));
                a = p[a][i];    b = p[b][i];
            }
            //printf("maxe = %d\n",maxe);
        }
        maxe = max(maxe,mlen[a][0]);    //和求LCA不同，这里要同时对两个节点更新最大值
        maxe = max(maxe,mlen[b][0]);
        //printf("a = %d  maxe = %d\n",a,maxe);
    }
    return maxe;
}

int main()
{
    //freout;
    bool flag = false;
    while(~scanf("%d%d",&N,&M)){
        if(flag) puts("");
        flag = true;
        E.clear();
        for(int i=1; i<=N; i++)
            G[i].clear();
        for(int i=0; i<M; i++){
            edge t;
            scanf("%d%d%d",&t.from,&t.to,&t.v);
            E.pb(t);
        }
        sort(E.begin(),E.end());
        kruskal();
        dfs(1,-1,0);
        lca_init(N);
        int Q;
        sc(Q);
        for(int i=0; i<Q; i++){
            int a,b;
            sc(a); sc(b);
            printf("%d\n",query(a,b));
        }
    }
}

```