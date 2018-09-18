---
title: UVAlive3211-2SAT模板题
date: 2017-09-03 18:32:08
tags: [2SAT]
categories: ACM
---
## UVAlive-3211
题意大概是求n架飞机每个飞机两个降落时间取一个使相邻降落的飞机时间差最大  
要最优化差值，二分转化为在某一时间下能否满足，按照相邻时间是否大于当前二分的时间构建2-SAT
模板题模板题  

```c++
ll mod = 10000;
ll INF = 1LL<<60LL;
const double eps = 1e-8;
template<typename T> T gcd(T a,T b)
{if(!b)return a;return gcd(b,a%b);}

//struct time{
//    int time,id;
//}t[MAXN];
int t[MAXN];

struct SAT2{
    int vis[MAXN*2],cnt;
    vector<int> G[MAXN*2];
    int n;
    bool mark[MAXN*2];
    bool dfs(int x){
        if(mark[x^1]) return false;
        else if(mark[x]) return true;
        mark[x] = true;
        vis[cnt++] = x;
        for(int i=0; i<G[x].size(); i++){
            if(!dfs(G[x][i])) return false;
        }
        return true;
    }

    void init(int n){
        this->n = n;
        for(int i=0; i<=2*n; i++){
            vis[i] = false;
            G[i].clear();
        }
        memset(mark,false,sizeof(mark));
    }

    // 当 x!=xval or y != yval(val=0/1)
    // 则由于拆点每个x或者y变成两个点
    // x==xval=>y!=yval     y==yval=>x!=xval
    void add_clause(int x,int xval,int y,int yval){
        x = x*2+xval;
        y = y*2+yval;
        G[x].pb(y^1);
        G[y].pb(x^1);
    }

    bool solve(){
        for(int i=0; i<2*n; i+=2)
        if(!mark[i] && !mark[i^1]){
            cnt = 0;
            if(!dfs(i)){
                while(cnt>0) mark[vis[--cnt]] = false;
                if(!dfs(i^1)) return false;
            }
        }
        return true;
    }
}st;
int n;

bool solve(int T)
{
    st.init(n);
    for(int i=0; i<2*n; i++){
        int j;
        if(i%2) j = i+1;
        else    j = i+2;
        for(;j<2*n;j++){
            if(abs(t[i]-t[j])<T){
                st.add_clause(i/2,i&1,j/2,j&1);
            }
        }
    }
    return st.solve();
}

int main()
{
    while(sc(n)==1&&n){
        for(int i=0; i<n; i++){
            int e,l;
            sc(e);  sc(l);
            t[i*2] = e;    //t[i*2].id = i*2;
            t[i*2+1] = l;  //t[i*2+1].id=i*2+1;
        }
        int lt = 0,rt = 1e7;
        while(lt<rt){
            //cout<<lt<<" "<<rt<<endl;
            int mt = (lt+rt+1)/2;
            if(solve(mt))
                lt = mt;
            else
                rt = mt-1;
        }
        cout<<lt<<endl;
    }
}

```
