---
title: 简单的LCA==
date: 2017-09-03 13:49:50
tags: [LCA]
categories: ACM
---
题源： NAIPC 2016 The University Of Chicago
校内赛一道LCA==，写掉了一个等号调了一下午== 
dfs爆栈然后手写栈模拟了一发

```c++
int depth[MAXN*2];
struct ST{
    int mm[2*MAXN];
    int dp[2*MAXN][25];
    void init(int n){
        mm[0] = -1;
        for(int i=1;i<=n; i++){
            mm[i] = ((i&(i-1))==0)?mm[i-1]+1:mm[i-1];
            dp[i][0] = i;
        }
        for(int j=1 ;j<=mm[n] ;j++)
            for(int i=1; i+(1<<j)-1<=n; i++)
                dp[i][j] = (depth[dp[i][j-1]] < depth[dp[i+(1<<(j-1))][j-1]])? dp[i][j-1]:dp[i+(1<<(j-1))][j-1];
    }
    int query(int a,int b){
        if(a>b)swap(a,b);
        int k = mm[b-a+1];
        return depth[dp[a][k]]<=depth[dp[b-(1<<k)+1][k]]?dp[a][k]:dp[b-(1<<k)+1][k];
    }
}st;

vector<vector<int> >G(MAXN);
int vs[MAXN*2];
int pos[MAXN];


//void dfs(int v,int pre,int d,int &k){
//    pos[v] = k;
//    vs[k] = v;
//    depth[k++] = d;
//    for(int i=0; i<G[v].size(); i++){
//        if(G[v][i]!=pre){
//            dfs(G[v][i],v,d+1,k);
//            vs[k] = v;
//            depth[k++]=d;
//        }
//    }
//}

struct dfs{
    int v,pre,d,i;
    dfs(int vv,int pp,int dd,int ii):v(vv),pre(pp),d(dd),i(ii){}
};
stack<dfs> s;

void solve(int &k){
    s.push(dfs(1,0,0,0));
    int v,pre,d,i;
    while(!s.empty()){
        dfs cur = s.top();
        s.pop();
        //cout<<s.size()<<endl;
        v = cur.v;  pre = cur.pre; d = cur.d; i = cur.i;
        if(i==0){
            pos[v] = k;
            vs[k] = v;
            depth[k++] = d;
        }
        else{
            vs[k] = v;
            depth[k++]=d;
        }
        //cout<<v<<" "<<G[v].size()<<endl;
        if(i<G[v].size() && G[v][i]==pre) i++;
        if(i<G[v].size()){
            //cout<<"from:"<<v<<" to:"<<G[v][i]<<endl;
            cur.i = i+1;
            s.push(cur);
            s.push(dfs(G[v][i],v,d+1,0));
        }
    }
}

int lca_query(int v,int u)
{
    int s = min(pos[v],pos[u]);
    int e = max(pos[v],pos[u]);
    return 1+depth[s]+depth[e]-2*depth[st.query(s,e)];
}

int main()
{
    frein;
    int n;
    scanf("%d",&n);
    for(int i=0; i<n-1; i++){
        int f,t;
        scanf("%d%d",&f,&t);
        G[f].pb(t);
        G[t].pb(f);
    }
//    for(int i=1;i<n; i++){
//        cout<<i<<":  "<<endl;
//        for(int j=0; j<G[i].size(); j++)
//            cout<<G[i][j]<<" ";
//        cout<<endl;
//    }
    int cnt=1;
    solve(cnt);
//    for(int i=0; i<cnt;i++)
//        cout<<vs[i]<<" depth: "<<depth[i]<<endl;
//    for(int i=1; i<cnt; i++){
//        cout<<"node:"<<vs[i]<<" depth:"<<depth[i]<<endl;
//    }
//    for(int i=1; i<=n; i++)
//        cout<<"vis"<<i<<" is "<<pos[i]<<endl;
    st.init(2*n-1);
    ll ans = 0;
    for(int i=1; i<=n/2; i++)
        for(int j=2; i*j<=n; j++){
            ans += lca_query(i,i*j);
        }
    cout<<ans<<endl;
    return 0;
}
```
