---
title: c++常用算法模板
published: 2025-12-18
description: ''
image: ''
tags: ["算法"]
category: ''
draft: false 
lang: ''
---

# c++常用算法模板
### SPFA单源最短路模板
```cpp
#include<bits/stdc++.h>
using namespace std;
typedef long long LL;
const int inf=0x3f3f3f3f;
const int N=114514;
const LL INF = (1LL<<31)-1;
int d[N],n,m,s;//s为起始点
bool in_queue[N];
queue<int> q;
struct node{	
	int w,v;
	node(){}
	node(int vv,int ww){
		v=vv;
		w=ww;
	}
};
vector<node> g[N];
void spfa(int s){
	memset(in_queue,0,sizeof(in_queue));
	for(int i=1; i<=n; ++i)
		d[i] = INF;
	d[s]=0;
	in_queue[s]=1;
	q.push(s);
	while(!q.empty()){
		int v=q.front();
		q.pop();
		in_queue[v]=0;
		for(int i=0;i<g[v].size();i++){
			int x=g[v][i].v;
			if(d[x]>d[v]+g[v][i].w){
				d[x]=d[v]+g[v][i].w;
				if(!in_queue[x]){
					q.push(x);
					in_queue[x]=1;
				}
			}
		}
	}
}
int main(){
	cin>>n>>m>>s;
	for(int i=0;i<m;i++){
		int u,v,w;
		cin>>u>>v>>w;
		g[u].push_back(node(v,w));
		g[u].push_back(node(u,w));
	}
	spfa(s);
	for(int i=1;i<=n;i++){
		cout<<d[i]<<" ";
	}
}
```

------------

### 快读快写模板
```cpp
inline int read(){
	int f=1,s=0;char c=getchar();
	while(c<'0'||c>'9'){
      if(c=='-') f=-1;c=getchar();
   }
	while(c>='0'&&c<='9'){
      s=s*10+c-'0';c=getchar();
   }
	return f*s;
}//快读
inline void print(int x){
	if(x<0){
      putchar('-');print(-x);return;
   }
	if(x>9)
      print(x/10);
	putchar(x%10+'0');
}//快写
//用法:a=read();=cin>>a;
//    print(b);=cout<<b;
```

------------

### Dijkstra单源最短路模板
```cpp
#include<bits/stdc++.h>
using namespace std;
const int maxn=100010,maxm=500010;
struct edge{
    int to,dis,next;
}e[maxm];
int head[maxn],dis[maxn],cnt;
bool vis[maxn];
int n,m,s;
void add(int u,int v,int d){
    cnt++;
    e[cnt].dis=d;
    e[cnt].to=v;
    e[cnt].next=head[u];
    head[u]=cnt;
}
struct node{
    int dis,pos;
    bool operator<(const node &x)const{
        return x.dis<dis;
    }
};
priority_queue<node> q;
void dij(){
    dis[s]=0;
    q.push((node){0,s});
    while(!q.empty()){
        node tmp=q.top();
        q.pop();
        int x=tmp.pos,d=tmp.dis;
        if(vis[x])continue;
        vis[x]=1;
        for(int i=head[x];i;i=e[i].next){
            int y=e[i].to;
            if(dis[y]>dis[x]+e[i].dis){
                dis[y]=dis[x]+e[i].dis;
                if(!vis[y]){
                    q.push((node){dis[y],y});
                }
            }
        }
    }
}
int main(){
    cin>>n>>m>>s;
    for(int i=1;i<=n;i++){
        dis[i]=0x7fffffff;
    }
    for(int i=0;i<m;i++){
        int u,v,d;
        cin>>u>>v>>d;
        add(u,v,d);
    }
    dij();
    for(int i=1;i<=n;i++){
        cout<<dis[i]<<" ";
    }
    return 0;
}
```

------------

### floyd多源最短路模板
```cpp
#include<bits/stdc++.h>
using namespace std;
int d[110][110],n,m,u,v,w;
void floyd(){
	for(int i=1;i<=n;i++)d[i][i]=0;
	for(int k=1; k<=n; k++)
		for(int i=1; i<=n; i++)
			for(int j=1; j<=n; j++)
				d[i][j]=min(d[i][j],d[i][k]+d[k][j]);
}
int main() {
	memset(d,0x3f,sizeof(d));
	cin>>n>>m;
	for(int i=1; i<=m; i++){
		cin>>u>>v>>w;
		d[u][v]=w;
		d[v][u]=w;
	}
	floyd();
	for(int i=1; i<=n; i++) {
		for(int j=1; j<=n; j++)
			cout<<d[i][j]<<" ";
		cout<<endl;
	}
	return 0;
}
```

------------

### 高精度加法模板
```cpp
#include<bits/stdc++.h>
using namespace std;  
const int L=110;  
string add(string a,string b)//只限两个非负整数相加  
{  
    string ans;  
    int na[L]={0},nb[L]={0};  
    int la=a.size(),lb=b.size();  
    for(int i=0;i<la;i++) na[la-1-i]=a[i]-'0';  
    for(int i=0;i<lb;i++) nb[lb-1-i]=b[i]-'0';  
    int lmax=la>lb?la:lb;  
    for(int i=0;i<lmax;i++) na[i]+=nb[i],na[i+1]+=na[i]/10,na[i]%=10;  
    if(na[lmax]) lmax++;  
    for(int i=lmax-1;i>=0;i--) ans+=na[i]+'0';  
    return ans;  
}  
int main()  
{  
    string a,b;  
    cin>>a>>b；
    cout<<add(a,b)<<endl;  
    return 0;  
}
```

------------

### 高精度减法模板
```cpp
#include<bits/stdc++.h> 
using namespace std;  
const int L=110;  
string sub(string a,string b)//只限大的非负整数减小的非负整数  
{  
    string ans;  
    int na[L]={0},nb[L]={0};  
    int la=a.size(),lb=b.size();  
    for(int i=0;i<la;i++) na[la-1-i]=a[i]-'0';  
    for(int i=0;i<lb;i++) nb[lb-1-i]=b[i]-'0';  
    int lmax=la>lb?la:lb;  
    for(int i=0;i<lmax;i++)  
    {  
        na[i]-=nb[i];  
        if(na[i]<0) na[i]+=10,na[i+1]--;  
    }  
    while(!na[--lmax]&&lmax>0)  ;lmax++;  
    for(int i=lmax-1;i>=0;i--) ans+=na[i]+'0';  
    return ans;  
}  
int main()  
{  
    string a,b;  
    cin>>a>>b;
    cout<<sub(a,b)<<endl;  
    return 0;  
}
```

------------

### 高精度乘法模板
```cpp
#include<bits/stdc++.h>
using namespace std;
const int L=110;
string mul(string a,string b) { //高精度乘法a,b,均为非负整数
	string s;
	int na[L],nb[L],nc[L],La=a.size(),Lb=b.size();
	fill(na,na+L,0);
	fill(nb,nb+L,0);
	fill(nc,nc+L,0);
	for(int i=La-1; i>=0; i--) na[La-i]=a[i]-'0';
	for(int i=Lb-1; i>=0; i--) nb[Lb-i]=b[i]-'0';
	for(int i=1; i<=La; i++)
		for(int j=1; j<=Lb; j++)
			nc[i+j-1]+=na[i]*nb[j];
	for(int i=1; i<=La+Lb; i++)
		nc[i+1]+=nc[i]/10,nc[i]%=10;
	if(nc[La+Lb]) s+=nc[La+Lb]+'0';
	for(int i=La+Lb-1; i>=1; i--)
		s+=nc[i]+'0';
	return s;
}
int main() {
	string a,b;
	cin>>a>>b;
	cout<<mul(a,b)<<endl;
	return 0;
}
```

------------

### 高精度除法/取余模板
```cpp
#include<bits/stdc++.h>
using namespace std;
const int L=110;
int sub(int *a,int *b,int La,int Lb) {
	if(La<Lb) return -1;
	if(La==Lb) {
		for(int i=La-1; i>=0; i--)
			if(a[i]>b[i]) break;
			else if(a[i]<b[i]) return -1;

	}
	for(int i=0; i<La; i++) {
		a[i]-=b[i];
		if(a[i]<0) a[i]+=10,a[i+1]--;
	}
	for(int i=La-1; i>=0; i--)
		if(a[i]) return i+1;
	return 0;

}
string div(string n1,string n2,int nn) {
	string s,v;
	int a[L],b[L],r[L],La=n1.size(),Lb=n2.size(),i,tp=La;
	fill(a,a+L,0);
	fill(b,b+L,0);
	fill(r,r+L,0);
	for(i=La-1; i>=0; i--) a[La-1-i]=n1[i]-'0';
	for(i=Lb-1; i>=0; i--) b[Lb-1-i]=n2[i]-'0';
	if(La<Lb || (La==Lb && n1<n2)) {
		return n1;
	}
	int t=La-Lb;
	for(int i=La-1; i>=0; i--)
		if(i>=t) b[i]=b[i-t];
		else b[i]=0;
	Lb=La;
	for(int j=0; j<=t; j++) {
		int temp;
		while((temp=sub(a,b+j,La,Lb-j))>=0) {
			La=temp;
			r[t-j]++;
		}
	}
	for(i=0; i<L-10; i++) r[i+1]+=r[i]/10,r[i]%=10;
	while(!r[i]) i--;
	while(i>=0) s+=r[i--]+'0';
	i=tp;
	while(!a[i]) i--;
	while(i>=0) v+=a[i--]+'0';
	if(v.empty()) v="0";
	if(nn==1) return s;
	if(nn==2) return v;
}
int main() {
	string a,b;
	cin>>a>>b;
	cout<<div(a,b,1)<<endl;
	return 0;
}
```

------------

### 高精度阶乘模板
```cpp
#include<bits/stdc++.h>
using namespace std;
const int L=100005;
int a[L];
string fac(int n) {
	string ans;
	if(n==0) return "1";
	fill(a,a+L,0);
	int s=0,m=n;
	while(m) a[++s]=m%10,m/=10;
	for(int i=n-1; i>=2; i--) {
		int w=0;
		for(int j=1; j<=s; j++) a[j]=a[j]*i+w,w=a[j]/10,a[j]=a[j]%10;
		while(w) a[++s]=w%10,w/=10;
	}
	while(!a[s]) s--;
	while(s>=1) ans+=a[s--]+'0';
	return ans;
}
int main() {
	int n;
	cin>>n;
	cout<<fac(n)<<endl;
	return 0;
}
```

------------

### 高精度幂模板
```cpp
#include<bits/stdc++.h>
using namespace std;
#define L(x) (1 << (x))
const double PI = acos(-1.0);
const int Maxn = 133015;
double ax[Maxn], ay[Maxn], bx[Maxn], by[Maxn];
char sa[Maxn/2],sb[Maxn/2];
int sum[Maxn];
int x1[Maxn],x2[Maxn];
int revv(int x, int bits) {
	int ret = 0;
	for (int i = 0; i < bits; i++) {
		ret <<= 1;
		ret |= x & 1;
		x >>= 1;
	}
	return ret;
}
void fft(double * a, double * b, int n, bool rev) {
	int bits = 0;
	while (1 << bits < n) ++bits;
	for (int i = 0; i < n; i++) {
		int j = revv(i, bits);
		if (i < j)
			swap(a[i], a[j]), swap(b[i], b[j]);
	}
	for (int len = 2; len <= n; len <<= 1) {
		int half = len >> 1;
		double wmx = cos(2 * PI / len), wmy = sin(2 * PI / len);
		if (rev) wmy = -wmy;
		for (int i = 0; i < n; i += len) {
			double wx = 1, wy = 0;
			for (int j = 0; j < half; j++) {
				double cx = a[i + j], cy = b[i + j];
				double dx = a[i + j + half], dy = b[i + j + half];
				double ex = dx * wx - dy * wy, ey = dx * wy + dy * wx;
				a[i + j] = cx + ex, b[i + j] = cy + ey;
				a[i + j + half] = cx - ex, b[i + j + half] = cy - ey;
				double wnx = wx * wmx - wy * wmy, wny = wx * wmy + wy * wmx;
				wx = wnx, wy = wny;
			}
		}
	}
	if (rev) {
		for (int i = 0; i < n; i++)
			a[i] /= n, b[i] /= n;
	}
}
int solve(int a[],int na,int b[],int nb,int ans[]) {
	int len = max(na, nb), ln;
	for(ln=0; L(ln)<len; ++ln);
	len=L(++ln);
	for (int i = 0; i < len ; ++i) {
		if (i >= na) ax[i] = 0, ay[i] =0;
		else ax[i] = a[i], ay[i] = 0;
	}
	fft(ax, ay, len, 0);
	for (int i = 0; i < len; ++i) {
		if (i >= nb) bx[i] = 0, by[i] = 0;
		else bx[i] = b[i], by[i] = 0;
	}
	fft(bx, by, len, 0);
	for (int i = 0; i < len; ++i) {
		double cx = ax[i] * bx[i] - ay[i] * by[i];
		double cy = ax[i] * by[i] + ay[i] * bx[i];
		ax[i] = cx, ay[i] = cy;
	}
	fft(ax, ay, len, 1);
	for (int i = 0; i < len; ++i)
		ans[i] = (int)(ax[i] + 0.5);
	return len;
}
string mul(string sa,string sb) {
	int l1,l2,l;
	int i;
	string ans;
	memset(sum, 0, sizeof(sum));
	l1 = sa.size();
	l2 = sb.size();
	for(i = 0; i < l1; i++)
		x1[i] = sa[l1 - i - 1]-'0';
	for(i = 0; i < l2; i++)
		x2[i] = sb[l2-i-1]-'0';
	l = solve(x1, l1, x2, l2, sum);
	for(i = 0; i<l || sum[i] >= 10; i++) {
		sum[i + 1] += sum[i] / 10;
		sum[i] %= 10;
	}
	l = i;
	while(sum[l] <= 0 && l>0)    l--;
	for(i = l; i >= 0; i--)    ans+=sum[i] + '0';
	return ans;
}
string Pow(string a,int n) {
	if(n==1) return a;
	if(n&1) return mul(Pow(a,n-1),a);
	string ans=Pow(a,n/2);
	return mul(ans,ans);
}
int main() {
	cin.sync_with_stdio(false);
	string a;
	int b;
	cin>>a>>b;
	cout<<Pow(a,b)<<endl;
	return 0;
}
```

------------

### 高精度进制转换模板
```cpp
#include<iostream>
#include<algorithm>
using namespace std;
bool judge(string s) {
	for(int i=0; i<s.size(); i++)
		if(s[i]!='0') return 1;
	return 0;
}
string solve(string s,int n,int m) { //n进制转m进制只限0-9进制，若涉及带字母的进制，稍作修改即可
	string r,ans;
	int d=0;
	if(!judge(s)) return "0";
	while(judge(s)) {
		for(int i=0; i<s.size(); i++) {
			r+=(d*n+s[i]-'0')/m+'0';
			d=(d*n+(s[i]-'0'))%m;
		}
		s=r;
		r="";
		ans+=d+'0';
		d=0;
	}
	reverse(ans.begin(),ans.end());
	return ans;
}
int main() {
	string s;
	int n,m;
	cin>>s>>n>>m;
	cout<<solve(s,n,m)<<endl;
	return 0;
}
```

------------

### 高精度平方根模板
```cpp
#include<bits/stdc++.h>
using namespace std;
const int L=2015;
string add(string a,string b) {
	string ans;
	int na[L]= {0},nb[L]= {0};
	int la=a.size(),lb=b.size();
	for(int i=0; i<la; i++) na[la-1-i]=a[i]-'0';
	for(int i=0; i<lb; i++) nb[lb-1-i]=b[i]-'0';
	int lmax=la>lb?la:lb;
	for(int i=0; i<lmax; i++) na[i]+=nb[i],na[i+1]+=na[i]/10,na[i]%=10;
	if(na[lmax]) lmax++;
	for(int i=lmax-1; i>=0; i--) ans+=na[i]+'0';
	return ans;
}
string sub(string a,string b) {
	string ans;
	int na[L]= {0},nb[L]= {0};
	int la=a.size(),lb=b.size();
	for(int i=0; i<la; i++) na[la-1-i]=a[i]-'0';
	for(int i=0; i<lb; i++) nb[lb-1-i]=b[i]-'0';
	int lmax=la>lb?la:lb;
	for(int i=0; i<lmax; i++) {
		na[i]-=nb[i];
		if(na[i]<0) na[i]+=10,na[i+1]--;
	}
	while(!na[--lmax]&&lmax>0)  ;
	lmax++;
	for(int i=lmax-1; i>=0; i--) ans+=na[i]+'0';
	return ans;
}
string mul(string a,string b) {
	string s;
	int na[L],nb[L],nc[L],La=a.size(),Lb=b.size();
	fill(na,na+L,0);
	fill(nb,nb+L,0);
	fill(nc,nc+L,0);
	for(int i=La-1; i>=0; i--) na[La-i]=a[i]-'0';
	for(int i=Lb-1; i>=0; i--) nb[Lb-i]=b[i]-'0';
	for(int i=1; i<=La; i++)
		for(int j=1; j<=Lb; j++)
			nc[i+j-1]+=na[i]*nb[j];
	for(int i=1; i<=La+Lb; i++)
		nc[i+1]+=nc[i]/10,nc[i]%=10;
	if(nc[La+Lb]) s+=nc[La+Lb]+'0';
	for(int i=La+Lb-1; i>=1; i--)
		s+=nc[i]+'0';
	return s;
}
int sub(int *a,int *b,int La,int Lb) {
	if(La<Lb) return -1;
	if(La==Lb) {
		for(int i=La-1; i>=0; i--)
			if(a[i]>b[i]) break;
			else if(a[i]<b[i]) return -1;

	}
	for(int i=0; i<La; i++) {
		a[i]-=b[i];
		if(a[i]<0) a[i]+=10,a[i+1]--;
	}
	for(int i=La-1; i>=0; i--)
		if(a[i]) return i+1;
	return 0;

}
string div(string n1,string n2,int nn) {
	string s,v;
	int a[L],b[L],r[L],La=n1.size(),Lb=n2.size(),i,tp=La;
	fill(a,a+L,0);
	fill(b,b+L,0);
	fill(r,r+L,0);
	for(i=La-1; i>=0; i--) a[La-1-i]=n1[i]-'0';
	for(i=Lb-1; i>=0; i--) b[Lb-1-i]=n2[i]-'0';
	if(La<Lb || (La==Lb && n1<n2)) {
		return n1;
	}
	int t=La-Lb;
	for(int i=La-1; i>=0; i--)
		if(i>=t) b[i]=b[i-t];
		else b[i]=0;
	Lb=La;
	for(int j=0; j<=t; j++) {
		int temp;
		while((temp=sub(a,b+j,La,Lb-j))>=0) {
			La=temp;
			r[t-j]++;
		}
	}
	for(i=0; i<L-10; i++) r[i+1]+=r[i]/10,r[i]%=10;
	while(!r[i]) i--;
	while(i>=0) s+=r[i--]+'0';
	
	i=tp;
	while(!a[i]) i--;
	while(i>=0) v+=a[i--]+'0';
	if(v.empty()) v="0";
	
	if(nn==1) return s;
	if(nn==2) return v;
}
bool cmp(string a,string b) {
	if(a.size()<b.size()) return 1;
	if(a.size()==b.size()&&a<=b) return 1;
	return 0;
}
string BigInterSqrt(string n) {
	string l="1",r=n,mid,ans;
	while(cmp(l,r)) {
		mid=div(add(l,r),"2",1);
		if(cmp(mul(mid,mid),n)) ans=mid,l=add(mid,"1");
		else r=sub(mid,"1");
	}
	return ans;
}
string DeletePreZero(string s) {
	int i;
	for(i=0; i<s.size(); i++)
		if(s[i]!='0') break;
	return s.substr(i);
}
int main() {
	string n;
	int t;
	cin>>n;
	n=DeletePreZero(n);
	cout<<BigInterSqrt(n)<<endl;
	return 0;
}
```


------------

### 最小生成树模板
```cpp
#include <iostream>
#include <cstdio>
#include <cstring>
using namespace std;
int n,m;
const int N = 510;
const int INF = 0x3f3f3f3f;
int g[N][N],dist[N];
bool st[N];
int prim() {
	int res = 0;
	memset(dist,0x3f,sizeof(dist));
	for(int i=0; i<n; i++) {
		int t = -1;
		for(int j=1; j<=n; j++) {
			if(!st[j] && (t == -1 || dist[t] > dist[j])) {
				t = j;
			}
		}
		if(i && dist[t] == INF) {
			return INF;
		}
		st[t] = true;
		if(i) {
			res += dist[t];
		}
		for(int j=1; j<=n; j++) {
			dist[j] = min(dist[j],g[t][j]);
		}
	}
	return res;
}
int main() {
	memset(g,0x3f,sizeof(g));
	scanf("%d%d",&n,&m);
	for(int i=1; i<=m; i++) {
		int u,v,w;
		scanf("%d%d%d",&u,&v,&w);
		g[u][v] = g[v][u] = min(g[u][v],w);
	}
	int res =  prim();
	if(res == INF) {
		printf("impossible");
	} else {
		printf("%d",res);
	}
	return 0;
}

```

# 完结撒花！！！
#### 有错误或更优解欢迎指出