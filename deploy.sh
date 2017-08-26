rm -rf public
if [ ! -d "../jojo23333.github.io" ]; then
git clone git@github.com:jojo23333/jojo23333.github.io.git ..
fi
hexo clean
hexo g
cd  ../jojo23333.github.io
git pull
rm  -rf ./*
mv  -f ../Blog/public/*  ./
echo jonassenli.com > CNAME #Used for costom sitename
git add -A
git commit -m"Update"
git push origin master
