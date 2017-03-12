rm -rf public
if [ ! -d "../jojo23333.github.io" ]; then
git clone git@github.com:jojo23333/jojo23333.github.io.git ..
fi
hexo clean
hexo g
cd  ../jojo23333.github.io
git pull
mv  -rf ../Blog/public/*  ./
git add -A
git commit -m"Update"
git push origin master
