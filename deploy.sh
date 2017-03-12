rm -rf public
git clone git@github.com:jojo23333/jojo23333.github.io.git ./public
hexo clean
hexo g
cd ./public
git add -A
git commit -m"Update"
git push origin master
