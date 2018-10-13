git clone git@github.com:KazukiOnodera/PLAsTiCC-2018.git
mkdir PLAsTiCC-2018/input
mkdir PLAsTiCC-2018/output
mkdir PLAsTiCC-2018/data
mkdir PLAsTiCC-2018/feature
mkdir PLAsTiCC-2018/py
mkdir PLAsTiCC-2018/jn
mkdir PLAsTiCC-2018/py/LOG
cd PLAsTiCC-2018/input
kaggle competitions download -c PLAsTiCC-2018
cd ../
echo *.DS_Store > .gitignore
echo ~$*.xls* >> .gitignore
echo feature/ >> .gitignore
echo input/ >> .gitignore
echo output/ >> .gitignore
echo data/ >> .gitignore
echo external/ >> .gitignore
echo jn/.ipynb_checkpoints >> .gitignore
echo py/*.model >> .gitignore
echo py/*.p >> .gitignore
echo py/__pycache__/* >> .gitignore
echo py/~$*.xls* >> .gitignore
cat .gitignore

gitupdate
