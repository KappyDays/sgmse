# apt-get install update -y
# apt-get install git
rm -rf .git
git init
git config --global user.email "rkdfuf2wkd@naver.com"
git config --global user.name "KappyDays"
git remote add origin https://github.com/KappyDays/sgmse.git
# 여기서 merge를 위해 중복 파일 제거할 것 (밑 명령어 실행 후 중복파일 확인하고 제거해도 됨)
## 중복 파일 존재 에러 예시 -> error: The following untracked working tree files would be overwritten by merge: ##
# git pull origin main
