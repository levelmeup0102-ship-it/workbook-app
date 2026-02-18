# 레벨미업 워크북 웹앱

## 파일 구조
```
main.py          ← 웹 서버
pipeline.py      ← AI 워크북 엔진
template.html    ← 워크북 디자인
requirements.txt ← Python 패키지
Procfile         ← Railway 실행 설정
static/
  index.html     ← 프론트엔드 (사용자 화면)
  manifest.json  ← PWA 설정
  icon-192.png   ← 앱 아이콘
```

## Railway 배포 방법
1. GitHub에 이 저장소 생성
2. railway.app → New Project → Deploy from GitHub repo
3. 환경변수 설정:
   - `APP_PASSWORD` = 원하는 비밀번호
   - `ANTHROPIC_API_KEY` = sk-ant-api03-...
4. Deploy!
