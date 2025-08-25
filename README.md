<div>
<h1 style="color: #0077B6; font-size: 3.5em; font-weight: bold; text-shadow: 2px 2px 4px #cccccc;">
    잡어드림 🐟🏭
  </h1>
  <p style="font-size: 1.2em;">
    <strong>어부와 공장을 잇는 스마트한 혼획물 거래 플랫폼</strong>
  </p>
</div>

<br>
📖 프로젝트 소개
잡어드림은 어업 과정에서 발생하는 소중한 **혼획물 (Bycatch)**이 버려지지 않도록, <br>
이를 필요로 하는 공장 및 가공업체와 어부를 직접 연결하는 똑똑한 거래 플랫폼입니다.
<br>

<br>
✨ 주요 기능

👤 회원 관리
어부와 공장 사용자를 위한 역할 기반(RBAC) 회원가입 및 JWT 기반 인증/인가 시스템을 제공합니다.

📝 혼획물 등록 및 관리
어부가 직접 잡은 혼획물의 종류, 수량, 희망 가격, 사진 등을 손쉽게 등록하고 관리(CRUD)할 수 있습니다.

🤖 AI 기반 어종 분석
사용자가 혼획물 영상을 업로드하면, AI 모델이 이를 실시간으로 분석하여 어종을 자동으로 식별하고 정보를 제공합니다.

🤝 거래 매칭 시스템
공장 사용자는 지역, 어종, 수량 등 다양한 조건으로 등록된 혼획물을 검색하고, 원하는 상품에 구매 제안을 보낼 수 있습니다.

📈 실시간 거래 관리
구매 제안부터 수락, 거절에 이르기까지 거래의 모든 과정을 실시간으로 추적하고 상태를 관리합니다.

🔔 푸시 알림
새로운 거래 제안이나 중요한 상태 변경이 발생했을 때, 관련 사용자에게 즉시 알림을 전송하여 빠른 소통을 돕습니다. (향후 확장 예정)


## 출처
본 프로젝트의 AI 어종 분석 기능은 아래 오픈소스 프로젝트의 코드와 사전 학습된 모델을 기반으로 구현되었습니다.

* **원본 저장소 (Codebase Reference)**
    * [fishial/fish-identification](https://github.com/fishial/fish-identification)

* **사전 학습 모델 (Pre-trained Model)**
    * **Model Name**: `Fish Classification - beitv2_base_patch16_224`
    * **Specifications**:
        * Classes: `640`
        * Embed: `512`
        * Format: `torchscript`
    * **Download**: 모델 파일과 데이터셋은 위 원본 저장소에서 직접 다운로드하실 수 있습니다.






