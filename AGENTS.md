# ISVL++

> 공통 규칙은 루트 `../AGENTS.md`를 따른다. 이 문서는 ISVL++ 전용 규칙이다.

## 1. 개요

- 원본: https://github.com/ISVL119/isvl-plusplus (클론 시점 커밋 `ba197c8`, 2026-05-22)
- 논문: `isvl++.pdf` (저장소 포함) — VAND 4.0 Industrial Track (MVTec AD 2)
- 방식: INP-Former 기반이며 **카테고리마다 모델을 따로 학습**한다(one-class-per-model).
  - backbone: DINOv3 ViT-H/16+
  - 이상 합성: SAM 3로 foreground를 추출하고, 실패하면 CPR 방식으로 대체한다.
  - threshold: mean + k·std에서 k를 DTD 기반 합성 이상으로 튜닝한다.
  - 후처리: HQ-SAM 기반 SAM-Finer (RoBiS 아이디어)
- 라이선스: 원본 코드는 **CC BY-NC 4.0** (비상업적 용도). `LICENSE`, `NOTICE` 참고

## 2. 환경 (uv)

- 원본 환경: conda, Python **3.11.13**, torch 2.5.1 / torchvision 0.20.1 / torchaudio 2.5.1 (**cu118**), `requirements.txt`
- **B200 문제**: torch 2.5.1 cu118은 sm_100(B200)을 지원하지 않는다. 원본 버전 그대로는 GPU에서 실행되지 않을 가능성이 높다.
- uv 전환 방침:
  1. `pyproject.toml`을 새로 만들고 `requirements.txt`의 패키지 버전을 그대로 옮긴다.
  2. torch 계열만 B200 호환 버전(cu128 이상)으로 올린다. 정확한 버전은 `미정`.
  3. 바꾼 버전과 그로 인한 결과 차이는 "로컬 변경 내역"에 기록한다.
- 유의할 고정 버전: `numpy==1.26.4`, `timm==0.9.12`, `transformers==4.57.0`, `opencv-python-headless==4.6.0.66`, `imagecodecs==2024.6.1`

## 3. 데이터 및 가중치

| 항목 | 경로 | 비고 |
|---|---|---|
| MVTec AD 2 | `./datasets/mvtec_ad_2` | 학습 및 평가 |
| DTD | `./datasets/dtd` | threshold 선택용 |
| Anomaly bank | `./sam3-260512/data/anomaly` | 저장소에 포함 |
| DINOv3 | `./backbones/weights/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth` | Meta 신청 필요 |
| SAM 3 | `./backbones/weights/sam3.pt` | HF `facebook/sam3` |
| HQ-SAM | `./backbones/weights/sam_hq_vit_h.pth` | HF `lkeab/hq-sam` |
| DenseNet | 실행 시 HF에서 자동 다운로드 | |
| 학습된 체크포인트 | `./saved_results/INP-Former-Multi-Class_dataset=Mvtec_ad_2_Encoder=dinov3_vith16plus_Resize=512_INP_num=6_Seg=512` | Google Drive(README) |
| 제공 threshold | `./select_thresholds_json/final_thresholds_by_split.json` | |

- **디스크 사용량**: 전체 과정에서 약 **160 GB**를 쓴다(압축 데이터 약 30 GB 포함). 실행 전에 여유 공간을 확인한다.

## 4. 실행

전체 파이프라인은 `reproduce_result.sh`에 있다. 단계와 명령은 다음과 같다.

| 단계 | 명령 |
|---|---|
| 1. 데이터 분할 | `split_crop.py`, `copy_images.py` |
| 2. 이상 합성 | `cd sam3-260512` → `1_foreground_extraction.py`, `2_generate_synthetic_anomaly.py` → 결과를 `../datasets/`로 이동 |
| 3. 학습 (카테고리별) | `isvl_train_and_test++.py --use_synth_anomalies --item_list <cat> --phase train --synth_anomaly_root ./datasets/synthesized_mvtecad2_1024rgbl/<cat>` |
| 4. 추론 | `isvl_train_and_test++.py --item_list <cat> --phase test` |
| 5. threshold 선택 | `isvl_select_threshold.py --item_list <cat> --phase true_val --synthetic_k_enable --illumination_calibration_enable` |
| 6. 이진화 및 후처리 | `threshold_map.py`, `SAMHQ-Finer.py` |
| 7. 제출 | `convert_tiff_to_float16.py`, `check_and_prepare_data_for_upload.py ./results/` → `results.tar.gz` |

- 카테고리: can, fabric, fruit_jelly, sheet_metal, vial, walnuts, wallplugs, rice
- 모든 Python 명령은 `uv run python ...`으로 실행한다.

### 빠른 검증 (추론만)

1. 체크포인트를 `saved_results/...`에 둔다.
2. `uv run python split_crop.py`를 실행한다.
3. 4단계(추론)를 실행한다.
4. 5단계는 건너뛰고 `select_thresholds_json/final_thresholds_by_split.json`을 작업 디렉터리의 `./final_thresholds_by_split.json`으로 복사한다.
5. 6~7단계를 실행한다.

## 5. 주의사항

- **threshold가 하드웨어마다 달라지는 문제**: 원본 저자 보고에 따르면 RTX 4090과 3090에서 `isvl_select_threshold.py` 결과가 달랐다. B200에서도 다를 수 있으므로 **재현할 때는 제공된 threshold JSON을 쓴다**.
- 출력 파일 다수(`final_thresholds_by_split.json`, `results/`, `results.tar.gz`)가 현재 작업 디렉터리에 생성된다. 항상 `ISVL_plus_plus/`에서 실행한다.

## 6. 로컬 변경 내역

| 날짜 | 파일 | 내용 | 이유 |
|---|---|---|---|
| - | - | 없음 | - |

## 7. 진행 상태 (1단계) — 추후 진행

> 2026-10-16: 사용자 지시로 보류. SuperADD, BALM을 먼저 진행한다.

- [ ] `pyproject.toml` 작성 및 `uv sync` 성공
- [ ] B200 호환 torch 버전 확정
- [ ] 데이터셋 및 가중치 준비
- [ ] 추론 재현 (제공 체크포인트와 threshold 사용)
