# HOT3D eval_metric 서버 실행 패키지

이 디렉터리 하나를 다른 Linux 서버로 복사하면 기본 입력 2개에 대한 평가를 실행할 수 있다. 원본 프로젝트의 사용자 홈 경로에 의존하지 않도록 입력, 객체 데이터, MANO 모델 경로를 패키지 기준 상대경로로 변경했다.

## 포함 항목

- `hot3d/`: 평가 코드와 로컬 Python 의존 코드
- `inputs/`: 기본 seen/unseen 평가 결과 pickle 2개
- `data/obj.pkl`, `data/object_mesh/`: 32개 객체의 점군 메타데이터와 메시
- `models/MANO_LEFT.pkl`, `models/MANO_RIGHT.pkl`: 평가에 필요한 MANO 모델
- `requirements.txt`: headless metric 평가용 필수 패키지
- `requirements-visualize.txt`: Rerun/Open3D 시각화까지 할 때의 추가 패키지

MANO 파일의 사용 및 재배포 조건은 `models/LICENSE.txt`를 확인한다.

## 권장 환경

- Linux x86_64
- Python 3.10
- 충분한 RAM과 디스크 공간
- GPU는 필수가 아니지만, CUDA GPU가 있으면 아래 GPU 평가기를 권장한다.

## 설치 및 기본 평가

```bash
tar -xzf eval.tar.gz
cd eval
python3 check_package.py
./setup.sh
./run_eval.sh
```

결과는 실행 디렉터리에 `new_metrics_per_frame_*.csv`와 `new_metrics_summary_*.md`로 저장된다.

CUDA 전용 PyTorch wheel이 필요하면 `setup.sh` 실행 전에 서버 CUDA 버전에 맞는 PyTorch를 별도로 설치한 뒤 나머지 requirements를 설치한다.

## GPU 평가

`eval_metric_gpu.py`는 원본의 입력/출력과 metric threshold를 그대로 쓰면서
ID closest-point, mesh 내부 판정, IV containment, CR, fingertip contact를
PyTorch CUDA kernel로 계산한다.

```bash
./run_eval_gpu.sh \
  --device cuda:0 \
  --input my_seen.pkl \
  --new-metric-csv-output outputs/per_frame_gpu.csv \
  --new-metric-md-output outputs/summary_gpu.md
```

Conda 환경을 사용할 때는 해당 환경을 활성화하거나 Python 경로를 지정한다.

```bash
PYTHON_BIN=/path/to/conda/env/bin/python ./run_eval_gpu.sh --device cuda:0
```

기본 `float64` geometry와 CPU-reference MANO 조합은 CPU 결과 호환 모드다.
더 빠른 처리가 필요하고 마지막 소수점 수준의 차이를 허용할 수 있다면 다음
옵션을 추가한다.

```bash
./run_eval_gpu.sh --gpu-dtype float32 --gpu-mano --device cuda:0
```

Part Acc., PCP, G2C, GSR만 빠르게 확인하려면 `--quick`을 사용한다. 이 모드는
전체 sequence의 contact를 탐색하지만 ID는 GSR에 필요한 마지막 3 frame에서만
계산하고, IV/가속도/diversity 계산은 생략한다.

```bash
./run_eval_gpu.sh \
  --device cuda:0 \
  --quick \
  --input my_seen.pkl \
  --new-metric-md-output outputs/quick_summary.md
```

VRAM이 부족하면 `--gpu-query-chunk 128 --gpu-face-chunk 1024`처럼 chunk를
줄인다. 기본값은 각각 256과 2048이다.

## 입력과 출력 지정

`--input`은 반복해서 사용할 수 있다. 상대경로는 `--input-dir`을 기준으로 찾는다.

```bash
./run_eval.sh \
  --input my_seen.pkl \
  --input my_unseen.pkl \
  --input-dir /data/eval_inputs \
  --obj-pkl /data/hot3d/obj.pkl \
  --new-metric-csv-output outputs/per_frame.csv \
  --new-metric-md-output outputs/summary.md
```

일부 샘플만 빠르게 확인하려면 다음처럼 실행한다.

```bash
./run_eval.sh --sample-idx 0 1 2 --fast-penetration-metric
```

기본 MANO 모델 디렉터리를 바꾸려면 환경변수를 사용한다.

```bash
MANO_MODEL_DIR=/data/mano/models ./run_eval.sh --input result.pkl
```

## 시각화 기능

서버에서 Rerun/Open3D 기반 옵션도 사용할 경우 다음을 추가 설치한다.

```bash
.venv/bin/python -m pip install -r requirements-visualize.txt
```

일반 metric CSV/Markdown 생성에는 이 선택 의존성이 필요하지 않다.
