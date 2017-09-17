# TicTacToe with TensorFlow and Cloud ML Engine

## バケットの作成

```bash
PROJECT_ID=`gcloud config list project --format "value(core.project)"`
gsutil mb -l us-central1 "gs://${PROJECT_ID}-ml"
gsutil mb -l us-central1 "gs://${PROJECT_ID}-ml-staging"
```

## リポジトリのクローン

```bash
git clone https://github.com/sfujiwara/tictactoe-ml-engine.git
cd tictactoe-ml-engine
```

## データを Cloud Storage にコピー

```bash
PROJECT_ID=`gcloud config list project --format "value(core.project)"`
gsutil cp data/tictactoe.csv gs://${PROJECT_ID}-ml/data/
```

## Cloud ML Engine にジョブを投げる

```bash
PROJECT_ID=`gcloud config list project --format "value(core.project)"`
JOB_NAME="tictactoe`date '+%Y%m%d%H%M%S'`"
CSV_FILE="gs://${PROJECT_ID}-ml/data/tictactoe.csv"
```

```bash
gcloud ml-engine jobs submit training ${JOB_NAME} \
  --package-path=trainer \
  --module-name=trainer.task \
  --staging-bucket="gs://${PROJECT_ID}-ml-staging" \
  --region=us-central1 \
  --config=config.yaml \
  --runtime-version 1.2 \
  -- \
  --output_path="gs://${PROJECT_ID}-ml/tictactoe/${JOB_NAME}" \
  --csv_file=${CSV_FILE}
```