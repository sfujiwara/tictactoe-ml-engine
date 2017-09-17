## バケットの作成

```bash
PROJECT_ID=`gcloud config list project --format "value(core.project)"`
gsutil mb -l us-central1 "gs://${PROJECT_ID}-ml"
gsutil mb -l us-central1 "gs://${PROJECT_ID}-ml-staging"
```

## Cloud ML Engine にジョブを投げる

```bash
JOB_NAME="tictactoe`date '+%Y%m%d%H%M%S'`"
PROJECT_ID=`gcloud config list project --format "value(core.project)"`
CSV_FILE="gs://${PROJECT_ID}-ml/data/tictactoe.csv"
gsutil cp data/tictactoe.csv gs://${PROJECT_ID}-ml/data/
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