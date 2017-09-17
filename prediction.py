import json
import subprocess
from oauth2client.client import GoogleCredentials
from googleapiclient import discovery


PROJECT_ID = subprocess.check_output(
    "gcloud config list project --format 'value(core.project)'",
    shell=True
).rstrip()
MODEL_NAME = "tictactoe"

print("Calling ML model on Cloud ML Engine...")
credentials = GoogleCredentials.get_application_default()
ml = discovery.build("ml", "v1", credentials=credentials)
data = {
    "instances": [
        {"x": [0, 0, 0, 0, 0, 0, 0, 0, 0], "key": 0}
    ]
}

req = ml.projects().predict(
    body=data,
    name="projects/{0}/models/{1}".format(PROJECT_ID, MODEL_NAME)
)

print(json.dumps(req.execute(), indent=2))
