podman build -t inference-model .
podman tag inference-model asia-southeast2-docker.pkg.dev/capstone-project-c242-ps363/backend/inference-model
podman push asia-southeast2-docker.pkg.dev/capstone-project-c242-ps363/backend/inference-model
gcloud run deploy inference-model --image=asia-southeast2-docker.pkg.dev/capstone-project-c242-ps363/backend/inference-model --platform=managed --region=asia-southeast2 --allow-unauthenticated
