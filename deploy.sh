podman build -t inference-model .
podman tag inference-model asia-southeast2-docker.pkg.dev/capstone-project-c242-ps363/backend/inference-model
podman push asia-southeast2-docker.pkg.dev/capstone-project-c242-ps363/backend/inference-model
gcloud run deploy inference-model --image=asia-southeast2-docker.pkg.dev/capstone-project-c242-ps363/backend/inference-model --platform=managed --region=asia-southeast2 --allow-unauthenticated --memory=1Gi

# update nginx
url=$(gcloud run services describe inference-model --platform=managed --region=asia-southeast2 --format='value(status.url)')
url=${url#https://}
echo "setup $url to nginx"
gcloud compute ssh nginx -- -p 22 "sudo sed 's/proxy_pass.*app;/proxy_pass https:\/\/$url;/g' -i /etc/nginx/conf.d/signlang.conf; sudo nginx -t; sudo systemctl reload nginx"
