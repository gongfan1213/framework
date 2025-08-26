### 09 CI/CD（GitHub Actions）与发布流程

#### 目标
- 代码提交→构建镜像→推送Registry→部署到K8s（Helm）→自动回滚。

#### GitHub Actions 工作流（示例）
```yaml
name: ci-cd
on:
  push:
    branches: [ main ]
jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: docker/setup-buildx-action@v3
      - uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      - uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: ghcr.io/owner/llm-infer:${{ github.sha }}
      - name: Helm Deploy
        uses: azure/k8s-deploy@v5
        with:
          namespace: llm
          manifests: |
            chart/templates/deployment.yaml
            chart/templates/service.yaml
          images: |
            ghcr.io/owner/llm-infer:${{ github.sha }}
```

#### 密钥管理
- 使用 GitHub Secrets 存储 K8s kubeconfig/Registry 凭据；
- 或在Runner上配置OIDC到云提供商。


