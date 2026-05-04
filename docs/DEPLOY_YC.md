# CI/CD в Yandex Cloud (Serverless Containers)

Этот проект разворачивается через GitHub Actions:

- `CI` (`.github/workflows/ci.yml`) — проверки кода и конфигов.
- `CD` (`.github/workflows/cd.yml`) — сборка Docker-образа, push в Yandex Container Registry и деплой новой ревизии в Serverless Container.

## Используемые параметры

- `YC_REGISTRY_ID`: `crp80kqtu9obombh8iv7`
- `YC_SERVERLESS_CONTAINER_ID`: `bba89gp29g45b9fbutt4`
- `YC_CLOUD_ID`: `b1gp1s5aref35kjosaqt`
- `YC_FOLDER_ID`: `b1g6kqkdbv9d7o7qle7p`
- `YC_BUCKET_NAME`: `cbr-predictor-veb`

## Секреты GitHub Actions

В репозитории нужно добавить secrets:

- `YC_SA_KEY_JSON` — JSON-ключ сервисного аккаунта.
- `YC_CLOUD_ID`
- `YC_FOLDER_ID`
- `YC_REGISTRY_ID`
- `YC_SERVERLESS_CONTAINER_ID`
- `YC_BUCKET_NAME`
- `YC_REGION` (`ru-central1`, опционально)

## Поток деплоя

1. Merge в `main`.
2. Workflow `CD to Yandex Serverless Containers`:
   - логинится в `cr.yandex`,
   - собирает образ `cbr-predictor:${GITHUB_SHA}`,
   - пушит `${GITHUB_SHA}` и `latest`,
   - выкатывает новую ревизию в Serverless Container.

## Локальная проверка API

```bash
docker build -t cbr-predictor:local .
docker run --rm -p 8080:8080 cbr-predictor:local
```

Проверка:

```bash
curl http://localhost:8080/health
```

## Примечание

Endpoint `/predict` требует готовых модельных артефактов в `data/processed/models`.  
В демонстрационном режиме используйте `/health` для проверки успешного деплоя контейнера.
