# Docker Usage

## Build and Run with Docker Compose

1. Build and start both services:
   ```sh
   docker-compose -f docker/docker-compose.yml up --build
   ```

2. Access the API at [http://localhost:8000](http://localhost:8000)
   - Docs: [http://localhost:8000/docs](http://localhost:8000/docs)

3. Access the dashboard at [http://localhost:8501](http://localhost:8501)

## Build and Run Individually

### API
```sh
docker build -f docker/Dockerfile.api -t portfolio-risk-api .
docker run -p 8000:8000 portfolio-risk-api
```

### Dashboard
```sh
docker build -f docker/Dockerfile.dashboard -t portfolio-risk-dashboard .
docker run -p 8501:8501 portfolio-risk-dashboard
``` 