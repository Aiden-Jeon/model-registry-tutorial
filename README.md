# Model Registry Tutorial

tutorial code for model registry

## Model registry

1. Run model registry with docker compose command

    ```bash
    docker compose -f docker/model-registry/docker-compose.yaml up -d
    ```

2. Check whether you can access your model registry: [http://localhost:5001](http://localhost:5001)

## Train model

1. Install packages:

    ```bash
    pip install -r src/requirements.txt
    ```

2. Train model through script you can choose scikit-learn native model or pyfunc custom model

    - scikit-learn model

        ```bash
        python src/train.py
        ```

    - pyfunc custom model

        ```bash
        python src/train_custom_model.py
        ```

## API serving

1. Check your run id from model registry and download model.

    ```bash
    python docker/api-serving/download_model.py --run-id <YOUR_RUN_ID_HERE>
    ```

2. Build docker image.

    ```bash
    docker build ./docker/api-serving -f ./docker/api-serving/Dockerfile -t api-serving:latest
    ```

3. Run your image.

    ```bash
    docker run -p 8000:8000 api-serving:latest
    ```

4. Check your swagger: [http://localhost:8000](http://localhost:8000)
