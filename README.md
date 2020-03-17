# kassandra


## Run Dockerfile

### 1. Install Extended Isolation Forest
```python
pip install git+https://github.com/albact7/eif.git
```

### 2. Build

```bash
docker image build -t kassandra .
```

### 3. Run

```bash
docker run -p kassandra
```
