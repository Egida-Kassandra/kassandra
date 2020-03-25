# kassandra


## Prepare environment

### Install Python 3 and pip

```bash
apt install -y python3 pip3 virtualenv
```

### Create virtualenv and install requirements

### Install Extended Isolation Forest

```python
virtualenv -p /usr/bin/python3 venv
source venv/bin/activate
pip install -r requirements.txt
pip install git+https://github.com/albact7/eif.git
```

## Run Dockerfile

### 1. Build

```bash
docker image build -t kassandra .
```

### 2. Run

```bash
docker run -p kassandra
```
