# kassandra

<!-- PROJECT SHIELDS -->
[![KASSANDRA VERSION](https://img.shields.io/badge/kassandra-v0.1-blue?style=for-the-badge&color=8B12D1)](https://github.com/albact7/kassandra)
[![GitHub license](https://img.shields.io/badge/license-Apache-blue?style=for-the-badge)](https://github.com/albact7/kassandra/blob/master/LICENSE)
[![GitHub release](https://img.shields.io/badge/release-v.0.0.1-yellowgreen?style=for-the-badge)](https://github.com/albact7/kassandra/releases)

<!-- PROJECT LOGO -->

<br />
<div align="center">
  <a href="https://github.com/albact7/kassandra">
    <img src="img/logo.svg" alt="Logo" width="180" height="180">
  </a>

  <p align="center">
    <br />
    <a href="https://github.com/albact7/kassandra"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/albact7/kassandra">View Source</a>
    ·
    <a href="https://github.com/albact7/kassandra/issues">Report Bug</a>
    ·
    <a href="https://github.com/albact7/kassandra/issues">Request Feature</a>
  </p>
</div>

<!-- TABLE OF CONTENTS -->
## Table of Contents

* [About the Project](#about-the-project)
* [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
* [License](#license)
* [Contact](#contact)

<!-- ABOUT THE PROJECT -->
## About the Project

Kassandra analyzes user activity and detects anomalous behaviour  in HTTP requests that could be identifies as non-malicious by other systems. Kassandra allows designing of anomaly detection policies.


### Prerequisites

#### Prepare environment

##### Install Python 3 and pip

```bash
apt install -y python3 pip3 virtualenv
```

### Installation

#### 1. Download the source from [here](https://github.com/albact7/kassandra/releases).
 
#### 2. Create virtualenv

```python
virtualenv -p /usr/bin/python3 venv
source venv/bin/activate
```
#### 3. Install requirements
Run install.bat

<!-- RUN KASSANDRA -->
## Getting started
To start running Kassandra 0.1 run the following on the root folder of the project.
```bash
python kassandra.py
```
This will run an example by default.
## Try on my own
### Needed files
To test Kassandra with you own files you should change [here](https://github.com/albact7/kassandra/blob/master/kassandra.py) the path to those files.
You will need:
1. Train file. Log file with a huge number (40000 is OK) of HTTP requests of a server.
2. Test file. Log file with some HTTP requests for testing.
### Designing of anomaly detection policies
You can also customize the anomaly values obtained by editing [config.yml](https://github.com/albact7/kassandra/blob/master/kass_nn/config/config.yml)
* Danger values are reserved to change the weigh for each characteristc 
* Extended Isolation Forests are reserved for adjust the Machine Learning model to the training data

<!-- RUNNING TESTS -->
## Running tests
To run any test run the following command, being "test_file" any of the files present on the root folder like "tst_level_*.py".

```bash
python filename
```
Each test file can be edited to run a different set of HTTP requests, modify the corresponding file name according to:
* Level 1 tests in [here](https://github.com/albact7/kassandra/tree/master/kass_nn/level_1/test_logs/main)
* Level 2 tests in [here](https://github.com/albact7/kassandra/tree/master/kass_nn/level_2/test_logs)

### Run Dockerfile

#### 1. Build

```bash
docker image build -t kassandra .
```

#### 2. Run

```bash
docker run -p kassandra
```



<!-- LICENSE -->
## License

Distributed under the Apache 2.0 License. See `LICENSE` for more information.

<!-- CONTACT -->
## Contact

Authors:

* [Alba Cotarelo Tuñón]
* [Antonio Payá González](https://antoniopg.tk)
* [Jose Manuel Redondo Lopez](http://orcid.org/0000-0002-0939-0186)

Project Link: [https://github.com/albact7/kassandra](https://github.com/albact7/kassandra)
