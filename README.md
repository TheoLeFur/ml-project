# Introduction

<p> This repository contains the code for the project from the **Introduction to Machine Learning course** </p>

# Installation

<p> Start by cloning the repository using: </p>

```angular2html
git clone https://github.com/TheoLeFur/ml-project.git
```

<p> Create a virtual environment. For example, using Conda, write:</p>

```angular2html
conda create -n ml-project python==3.11
```

<p> Then activate your environment using: </p>

```angular2html
conda activate ml-project
```

<p>Finally, install the necessary dependencies by running: </p>

```angular2html
pip3 install -r requirements.txt
```

<p>from the<code>ml-project</code> directory</p>

# Running the models

<p> For running the models, from the <code>ml-project</code> directory, run the scripts</p>

```angular2html
/bin/bash scripts/run_simple.sh
```

for simple running,

```angular2html
/bin/bash scripts/run_with_cv.sh
```

for cross-validation,

```angular2html
/bin/bash scripts/run_with_cv_and_tune.sh
```

for cross-validation and parameters tuning. Make sure to replace the default arguments with desired values. 


