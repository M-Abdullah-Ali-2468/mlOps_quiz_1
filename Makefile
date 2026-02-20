# Install dependencies
install:
	pip install -r requirements.txt

# Data preprocessing
preprocess:
	python src/preprocess_data.py

# Model training
train:
	python src/train.py

# Model evaluation
evaluate:
	python src/evaluate.py

# Run full pipeline
all: preprocess train evaluate