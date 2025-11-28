docker build -t backend-llm .

docker compose up -d

python3 ./backend/main.py

pip install confluent-kafka

pip install kafka-python