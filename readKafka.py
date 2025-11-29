from kafka import KafkaProducer, KafkaConsumer

consumer_toget = KafkaConsumer('topic_1', bootstrap_servers='localhost:9092')

for msg in consumer_toget:
    print(msg)
    print(msg.key.decode("utf-8") )
    s = msg.key.decode("utf-8")

    