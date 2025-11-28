from kafka import KafkaProducer, KafkaConsumer

consumer_toget = KafkaConsumer('some_topic',bootstrap_servers='localhost:9092')

for msg in consumer_toget:
    print(msg)
    print(msg.key.decode("utf-8") )
    s = msg.key.decode("utf-8")

    