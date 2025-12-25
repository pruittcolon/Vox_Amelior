import json
import time

import pandas as pd
from confluent_kafka import Consumer, KafkaError


class KafkaLoader:
    """
    Universal Kafka Loader.
    Consumes messages from a Kafka topic and converts them to a pandas DataFrame.
    """

    def __init__(self, bootstrap_servers: str, group_id: str = "ml-agent-group", auto_offset_reset: str = "earliest"):
        """
        Initialize Kafka Consumer.
        """
        self.bootstrap_servers = bootstrap_servers
        self.group_id = group_id
        self.auto_offset_reset = auto_offset_reset
        self.consumer_config = {
            "bootstrap.servers": bootstrap_servers,
            "group.id": group_id,
            "auto.offset.reset": auto_offset_reset,
            "enable.auto.commit": False,  # Manual control or just reading
        }

    def test_connection(self) -> bool:
        """
        Verify connection by listing topics.
        """
        try:
            # Create a temporary consumer to check connection
            consumer = Consumer(self.consumer_config)
            cluster_meta = consumer.list_topics(timeout=10)
            consumer.close()
            return cluster_meta is not None
        except Exception as e:
            print(f"Connection failed: {e}")
            return False

    def consume_messages(self, topic: str, max_messages: int = 100, timeout: float = 10.0) -> pd.DataFrame:
        """
        Consume messages from a topic.

        :param topic: Kafka topic to consume from.
        :param max_messages: Maximum number of messages to read.
        :param timeout: Total time to wait for messages (seconds).
        """
        consumer = Consumer(self.consumer_config)
        consumer.subscribe([topic])

        messages = []
        start_time = time.time()

        try:
            while len(messages) < max_messages:
                # Check timeout
                if time.time() - start_time > timeout:
                    break

                # Poll for message
                msg = consumer.poll(1.0)

                if msg is None:
                    continue
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        continue
                    else:
                        print(f"Kafka Error: {msg.error()}")
                        break

                # Process message
                try:
                    val = msg.value().decode("utf-8")
                    # Try parsing JSON
                    try:
                        data = json.loads(val)
                    except json.JSONDecodeError:
                        # Fallback to raw string
                        data = {"raw_value": val}

                    messages.append(data)
                except Exception as e:
                    print(f"Error decoding message: {e}")

        finally:
            consumer.close()

        return pd.DataFrame(messages)
