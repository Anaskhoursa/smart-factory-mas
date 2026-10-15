import time

from messaging.client import MQTTClient


def main():
    client = MQTTClient(
        client_id="mqtt-publisher",
    )

    client.connect()

    time.sleep(1)

    client.publish(
        "factory/test",
        "Hello from the smart factory!",
    )

    print("Message published.")

    time.sleep(1)

    client.disconnect()


if __name__ == "__main__":
    main()