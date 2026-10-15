import time

from messaging.client import MQTTClient


def main():
    client = MQTTClient(
        client_id="mqtt-test-client",
    )

    client.connect()

    print("MQTT client connected.")

    time.sleep(2)

    client.disconnect()

    print("MQTT client disconnected.")


if __name__ == "__main__":
    main()