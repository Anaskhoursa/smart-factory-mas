import time

from messaging.client import MQTTClient


def main():
    client = MQTTClient(
        client_id="mqtt-subscriber",
    )

    client.connect()

    client.subscribe("factory/test")

    print("Waiting for messages...")

    try:
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nStopping subscriber...")

    finally:
        client.disconnect()


if __name__ == "__main__":
    main()