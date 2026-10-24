from collections.abc import Callable
import paho.mqtt.client as mqtt


class MQTTClient:
    def __init__(
        self,
        client_id: str,
        host: str = "localhost",
        port: int = 1883,
        on_message: Callable | None = None,
    ):
        self.client_id = client_id
        self.host = host
        self.port = port
        self.message_handler = on_message

        self.client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2,
            client_id=client_id,
            protocol=mqtt.MQTTv5,
        )

        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message

    def _on_connect(self, client, userdata, flags, reason_code, properties):
        print(f"[MQTT] Connected as {self.client_id}: {reason_code}")

    def _on_disconnect(self, client, userdata, flags, reason_code, properties):
        print(f"[MQTT] Disconnected {self.client_id}: {reason_code}")

    def connect(self):
        self.client.connect(self.host, self.port, keepalive=60)
        self.client.loop_start()

    def disconnect(self):
        self.client.disconnect()
        self.client.loop_stop()

    def publish(self, topic: str, payload: str, qos: int = 0):
        info = self.client.publish(topic, payload, qos=qos)
        info.wait_for_publish()
        return info

    def subscribe(self, topic: str, qos: int = 0):
        result, _ = self.client.subscribe(topic, qos=qos)
        if result != mqtt.MQTT_ERR_SUCCESS:
            raise RuntimeError(f"Failed to subscribe to {topic!r}")

    def _on_message(self, client, userdata, message):
        if self.message_handler is not None:
            self.message_handler(
                message.topic,
                message.payload,
                message.qos,
            )
        else:
            print(
                f"[MQTT] {message.topic}: "
                f"{message.payload.decode('utf-8', errors='replace')}"
            )
