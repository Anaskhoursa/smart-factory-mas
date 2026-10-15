import paho.mqtt.client as mqtt


class MQTTClient:
    def __init__(
        self,
        client_id: str,
        host: str = "localhost",
        port: int = 1883,
    ):
        self.client_id = client_id
        self.host = host
        self.port = port

        self.client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2,
            client_id=client_id,
            protocol=mqtt.MQTTv5,
        )

        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message

    def _on_connect(self, client, userdata, flags, reason_code, properties):
        print(
            f"[MQTT] Connected as '{self.client_id}' "
            f"with reason: {reason_code}"
        )

    def _on_disconnect(self, client, userdata, flags, reason_code, properties):
        print(
            f"[MQTT] Disconnected '{self.client_id}' "
            f"with reason: {reason_code}"
        )

    def connect(self):
        self.client.connect(
            self.host,
            self.port,
            keepalive=60,
        )
        self.client.loop_start()

    def disconnect(self):
        self.client.loop_stop()
        self.client.disconnect()

    def publish(self, topic: str, payload: str, qos: int = 0):
        result = self.client.publish(
            topic,
            payload,
            qos=qos,
        )

        result.wait_for_publish()

        return result
    
    
    def subscribe(self, topic: str, qos: int = 0):
        result, mid = self.client.subscribe(topic, qos=qos)

        if result != mqtt.MQTT_ERR_SUCCESS:
            raise RuntimeError(
                f"Failed to subscribe to topic '{topic}'"
            )

        print(f"[MQTT] Subscribed to '{topic}'")
        
    def _on_message(
        self,
        client,
        userdata,
        message,
    ):
        payload = message.payload.decode("utf-8")

        print(
            f"[MQTT] Message received | "
            f"topic={message.topic} | "
            f"payload={payload}"
        )