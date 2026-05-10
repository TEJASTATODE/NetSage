from fastapi import WebSocket

# =========================================================
# CONNECTION MANAGER
# =========================================================

class ConnectionManager:

    def __init__(self):

        self.active_connections = []

    # =====================================================
    # CONNECT
    # =====================================================

    async def connect(
        self,
        websocket: WebSocket
    ):

        await websocket.accept()

        self.active_connections.append(
            websocket
        )

    # =====================================================
    # DISCONNECT
    # =====================================================

    def disconnect(
        self,
        websocket: WebSocket
    ):

        if websocket in self.active_connections:

            self.active_connections.remove(
                websocket
            )

    # =====================================================
    # BROADCAST
    # =====================================================

    async def broadcast(
        self,
        message: dict
    ):

        disconnected = []

        for connection in self.active_connections:

            try:

                await connection.send_json(
                    message
                )

            except Exception:

                disconnected.append(
                    connection
                )

        # =================================================
        # REMOVE DEAD CONNECTIONS
        # =================================================

        for connection in disconnected:

            self.disconnect(
                connection
            )

# =========================================================
# GLOBAL MANAGER
# =========================================================

manager = ConnectionManager()