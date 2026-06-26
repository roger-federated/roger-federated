"""`python -m roger.federated.server` — serve the app with uvicorn.

Host/port from ROGER_SERVER_HOST/PORT (default 0.0.0.0:8000). TLS is terminated by Caddy in front
(see Caddyfile/DEPLOY.md); this process speaks plain HTTP on the loopback/proxy network. All
aggregation knobs are env vars read in app.create_app.
"""
import os

import uvicorn

from roger.federated.server.app import create_app

if __name__ == "__main__":
    uvicorn.run(create_app(),
                host=os.environ.get("ROGER_SERVER_HOST", "0.0.0.0"),
                port=int(os.environ.get("ROGER_SERVER_PORT", "8000")))
